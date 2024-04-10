import argparse

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin, LowLevelZeroPlugin, GeminiPlugin
from colossalai.cluster import DistCoordinator
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam

from dataset import OxfordPetDataset, oxford_pet_collator
import time
import torch
import tqdm
import transformers
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

def move_to_cuda(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

def run_forward_backward(
    model, optimizer, criterion,
    data_iter, booster,
):
    if optimizer is not None:
        optimizer.zero_grad()
        
    batch = next(data_iter)
    batch = move_to_cuda(batch, torch.cuda.current_device())
    outputs = model(**batch)
    loss = criterion(outputs, None)
    if optimizer is not None:
        booster.backward(loss, optimizer)
        
    return loss, outputs

def train(args, model, train_dataloader, eval_dataloader, optimizer, criterion, lr_scheduler, booster, logger):    
    logger.info(f"Start finetuning", ranks=[0])
    for epoch in range(args.num_epoch):
        model.train()
        train_data_iter = iter(train_dataloader)
        with tqdm.tqdm(range(len(train_dataloader)), desc=f"Epoch [{epoch + 1}]") as pbar:
            for _ in pbar:
                loss, _ = run_forward_backward(model, optimizer, criterion, train_data_iter, booster)
                optimizer.step()
                lr_scheduler.step()

                pbar.set_postfix({"loss": loss.item()})
        
        with torch.no_grad():
            model.eval()
            accum_loss = torch.zeros(1, device=torch.cuda.current_device())
            total_num = torch.zeros(1, device=torch.cuda.current_device())
            accum_correct = torch.zeros(1, device=torch.cuda.current_device())

            for batch in eval_dataloader:
                batch = move_to_cuda(batch, torch.cuda.current_device())
                loss, outputs = run_forward_backward(model, None, criterion, iter([batch]), booster)

                accum_loss += loss / len(eval_dataloader)
                logits = outputs["logits"]
                preds = torch.argmax(logits, dim=1)

                labels = batch["labels"]
                total_num += batch["labels"].shape[0]
                accum_correct += torch.sum(preds == labels)

            # NOTE: Collect loss, correctness, and total number from all GPUs when using Data Parallelism.
            avg_loss = "{:.4f}".format(accum_loss.item())
            accuracy = "{:.4f}".format(accum_correct.item() / total_num.item())
            print(
                f"Evaluation result for epoch {epoch + 1}: \
                    average_loss={avg_loss}, \
                    accuracy={accuracy}."
            )
            
    logger.info(f"Finish finetuning", ranks=[0])
    
    booster.save_model(model, args.output_path, shard=True)
    logger.info(f"Saving model checkpoint to {args.output_path}", ranks=[0])
    
def benchmark(args, model, train_dataloader, optimizer, criterion, lr_scheduler, booster, coordinator, logger):
    # What happens if the memory cap is set for different plugins?
    if args.mem_cap > 0:
        from colossalai.utils import colo_device_memory_capacity, colo_set_process_memory_fraction
        cuda_capacity = colo_device_memory_capacity(get_accelerator().get_current_device())
        if args.mem_cap * (1024**3) < cuda_capacity:
            colo_set_process_memory_fraction(args.mem_cap * (1024**3) / cuda_capacity)
            logger.info(f"Set memory cap as {args.mem_cap} GB", ranks=[0])
    
    logger.info(f"Start benchmarking", ranks=[0])
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    model.train()
    train_data_iter = iter(train_dataloader)
    
    with tqdm(range(args.max_train_steps), desc="Benchmarking") as pbar:
        for _ in pbar:
            _, _ = run_forward_backward(model, optimizer, criterion, train_data_iter, booster)
            optimizer.step()
            lr_scheduler.step()

            torch.cuda.synchronize()

    end_time = time.time()
    throughput = "{:.4f}".format((coordinator.world_size * args.max_train_steps * args.batch_size) / (end_time - start_time))
    max_mem = format_num(torch.cuda.max_memory_allocated(device=torch.cuda.current_device()))

    logger.info(
        f"Benchmarking finished: "
        f"- Batch size per gpu: {args.batch_size} "
        f"- Plugin: {args.plugin} "
        f"- Throughput: {throughput} "
        f"- Maximum memory usage per gpu: {max_mem}.",
        ranks=[0],
    )

    torch.cuda.empty_cache()
    

def main():
    args = parse_run()
    
    colossalai.launch_from_torch(config={}, seed=42)
    coordinator = DistCoordinator()
    
    disable_existing_loggers()
    logger = get_dist_logger()
    if coordinator.is_master():
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
    
    image_processor = ViTImageProcessor.from_pretrained(args.model_name)
    train_dataset = OxfordPetDataset(image_processor, split="train")
    eval_dataset = OxfordPetDataset(image_processor, split="test")
    
    config = ViTConfig.from_pretrained(args.model_name)
    config.num_labels = len(train_dataset.unique_labels)
    config.id2label = {str(i): c for i, c in enumerate(train_dataset.unique_labels)}
    config.label2id = {c: str(i) for i, c in enumerate(train_dataset.unique_labels)}
    model = ViTForImageClassification.from_pretrained(
        args.model_name, config=config, ignore_mismatched_sizes=True
    )
    
    if args.grad_checkpoint > 0:
        model.gradient_checkpointing_enable()
    
    booster_kwargs = {}
    if args.plugin == "torch_ddp_fp16":
        booster_kwargs["mixed_precision"] = "fp16"
    
    if args.plugin.startswith("torch_ddp"):
        plugin = TorchDDPPlugin()
    elif args.plugin == "gemini":
        # offload_optim_frac=1.0 means that all the optimizer state is offloaded to the CPU.
        # pin_memory=True means that model state and activations reside in an area of physical memory
        #   that the operating system will not swap to the disk,
        #   which allows faster data transfer to the GPU.
        plugin = GeminiPlugin(offload_optim_frac=1.0, pin_memory=True, initial_scale=2**5)
    elif args.plugin == "low_level_zero":
        plugin = LowLevelZeroPlugin(initial_scale=2**5)
    else:
        raise ValueError(f"Plugin with name {args.plugin} is not supported!")
    logger.info(f"Set plugin as {args.plugin}", ranks=[0])
    
    train_dataloader = plugin.prepare_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True, collate_fn=oxford_pet_collator
    )
    eval_dataloader = plugin.prepare_dataloader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        drop_last=True, collate_fn=oxford_pet_collator
    )
    
    # Linear Scaling Rule suggests that when
    # the batch size is multiplied by a factor of k,
    # the learning rate should also be multiplied by k
    # to keep the variance of the parameter updates consistent.
    optimizer = HybridAdam(model.parameters(),
                           lr=(args.learning_rate * coordinator.world_size),
                           weight_decay=args.weight_decay)
    
    total_steps = len(train_dataloader) * args.num_epoch
    num_warmup_steps = int(args.warmup_ratio * total_steps)
    lr_scheduler = CosineAnnealingWarmupLR(
        optimizer=optimizer, total_steps=total_steps, warmup_steps=num_warmup_steps
    )
    
    def criterion(outputs, inputs):
        return outputs.loss
    
    booster = Booster(plugin=plugin, **booster_kwargs)
    model, optimizer, _criterion, train_dataloader, lr_scheduler = booster.boost(
        model=model, optimizer=optimizer, criterion=criterion,
        dataloader=train_dataloader, lr_scheduler=lr_scheduler
    )
    
    if args.running_mode == "train":
        train(args, model, train_dataloader, eval_dataloader, optimizer, criterion, lr_scheduler, booster, logger)
    elif args.running_mode == "benchmark":
        benchmark(args, model, train_dataloader, optimizer, criterion, lr_scheduler, booster, coordinator, logger)
    else:
        raise ValueError(f"Running mode with name {args.running_mode} is not supported!")
    
def parse_run():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--running_mode",
        type=str,
        default="train",
        help="Valid running modes include 'train' and 'benchmark'.",
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/vit-base-patch16-224",
        help="Path to the pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_path", type=str, default="./output_model", help="The path for your saved model after fine-tuning."
    )
    parser.add_argument(
        "--plugin",
        type=str,
        default="gemini",
        help="Plugin to use. Valid plugins include 'torch_ddp', 'torch_ddp_fp16', 'gemini', and 'low_level_zero'.",
    )
    
    parser.add_argument("--num_epoch", type=int, default=3, help="Number of epochs.")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size (per DP group) for the training dataloader."
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Initial learning rate (after any potential warmup period) to use.",
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.3, help="Ratio of warmup steps to total training steps."
    )
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay rate to use.")
    parser.add_argument("--grad_checkpoint", type=bool, default=True, help="Whether to use gradient checkpointing.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")

    # Benchmarking arguments
    parser.add_argument("--mem_cap", type=int, default=0, help="A limit is set on the memory space which each GPU can use in gigabytes (GB).")
    parser.add_argument("--max_train_steps", type=int, default=20, help="The total number of training steps to be benchmarked.")
    
    args = parser.parse_args()
    return args

def format_num(num: int):
    factor = 1024
    suffix = "B"
    for unit in ["", " K", " M", " G", " T", " P"]:
        if num < factor:
            return f"{num:.2f}{unit}{suffix}"
        num /= factor

    
if __name__ == "__main__":
    main()
