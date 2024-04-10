#### Fine-tune ViT with ColossalAI

**GitHub repository**

https://github.com/trieu-le-arner/ViT_Fine-tuning_with_ColossalAI

**Model: ViT-base-patch16-224 with pretrained weights loaded from HuggingFace**

https://huggingface.co/google/vit-base-patch16-224

**Dataset**

The dataset comprises 37 pet categories, each represented by approximately 200 images. These images display significant variability in scale, pose, and lighting conditions. The dataset is divided into two parts: 3,680 images are designated for training, while 3,669 images are set aside for validation purposes.

https://huggingface.co/datasets/timm/oxford-iiit-pet

**Parallel settings (if any)**

The experiments were conducted on Colab using a single GPU. They explored the impact of various ColossalAI plugins and batch sizes on throughput and GPU memory usage.

**Instructions**

1. Upload the notebook namely "Colab_ViT_Fine_tuning_Colossal_AI.ipynb" to Google Colab
2. Choose T4 GPU runtime type (1 GPU)
3. Run all the cells in the notebook

**Experiment results on a single GPU**

| Plugin              | Accuracy | Wall time per epoch (seconds) |
| ------------------- | -------- | ----------------------------- |
| torch_ddp           | 0.9113   | 220                           |
| torch_ddp with fp16 | 0.9132   | 76                            |
| low_level_zero      | 0.9094   | 78                            |
| gemini              | 0.9080   | 108                           |

All plugins achieved roughly 90% accuracy after 3 epochs. However, Torch DDP was significantly slower than the others, being twice as slow as Gemini and three times as slow as Torch DDP with FP16 and Low Level Zero. This discrepancy may be attributed to Torch DDP's use of FP32 precision, whereas the other three plugins explicitly or by default utilize FP16 precision.

| Plugin         | Batch size | Throughput  | Maximum memory usage |
| -------------- | ---------- | ----------- | -------------------- |
| torch_ddp      | 8          | 23.6198     | 1.73 GB              |
| torch_ddp_fp16 | 8          | 56.8228     | 1.72 GB              |
| low_level_zero | 8          | 42.0017     | 1.64 GB              |
| gemini         | 8          | 22.2007     | **663.05 MB**        |
| torch_ddp      | 32         | 25.3715     | 2.10 GB              |
| torch_ddp_fp16 | 32         | **76.1189** | 2.01 GB              |
| low_level_zero | 32         | 74.4609     | 1.64 GB              |
| gemini         | 32         | 58.5107     | **663.05 MB**        |

The Gemini plugin has the **least** maximum GPU memory usage, maintaining constant consumption as batch sizes increase from 8 to 32, thanks to its static placement strategy. This strategy predetermines the temporary offloading of model parameters, gradients, or intermediate activations to CPU memory or disk, effectively reducing the GPU memory footprint. The parameters for Gemini, highlighting its efficient GPU memory usage, include `offload_optim_frac=1.0`, which offloads all optimizer states to the CPU, and `pin_memory=True`, ensuring model state and activations are in a non-swappable section of physical memory, speeding up transfers to the GPU.

| Plugin         | Batch size | Throughput  | Maximum memory usage | CPU offload |
| -------------- | ---------- | ----------- | -------------------- | ----------- |
| low_level_zero | 8          | 42.1272     | 1.66 GB              | False       |
| low_level_zero | 32         | **74.8434** | 1.67 GB              | False       |
| low_level_zero | 32         | 32.5331     | **625.86 MB**        | True        |

Increasing the batch size typically requires more GPU memory for activations. However, with Low Level Zero, even without CPU offloading, GPU memory consumption remains relatively constant while throughput increases, thanks to techniques like activation checkpointing or offloading activations to CPU. This is discussed in the "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" paper. For a batch size of 32, activating CPU offload for other model state data reduces maximum GPU memory usage by 2.5 times, from 1.67 GB to 625.86 MB, but throughput is halved due to the latency from accessing CPU memory and disk.
