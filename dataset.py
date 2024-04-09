from datasets import load_dataset
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class OxfordPetDataset(Dataset):
    def __init__(self, image_processor, split='train'):
        super().__init__()
        
        self.image_processor = image_processor
        
        self.dataset = load_dataset('timm/oxford-iiit-pet', split=split)
        
        # NOTE: Pad the labels when using Tensor Parallelism
        # to ensure the number of labels is divisible by the number of GPUs.
        self.unique_labels = self.dataset.features['label'].names
        
        self.samples = []
        for sample in tqdm(self.dataset, desc="Processing images..."):
            try:
                processed_sample = self.transform_image(sample)
                self.samples.append(processed_sample)
            except:
                print(f"Skipping an image due to error!")
    
    def transform_image(self, sample):
        input = self.image_processor(sample['image'], return_tensors='pt')        
        input['label'] = sample['label']
        return input
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def oxford_pet_collator(batch):
    return {
        "pixel_values": torch.cat([data["pixel_values"] for data in batch], dim=0),
        'labels': torch.tensor([data["label"] for data in batch], dtype=torch.int64),
    }