import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from datasets import load_dataset
from transformers import GPT2Tokenizer
import os

class TextDataset(Dataset):
    """Simple text dataset for language modeling"""
    
    def __init__(self, data, tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]['text']
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze(0)
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }

def get_dataloader(config, rank=0, world_size=1):
    """Create distributed dataloader"""
    
    dataset_name = config['data']['dataset_name']
    dataset_config = config['data'].get('dataset_config', None)
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    
    if dataset_config:
        print(f"  Config: {dataset_config}")
        dataset = load_dataset(
            dataset_name,
            dataset_config,
            split='train',
            streaming=False
        )
    else:
        dataset = load_dataset(
            dataset_name,
            split='train',
            streaming=False
        )
    
    # Take only a subset for quick testing if specified
    if 'max_samples' in config['data']:
        max_samples = config['data']['max_samples']
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"  Using {len(dataset)} samples for testing")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    train_dataset = TextDataset(
        dataset,
        tokenizer,
        max_length=config['data']['max_seq_len']
    )
    
    # Create sampler for distributed training
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
    
    # Create dataloader
    dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['micro_batch_size'],
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    return dataloader, tokenizer
