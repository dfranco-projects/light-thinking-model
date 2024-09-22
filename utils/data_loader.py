import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

class TextDataset(Dataset):
    def __init__(self, dataset_name: str, tokenizer_name: str, max_length: int):
        '''
        Initialize the TextDataset.

        Args:
            dataset_name (str): Name of the dataset to load.
            tokenizer_name (str): Name of the tokenizer to use.
            max_length (int): Maximum length of the tokenized input.
        '''
        self.dataset = load_dataset(dataset_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset['train'])

    def __getitem__(self, idx):
        item = self.dataset['train'][idx]
        inputs = self.tokenizer(item['text'], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze()

def create_dataloader(dataset_name: str, max_length: int, batch_size: int, tokenizer_name: str = 'gpt2') -> DataLoader:
    '''
    Create a DataLoader for the text dataset.

    Args:
        dataset_name (str): Name of the dataset to load.
        max_length (int): Maximum length of the tokenized input.
        batch_size (int): Size of the batches.
        tokenizer_name (str): Name of the tokenizer to use (default is 'gpt2').

    Returns:
        DataLoader: DataLoader for the text dataset.
    '''
    dataset = TextDataset(dataset_name, tokenizer_name, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


