from torch.utils.data import Dataset, DataLoader
from Tokenization.tokenize_data import tokenize_data

import datasets

def get_dataset(variation = 'de-en', split = True):
    if split:
        split_amt = {"train": "train[:25%]","validation": "validation[:25%]","test": "test[:25%]"}
    else:
        split_amt = None
    dataset = datasets.load_dataset('wmt/wmt14', variation,
                            split= split_amt
                                )
    return dataset
    
    

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset.map(tokenize_data, batched= True, load_from_cache_file=False)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        return row['inputs'], row['labels']