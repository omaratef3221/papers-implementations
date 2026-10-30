from prepare_data import get_dataset, CustomDataset
from Transformer import Transformer
from custom_trainer import trainer
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import wandb
from dotenv import load_dotenv

load_dotenv()


### Variables
VOCAB_SIZE= 37000
D_MODEL = 512
H = 8
MAX_LEN = 100
N_Enc = 6
N_Dec = 6
BATCH_SIZE = 64 # ----- The paper is too much for my macbook (32GB unified memory)
EPOCHS = 20 # ---- ~ to the paper as it is also too much
EPS = 1e-9
WARMUP_STEPS = 4000
DROPOUT = 0.1

DEVICE = 'mps'

wandb.init(
    project="paper 1 - Attention is all you need implementation",
    config={
        "VOCAB_SIZE": VOCAB_SIZE,
        "D_MODEL": D_MODEL,
        "H": H,
        "N_Enc": N_Enc,
        "N_Dec": N_Dec,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "EPS": EPS,
        "WARMUP_STEPS": WARMUP_STEPS,
        "DROPOUT":DROPOUT
    }
)


def lr_lambda(step, d_model = D_MODEL, warmup_steps = WARMUP_STEPS):
    step = max(step, 1)
    return d_model**-0.5 * np.min((step**-0.5, step*(warmup_steps**-1.5)))

def collate_fn(batch, pad_idx = 1):
    src, tgt = zip(*batch)
    src = [torch.tensor(s) for s in src]
    tgt = [torch.tensor(t) for t in tgt]

    src = pad_sequence(src, batch_first=True, padding_value=pad_idx)
    tgt = pad_sequence(tgt, batch_first=True, padding_value=pad_idx)

    return src, tgt

### Get, process dataset
dataset = get_dataset('de-en', split=True)

training_data = CustomDataset(dataset['train'])
validation_data = CustomDataset(dataset['validation'])
test_data = CustomDataset(dataset['test'])


train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle= True, collate_fn=collate_fn)
validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)
testing_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)

### Build Transformer, Optimizer and Loss
custom_transformer = Transformer(D_MODEL,
                                 H,
                                 VOCAB_SIZE,
                                 MAX_LEN,
                                 N_Enc,
                                 N_Dec, 
                                 DROPOUT)


model_total_params = sum(p.numel() for p in custom_transformer.parameters() if p.requires_grad)
print(f"Trainable parameters: {model_total_params / 1e6:.2f}M  ({model_total_params:,})")

custom_transformer.to(DEVICE)

loss = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=1)

optimizer = Adam(custom_transformer.parameters(), 
                 eps = EPS,
                 betas=(0.9, 0.98),
                 lr = 1
                 )

scheduler = lr_scheduler.LambdaLR(
    optimizer=optimizer,
    lr_lambda=lr_lambda
    )


## Train Loop

trainer(10, custom_transformer, optimizer, scheduler, train_dataloader, validation_dataloader, loss, DEVICE, 1, MAX_LEN)