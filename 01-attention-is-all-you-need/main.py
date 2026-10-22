from prepare_data import get_dataset, CustomDataset
from Transformer import Transformer
from custom_trainer import trainer
import torch.nn as nn
from torch.optim import Adam, lr_scheduler

### Variables
VOCAB_SIZE= 37000
D_MODEL = 512
H = 8
MAX_LEN = 100
N_Enc = 6
N_Dec = 6
BATCH_SIZE = 64 # -----  The paper is too much for my macbook (32GB unified memory)
EPOCHS = 25 # Smaller than the paper as it is also too much
EPS = 1e-9
WARMUP_STEPS = 4000

DEVICE = 'mps'

### Get, process dataset
dataset = get_dataset('de-en', split=True)

training_data = CustomDataset(dataset['train'])
validation_data = CustomDataset(dataset['validation'])
test_data = CustomDataset(dataset['test'])

### Build Transformer, Optimizer and Loss
custom_transformer = Transformer(D_MODEL,
                                 H,
                                 VOCAB_SIZE,
                                 MAX_LEN,
                                 N_Enc,
                                 N_Dec)

for param in custom_transformer.parameters():
    param.requires_grad = True

custom_transformer.to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = Adam(custom_transformer.parameters(), 
                 eps = EPS,
                 betas=(0.9, 0.98),
                 )

lr = lr_scheduler.LambdaLR(
    optimizer=optimizer,
    
                           )


