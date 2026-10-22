from tqdm import tqdm

def trainer(epochs, model, optimizer, scheduler, train_dataloader, validation_dataloader, loss_fn, device, pad_idx):
    model.train()

    total_loss_train_plot = []
    total_loss_validation_plot = []
    total_acc_train_plot = []
    total_acc_validation_plot = []

    for epoch in range(epochs):
        total_train_loss, total_val_loss, total_train_acc, total_val_acc = 0, 0, 0, 0

        for data in tqdm(train_dataloader):
            src, tgt = data

            src = src.to(device)
            tgt = tgt.to(device)

            decoder_input = tgt[:, :-1]
            labels = tgt[:, 1:]

            src_mask = src != pad_idx


            break
        break

    print(f"Epoch {epoch}")

    return 0

