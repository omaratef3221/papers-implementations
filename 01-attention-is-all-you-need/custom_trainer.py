from tqdm import tqdm
import torch

def trainer(epochs, model, optimizer, scheduler, train_dataloader, validation_dataloader, loss_fn, device, pad_idx, MAX_LEN):
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
            src_mask = src_mask.unsqueeze(1).unsqueeze(1)

            tgt_mask = decoder_input != pad_idx
            tgt_mask = tgt_mask.unsqueeze(1).unsqueeze(1)

            causal_masking = torch.tril(torch.ones(decoder_input.shape[1], decoder_input.shape[1], device=device), diagonal=0).bool()
            tgt_mask = tgt_mask & causal_masking

            logits = model(src, decoder_input, src_mask, tgt_mask)

            preds = torch.argmax(logits, -1)

            batch_loss = loss_fn(
                torch.reshape(logits, (logits.shape[0] * logits.shape[1], logits.shape[-1])),
                torch.reshape(labels, ((labels.shape[0] * labels.shape[1]),)),
                )

            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()    
            scheduler.step()
            print(batch_loss.item())
            break
        break

    print(f"Epoch {epoch}")

    return 0

