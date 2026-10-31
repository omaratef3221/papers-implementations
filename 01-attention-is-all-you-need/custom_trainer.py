from tqdm import tqdm
import torch
import wandb
import os

def trainer(epochs, model, optimizer, scheduler, train_dataloader, validation_dataloader, loss_fn, device, pad_idx, MAX_LEN, MODEL_PATH = 'model_artifacts_25p'):
    os.makedirs(MODEL_PATH, exist_ok=True)
    for epoch in range(epochs):
        total_train_loss, total_val_loss, total_train_acc, total_val_acc, total_train_tokens, total_val_tokens = 0, 0, 0, 0, 0, 0

        model.train()
        for data in train_dataloader:
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

            total_train_loss += batch_loss.item()

            total_train_acc += ((preds == labels) & (labels != pad_idx)).sum().item()
            total_train_tokens += (labels != pad_idx).sum().item()
            wandb.log({"step/loss": batch_loss.item(), "step/lr": scheduler.get_last_lr()[0]})

        model.eval()
        for data in validation_dataloader:
            
            with torch.no_grad():
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

                total_val_loss += batch_loss.item()

                total_val_acc += ((preds == labels) & (labels != pad_idx)).sum().item()
                total_val_tokens += (labels != pad_idx).sum().item()
            
        avg_train_loss = total_train_loss/len(train_dataloader)
        avg_train_acc = total_train_acc / total_train_tokens

        avg_val_loss = total_val_loss/len(validation_dataloader)
        avg_val_acc = total_val_acc / total_val_tokens

        wandb.log({
            "train/loss": avg_train_loss,
            "train/accuracy": avg_train_acc * 100,
            "validation/loss": avg_val_loss,
            "validation/accuracy": avg_val_acc * 100,
        })

        print(f"Epoch {epoch} with Training Accuracy of {avg_train_acc * 100} and Training Loss of: {avg_train_loss} Validation Accuracy {avg_val_acc * 100} Validation Loss {avg_val_loss}")

        torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            }, f"{MODEL_PATH}/last.pth")

    return 0

