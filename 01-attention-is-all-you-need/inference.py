import torch
from Transformer import Transformer
import torch
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

def invoke_model(model, tokenizer, input_text, config):
    VOCAB_SIZE= config['VOCAB_SIZE']
    D_MODEL = config['D_MODEL']
    H = config['H']
    MAX_LEN = config['MAX_LEN']
    N_Enc = config['N_Enc']
    N_Dec = config['N_Dec']
    DROPOUT = config['DROPOUT']

    tokenizer = AutoTokenizer.from_pretrained(tokenizer)


    custom_transformer = Transformer(D_MODEL,
                                    H,
                                    VOCAB_SIZE,
                                    MAX_LEN,
                                    N_Enc,
                                    N_Dec, 
                                    DROPOUT)

    ckpt = torch.load(model, map_location='cpu', weights_only=False)
    custom_transformer.load_state_dict(ckpt["model"])
    custom_transformer.eval()

    input_ids = tokenizer(
        [t.lower() for t in input_text],
        max_length = MAX_LEN,
        truncation = True,
        )['input_ids']

    input_ids = [torch.tensor([2] + ids + [3]) for ids in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
    src_mask = input_ids != 1
    src_mask = src_mask.unsqueeze(1).unsqueeze(1)

    with torch.no_grad():
        dec = torch.full((input_ids.shape[0], 1), 2)        # start token
        for token in range(MAX_LEN):
            T = dec.shape[1]
            tgt_mask = torch.tril(torch.ones(T, T)).bool()      # (T, T) broadcasts fine
            logits = custom_transformer(input_ids, dec, src_mask, tgt_mask)
            next_tok = logits[:, -1, :].argmax(-1)            # highest in vocab (greedy)
            dec = torch.cat([dec, next_tok.unsqueeze(1)], dim=1)     # shape (3,) → (3, 1)
            if (dec == 3).any(dim=1).all():
                break

    outputs= []
    for row in dec[:, 1:]:
        ids = row.tolist()
        if 3 in ids:
            ids = ids[:ids.index(3)]
            outputs.append(tokenizer.decode(ids))
    return outputs