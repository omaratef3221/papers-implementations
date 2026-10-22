from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('en-de-tokenizer')

def tokenize_data(data):
    inputs = [ex["en"].lower() for ex in data["translation"]]
    targets_de = [ex["de"].lower() for ex in data["translation"]]

    src_ids = tokenizer(inputs, max_length=48, truncation=True)["input_ids"]
    tgt_ids = tokenizer(targets_de, max_length=48, truncation=True)["input_ids"]

    return {
        'inputs': [[2] + ids + [3] for ids in src_ids],
        'labels': [[2] + ids + [3] for ids in tgt_ids],
    }