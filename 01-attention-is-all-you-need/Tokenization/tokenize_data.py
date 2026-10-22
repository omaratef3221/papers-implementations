from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('en-de-tokenizer')

def tokenize_data(data):
    inputs = [ex["en"].lower() for ex in data["translation"]]
    targets_de = [ex["de"].lower() for ex in data["translation"]]

    model_inputs = {}

    model_inputs['inputs'] = tokenizer(inputs,
                    max_length = 100,
                    truncation = True,
                    return_tensors = 'pt'
                    )["input_ids"]

    model_inputs['labels'] = tokenizer(targets_de,
                max_length = 100,
                truncation = True,
                return_tensors = 'pt'
                )["input_ids"]
    return model_inputs
