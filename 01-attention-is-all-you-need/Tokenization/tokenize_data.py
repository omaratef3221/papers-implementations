from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('en-de-tokenizer')

def tokenize_data(data):
    return tokenizer(data['en'],
                    max_length = 100,
                    truncation = True,
                    padding = "max_length",
                    return_tensors = 'pt'
                    ),
    tokenizer(data['de'],
                max_length = 100,
                truncation = True,
                padding = "max_length",
                return_tensors = 'pt'
                )
