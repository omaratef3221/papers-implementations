import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import PreTrainedTokenizerFast


def create_tokenizer(text_data: list, vocab_size: int, output_path: str, transformers = True):
    os.makedirs(output_path, exist_ok=True)
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]

    trainer = BpeTrainer(
        vocab_size = vocab_size,
        special_tokens = special_tokens,
        initial_alphabet = ByteLevel.alphabet()
    )
    tokenizer.train_from_iterator(text_data, trainer)
    # tokenizer.save(os.path.join( output_path, 'tokenizer.json' ))
    if transformers:
        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file = os.path.join( output_path, 'tokenizer.json' ),
            padding_side = 'right',
            truncation_side = 'right',
            unk_token= special_tokens[0],
            pad_token = special_tokens[1],
            bos_token = special_tokens[2],
            eos_token = special_tokens[3],
            
        )
        wrapped_tokenizer.save_pretrained(os.path.join(output_path))

    return f"Tokenizer trained and saved at {os.path.join(output_path)}.json"
