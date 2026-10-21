import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


def create_tokenizer(text_data: list, vocab_size: int, output_path: str):
    os.makedirs(output_path, exist_ok=True)
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size = vocab_size,
        special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"],
        initial_alphabet = ByteLevel.alphabet()
    )
    tokenizer.train_from_iterator(text_data, trainer)
    tokenizer.save(os.path.join( output_path, 'tokenizer.json' ))

    return f"Tokenizer trained and saved at {os.path.join(output_path, 'tokenizer.json')}.json"
