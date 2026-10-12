import torch.nn as nn 

from Layers.Embeddings import EmbeddingBlock
from Layers.PositionalEncoding import PositionalEncoding
from Layers.Projection import Projection

from Encoder import EncoderBlock
from Decoder import DecoderBlock



class Transformer(nn.Module):
    def __init__(self, D_MODEL, H, VOCAB_SIZE, MAX_LEN, N_Enc, N_Dec):
        super(Transformer, self).__init__()

        self.Encoder_Embedding = EmbeddingBlock(VOCAB_SIZE, D_MODEL)
        self.Encoder_POS_ENCODING = PositionalEncoding(MAX_LEN, D_MODEL)

        self.encoders = nn.ModuleList([EncoderBlock(D_MODEL, H) for i in range(N_Enc)])

        self.Decoder_Embedding = EmbeddingBlock(VOCAB_SIZE, D_MODEL)
        self.Decoder_POS_ENCODING = PositionalEncoding(MAX_LEN, D_MODEL)

        self.decoders = nn.ModuleList([DecoderBlock(D_MODEL, H) for i in range(N_Dec)])

        self.projection_layer = Projection(D_MODEL, VOCAB_SIZE)

    def forward(self, src_tokens, tgt_tokens, src_mask, tgt_mask):
        enc_embeddings = self.Encoder_Embedding(src_tokens)
        enc_pos = self.Encoder_POS_ENCODING(enc_embeddings)

        dec_embeddings = self.Decoder_Embedding(tgt_tokens)
        dec_pos = self.Decoder_POS_ENCODING(dec_embeddings)

        for encoder in self.encoders:
            enc_pos = encoder(enc_pos, src_mask)

        for decoder in self.decoders:
            dec_pos = decoder(dec_pos, enc_pos, src_mask, tgt_mask)

        return self.projection_layer(dec_pos)