import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

class BilingualDataset(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_tokens = torch.tensor([tokenizer_src.token_to_id(['[SOS]'])], dtype=torch.int64)
        self.eos_tokens = torch.tensor([tokenizer_src.token_to_id(['[EOS]'])], dtype=torch.int64)
        self.pad_tokens = torch.tensor([tokenizer_src.token_to_id(['[PAD]'])], dtype=torch.int64)

        self.seq_len = seq_len

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        src_target_pair = self.dataset[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_paddings = self.seq_len - len(enc_input_tokens) - 2
        dec_num_paddings = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_paddings < 0 or dec_num_paddings < 0:
            raise ValueError(f"Input sequence is too long. Max length is {self.seq_len}.")
        
        encoder_input = torch.cat(
            [
                self.sos_tokens,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_tokens,
                torch.tensor([self.pad_tokens] * enc_num_paddings, dtype=torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_tokens,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_tokens] * dec_num_paddings, dtype=torch.int64)
            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_tokens,
                torch.tensor([self.pad_tokens] * dec_num_paddings, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_attention_mask': (encoder_input != self.pad_tokens).unsqueeze(0).unsqueeze(0).int(),
            'decoder_attention_mask': (decoder_input != self.pad_tokens).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            'label': label,
            'src_text': src_text,
            'tgt_text': tgt_text
        }

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0



         