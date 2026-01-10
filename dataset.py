import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

class MeetingSummarizationDataset(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, seq_len_src, seq_len_tgt):
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src  # For transcripts
        self.tokenizer_tgt = tokenizer_tgt  # For summaries
        self.seq_len_src = seq_len_src
        self.seq_len_tgt = seq_len_tgt

        self.src_sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.src_eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.src_pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

        self.tgt_sos_token = torch.tensor([tokenizer_tgt.token_to_id('[SOS]')], dtype=torch.int64)
        self.tgt_eos_token = torch.tensor([tokenizer_tgt.token_to_id('[EOS]')], dtype=torch.int64)
        self.tgt_pad_token = torch.tensor([tokenizer_tgt.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        src_text = item['transcript']
        tgt_text = item['summary']

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Truncate if too long
        if len(enc_input_tokens) > self.seq_len_src - 2:
            enc_input_tokens = enc_input_tokens[:self.seq_len_src - 2]

        if len(dec_input_tokens) > self.seq_len_tgt - 2:
            dec_input_tokens = dec_input_tokens[:self.seq_len_tgt - 2]

        enc_num_paddings = self.seq_len_src - len(enc_input_tokens) - 2
        dec_num_paddings = self.seq_len_tgt - len(dec_input_tokens) - 1

        if enc_num_paddings < 0 or dec_num_paddings < 0:
            raise ValueError(f"Input sequence is too long after truncation. Max src: {self.seq_len_src}, Max tgt: {self.seq_len_tgt}.")

        encoder_input = torch.cat(
            [
                self.src_sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.src_eos_token,
                torch.tensor([self.src_pad_token] * enc_num_paddings, dtype=torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                self.tgt_sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.tgt_pad_token] * dec_num_paddings, dtype=torch.int64)
            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.tgt_eos_token,
                torch.tensor([self.tgt_pad_token] * dec_num_paddings, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len_src
        assert decoder_input.size(0) == self.seq_len_tgt
        assert label.size(0) == self.seq_len_tgt

        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask': (encoder_input != self.src_pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, src_seq_len) -> broadcastable to (batch, h, tgt_seq_len, src_seq_len)
            'decoder_mask': (decoder_input != self.tgt_pad_token).unsqueeze(0).int().unsqueeze(0) & causal_mask(decoder_input.size(0)),  # (1, tgt_seq_len, tgt_seq_len) -> broadcastable to (batch, h, tgt_seq_len, tgt_seq_len)
            'label': label,
            'src_text': src_text,
            'tgt_text': tgt_text,
            'id': item['id'],
            'uid': item['uid']
        }

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0


class BilingualDataset(MeetingSummarizationDataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__(dataset, tokenizer_src, tokenizer_tgt, seq_len, seq_len)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

    def __getitem__(self, index):
        item = self.dataset[index]

        if 'src' in item and 'tgt' in item:
            src_text = item['src']
            tgt_text = item['tgt']
        elif 'transcript' in item and 'summary' in item:
            src_text = item['transcript']
            tgt_text = item['summary']
        else:
            raise ValueError("Unknown dataset format")

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        if len(enc_input_tokens) > self.seq_len - 2:
            enc_input_tokens = enc_input_tokens[:self.seq_len - 2]

        if len(dec_input_tokens) > self.seq_len - 2:
            dec_input_tokens = dec_input_tokens[:self.seq_len - 2]

        enc_num_paddings = self.seq_len - len(enc_input_tokens) - 2
        dec_num_paddings = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_paddings < 0 or dec_num_paddings < 0:
            raise ValueError(f"Input sequence is too long. Max length is {self.seq_len}.")

        encoder_input = torch.cat(
            [
                self.src_sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.src_eos_token,
                torch.tensor([self.src_pad_token] * enc_num_paddings, dtype=torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                self.tgt_sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.tgt_pad_token] * dec_num_paddings, dtype=torch.int64)
            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.tgt_eos_token,
                torch.tensor([self.tgt_pad_token] * dec_num_paddings, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        result = {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask': (encoder_input != self.src_pad_token).unsqueeze(0).unsqueeze(0).int(),
            'decoder_mask': (decoder_input != self.tgt_pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            'label': label,
            'src_text': src_text,
            'tgt_text': tgt_text
        }

        if 'id' in item:
            result['id'] = item['id']
        if 'uid' in item:
            result['uid'] = item['uid']

        return result



