import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from dataset import BilingualDataset, causal_mask

from transformer_model import build_transformer

from config import get_config, get_weights_file_path

from tqdm import tqdm 

import warnings

def get_all_sentences(ds, lang):
    sentences = []
    for example in ds:
        if lang == 'en':
            sentences.append(example['src'])
        else:
            sentences.append(example['tgt'])
    return sentences

def get_or_build_tokenizer(config, dataset, lang):
    tokenizer_file = Path(config['tokenizer_file'].format(lang))
    tokenizer_file.parent.mkdir(parents=True, exist_ok=True)
    if tokenizer_file.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_file))
    else:
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(vocab_size=config['vocab_size'], special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[SOS]", "[POS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_file))

    return tokenizer

def save_dataset_to_json(dataset):
    with open('dataset.json', 'w') as f:
        for item in dataset:
            f.write(str(item) + '\n')

def get_dataset(config):
    raw_dataset = load_dataset('ai4bharat/samanantar', f'{config["lang_tgt"]}', split='train')

    save_dataset_to_json(raw_dataset)

    src_tokenizer = get_or_build_tokenizer(config, raw_dataset, config['lang_src'])
    tgt_tokenizer = get_or_build_tokenizer(config, raw_dataset, config['lang_tgt'])

    # print(src_tokenizer)
    # print(tgt_tokenizer)

    train_dataset_size = int(0.9 * len(raw_dataset))
    val_dset_size = len(raw_dataset) - train_dataset_size
    train_dataset, val_dataset = random_split(raw_dataset, [train_dataset_size, val_dset_size]) 
     
    train_dataset = BilingualDataset(train_dataset, src_tokenizer, tgt_tokenizer, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_dataset = BilingualDataset(val_dataset, src_tokenizer, tgt_tokenizer, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in raw_dataset:
        src_ids = src_tokenizer.encode(item['src']).ids
        tgt_ids = tgt_tokenizer.encode(item['tgt']).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence is: {max_len_src}, and target sentences is: {max_len_tgt}")

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer

def get_model(config, src_vocab_size, tgt_vocab_size):
    model = build_transformer(
        src_vocab_size=src_vocab_size,
        target_vocab_size=tgt_vocab_size,
        src_seq_len=config['seq_len'],
        target_seq_len=config['seq_len']
    )

    return model

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer = get_dataset(config)
    model = get_model(config, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size()).to(device)

    writer = SummaryWriter(log_dir=f"runs/{config['experiment_name']}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)


    initial_epoch = 0
    global_step = 0
    if config['preload'] is not None:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Loading model from {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}", unit="batch")
        for batch in batch_iterator:
            src = batch['encoder_input'].to(device)
            tgt = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            # src_mask = causal_mask(src.shape[1], device=device).to(device)
            # tgt_mask = causal_mask(tgt.shape[1], device=device).to(device)

            optimizer.zero_grad()

            encoder_output = model.encode(src, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, tgt, decoder_mask)
            logits = model.project(decoder_output)

            loss = loss_fn(logits.view(-1, tgt_tokenizer.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix(loss=f"{loss.item():6.3f}")

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)