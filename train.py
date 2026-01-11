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

def get_all_sentences(ds, field):
    sentences = []
    for example in ds:
        sentences.append(example[field])
    return sentences

def get_or_build_tokenizer(config, dataset, field):
    tokenizer_file = Path(config['tokenizer_file'].format(field))
    tokenizer_file.parent.mkdir(parents=True, exist_ok=True)
    if tokenizer_file.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_file))
    else:
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            vocab_size=config['vocab_size'],
            special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[SOS]", "[POS]", "[EOS]"],
            min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(dataset, field), trainer=trainer)
        tokenizer.save(str(tokenizer_file))

    return tokenizer

def save_dataset_to_json(dataset, filename='meetingbank_dataset.json'):
    import json
    with open(filename, 'w') as f:
        for i, item in enumerate(dataset):
            if i >= 10:
                break
            sample = {
                'id': item['id'],
                'uid': item['uid'],
                'summary': item['summary'][:200] + '...' if len(item['summary']) > 200 else item['summary'],
                'transcript': item['transcript'][:500] + '...' if len(item['transcript']) > 500 else item['transcript']
            }
            f.write(json.dumps(sample) + '\n')
    print(f"Dataset samples saved to {filename}")

def get_dataset(config):
    print(f"Loading dataset: {config['dataset_name']}")
    meetingbank = load_dataset(config['dataset_name'])

    train_raw = meetingbank['train']
    val_raw = meetingbank['validation']
    test_raw = meetingbank['test']

    print(f"Train size: {len(train_raw)}, Val size: {len(val_raw)}, Test size: {len(test_raw)}")

    save_dataset_to_json(train_raw)
    src_tokenizer = get_or_build_tokenizer(config, train_raw, 'transcript')
    tgt_tokenizer = get_or_build_tokenizer(config, train_raw, 'summary')

    print(f"Source vocab size: {src_tokenizer.get_vocab_size()}")
    print(f"Target vocab size: {tgt_tokenizer.get_vocab_size()}")

    from dataset import MeetingSummarizationDataset

    train_dataset = MeetingSummarizationDataset(
        train_raw,
        src_tokenizer,
        tgt_tokenizer,
        config['max_src_len'],
        config['max_tgt_len']
    )
    val_dataset = MeetingSummarizationDataset(
        val_raw,
        src_tokenizer,
        tgt_tokenizer,
        config['max_src_len'],
        config['max_tgt_len']
    )

    max_len_src = 0
    max_len_tgt = 0

    print("Calculating max sequence lengths...")
    for i, item in enumerate(train_raw):
        if i >= 1000:
            break
        src_ids = src_tokenizer.encode(item['transcript']).ids
        tgt_ids = tgt_tokenizer.encode(item['summary']).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of transcript (sampled): {max_len_src}")
    print(f"Max length of summary (sampled): {max_len_tgt}")

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer

def get_model(config, src_vocab_size, tgt_vocab_size):
    model = build_transformer(
        src_vocab_size=src_vocab_size,
        target_vocab_size=tgt_vocab_size,
        src_seq_len=config['max_src_len'],
        target_seq_len=config['max_tgt_len'],
        d_model_size=config.get('d_model', 512),
        d_ff=config.get('d_ff', 2048),
        h=config.get('h', 8),
        dropout=config.get('dropout', 0.1),
        n=config.get('n_layers', 6),
        use_hyper_connection=config.get('use_hyper_connection', False),
        hyper_n=config.get('hyper_n', 4)
    )

    return model

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer = get_dataset(config)
    model = get_model(config, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size()).to(device)

    writer = SummaryWriter(log_dir=f"runs/{config['experiment_name']}")

    muon_params = [p for p in model.parameters() if p.dim() == 2]
    other_params = [p for p in model.parameters() if p.dim() != 2]

    muon_optimizer = torch.optim.Muon(muon_params, lr=config.get('lr', 1e-4), weight_decay=config.get('weight_decay', 0.01))
    other_optimizer = torch.optim.AdamW(other_params, lr=config.get('lr', 1e-4), weight_decay=config.get('weight_decay', 0.01))

    initial_epoch = 0
    global_step = 0
    if config['preload'] is not None:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Loading model from {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        muon_optimizer.load_state_dict(state['muon_optimizer'])
        other_optimizer.load_state_dict(state['other_optimizer'])
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

            muon_optimizer.zero_grad()
            other_optimizer.zero_grad()

            encoder_output = model.encode(src, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, tgt, decoder_mask)
            logits = model.project(decoder_output)

            loss = loss_fn(logits.view(-1, tgt_tokenizer.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix(loss=f"{loss.item():6.3f}")

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping -> to avoid exploding gradients

            # if global_step % 100 == 0:
            #     total_grad = 0.0
            #     hc_grad = 0.0
            #     for name, p in model.named_parameters():
            #         if p.grad is not None:
            #             grad_norm = p.grad.data.norm(2).item()
            #             total_grad += grad_norm ** 2

            #             # updated: check for new HC param names
            #             if 'hc_' in name or any(x in name for x in ['static_alpha', 'static_beta', 'dynamic_alpha', 'dynamic_beta']):
            #                 hc_grad += grad_norm ** 2

                # print(f"Step {global_step}: Total Grad Norm: {total_grad**0.5:.4f}, HyperConnection Grad Norm: {hc_grad**0.5:.4f}")

                # for name, module in model.name_modules():
                #     if isinstance(module, HyperConnection):
                #         print(f"Step {global_step}: HyperConnection Layer {name} params:")
                #         print(f"  W_beta norm: {module.W_beta.data.norm(2).item():.4f}")
                #         print(f"  B norm: {module.B.data.norm(2).item():.4f}")

            muon_optimizer.step()
            other_optimizer.step()

            muon_optimizer.zero_grad()
            other_optimizer.zero_grad()

            global_step += 1

        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        Path(model_filename).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'muon_optimizer': muon_optimizer.state_dict(),
            'other_optimizer': other_optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)