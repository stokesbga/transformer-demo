import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BillingualDataset, causal_mask
from model import build_transformer
from config import get_config, get_weights_pathname

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import warnings
from pathlib import Path
import os


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = os.path.realpath(config['tokenizer_file'].format(lang))
    print('TOKENIZER PATH: %s', tokenizer_path)
    if os.path.exists(tokenizer_path):
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
            min_frequency=2
        )
        tokenizer.train_from_iterator(
            get_all_sentences(ds, lang), trainer=trainer)
        print('Saving new tokenizer file as: {}', tokenizer_path)
        tokenizer.save(tokenizer_path)
    return tokenizer


def get_ds(config):
    ds = load_dataset(
        'opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    tokenizer_src = get_or_build_tokenizer(config, ds, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds, config['lang_tgt'])

    print('CREATED TOKENIZER: ', tokenizer_src)

    train_ds_size = int(0.9*len(ds))
    eval_ds_size = len(ds) - train_ds_size
    train_ds_raw, eval_ds_raw = torch.utils.data.random_split(
        ds, [train_ds_size, eval_ds_size])

    train_ds = BillingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt,
                                 config['lang_src'], config['lang_tgt'], config['seq_len'])
    eval_ds = BillingualDataset(eval_ds_raw, tokenizer_src, tokenizer_tgt,
                                config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Dynamically set max seq_len for longest sentence in dataset
    max_len_src = 0
    max_len_tgt = 0

    for item in train_ds_raw:
        max_len_src = max(max_len_src, len(tokenizer_src.encode(
            item['translation'][config['lang_src']]).ids))
        max_len_tgt = max(max_len_tgt, len(tokenizer_src.encode(
            item['translation'][config['lang_tgt']]).ids))

    train_dataloader = DataLoader(
        train_ds, batch_size=config['batch_size'], shuffle=True)
    eval_dataloader = DataLoader(eval_ds, batch_size=1, shuffle=True)

    return train_dataloader, eval_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, max_len_src, max_len_tgt):
    model = build_transformer(
        src_vocab_size=max_len_src,
        tgt_vocab_size=max_len_tgt,
        src_seq_len=config['seq_len'],
        tgt_seq_len=config['seq_len'],
        d_model=config['d_model'],
    )
    return model


def train(config):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f'Using device: {device}')
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, eval_dataloader, tokenizer_src, tokenizer_tgt = get_ds(
        config)
    model = get_model(config, tokenizer_src.get_vocab_size(),
                      tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filenmae = get_weights_pathname
        state = torch.load(model_filenmae)
        initial_epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id(
        "[PAD]"), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask)
            y_pred = model.project(decoder_output)

            # Calculate loss
            # Get comparable label
            label = batch['label'].to(device)
            loss = loss_fn(
                y_pred.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix(f'loss: {loss.item():6.3f}')

            writer.add_scalar('train loss: ', loss.item(),
                              global_step=global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # Save model
        model_filename = get_weights_pathname(config, f'{epoch:02d}')
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train(config)
