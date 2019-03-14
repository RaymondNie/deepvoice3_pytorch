# coding: utf-8
"""
Synthesis waveform from trained model.

usage: synthesis.py [options] <checkpoint> <text_list_file> <dst_dir>

options:
    --hparams=<parmas>                Hyper parameters [default: ].
    --preset=<json>                   Path of preset parameters (json).
    --checkpoint-seq2seq=<path>       Load seq2seq model from checkpoint path.
    --checkpoint-postnet=<path>       Load postnet model from checkpoint path.
    --file-name-suffix=<s>            File name suffix [default: ].
    --max-decoder-steps=<N>           Max decoder steps [default: 500].
    --replace_pronunciation_prob=<N>  Prob [default: 0.0].
    --speaker_id=<id>                 Speaker ID (for multi-speaker model).
    --output-html                     Output html for blog post.
    --epochs=<N>                      How many times to iterate through sentences
    -h, --help               Show help message.
"""
from docopt import docopt
from torch import nn

import sys
import os
from os.path import dirname, join, basename, splitext
import audio
import time

import torch
import numpy as np
import pandas as pd
import nltk
import random

# The deepvoice3 model
from deepvoice3_pytorch import frontend
from hparams import hparams, hparams_debug_string
from train import guided_attention
from tqdm import tqdm
from torch import nn
from torch.utils import data as data_utils

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
_frontend = None  # to be set later

class InferDataset(object):
    def __init__(self, file_path):
        file = open(file_path, 'r')
        self.dataset = file.readlines()
        file.close()
    
    def __getitem__(self, idx):
        sequence = _frontend.text_to_sequence(self.dataset[idx][:-1], p=0), self.dataset[idx]
        while len(sequence) >= hparams.max_positions:
            idx = random.randint(0, len(self.dataset)-1)
            sequence = _frontend.text_to_sequence(self.dataset[idx][:-1], p=0), self.dataset[idx]
        return sequence

    def __len__(self):
        return len(self.dataset)

def _pad(seq, max_len, constant_values=0):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=constant_values)

def collate_fn(batch):
    # Lengths
    input_lengths = [len(x[0]) for x in batch]
    max_input_len = max(input_lengths)
    a = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.int)
    x_batch = torch.LongTensor(a)
    b = [x[1] for x in batch]
    return x_batch, b

def tts(model, text, p=0, speaker_id=None, fast=False, batch_synthesis=False):
    """Convert text to speech waveform given a deepvoice3 model.

    Args:
        text (str) : Input text to be synthesized
        p (float) : Replace word to pronounciation if p > 0. Default is 0.
    """
    model = model.to(device)
    model.eval()
    if fast:
        model.module.make_generation_fast_()

    sequence = np.array(_frontend.text_to_sequence(text, p=p))
    sequence = torch.from_numpy(sequence).unsqueeze(0).long().to(device)
    text_positions = torch.arange(1, sequence.size(-1) + 1).unsqueeze(0).long().to(device)
    speaker_ids = None if speaker_id is None else torch.LongTensor([speaker_id]).to(device)

    # Greedy decoding
    with torch.no_grad():
        mel_outputs, linear_outputs, alignments, done = model(
            sequence, text_positions=text_positions, speaker_ids=speaker_ids)

    linear_output = linear_outputs[0].cpu().data.numpy()
    spectrogram = audio._denormalize(linear_output)
    alignment = alignments[0].cpu().data.numpy()
    mel = mel_outputs[0].cpu().data.numpy()
    mel = audio._denormalize(mel)
    
    # # Predicted audio signal
    waveform = audio.inv_spectrogram(linear_output.T)
    
    return waveform, alignment, spectrogram, mel

def batch_tts(model, text, p=0, speaker_id=None, fast=False, batch_synthesis=False):
    """Convert text to speech waveform given a deepvoice3 model.

    Args:
        text (str) : Input text to be synthesized
        p (float) : Replace word to pronounciation if p > 0. Default is 0.
    """
    model = model.to(device)
    model.eval()

    if fast:
        model.module.make_generation_fast_()

    text_positions = torch.arange(1, text.size(-1) + 1).unsqueeze(0).repeat(hparams.batch_size, 1).long().to(device)
    speaker_ids = torch.LongTensor([speaker_id]).repeat(hparams.batch_size).to(device)

    # Greedy decoding
    with torch.no_grad():
        mel_outputs, linear_outputs, alignments, done = model(
            text, text_positions=text_positions, speaker_ids=speaker_ids)
    return linear_outputs, alignments, done

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_path = args["<checkpoint>"]
    text_list_file_path = args["<text_list_file>"]
    dst_dir = args["<dst_dir>"]
    checkpoint_seq2seq_path = args["--checkpoint-seq2seq"]
    checkpoint_postnet_path = args["--checkpoint-postnet"]
    max_decoder_steps = int(args["--max-decoder-steps"])
    file_name_suffix = args["--file-name-suffix"]
    replace_pronunciation_prob = float(args["--replace_pronunciation_prob"])
    output_html = args["--output-html"]
    speaker_id = args["--speaker_id"]
    n_epochs = int(args["--epochs"])

    # list of bad speakers
    # bad_speakers = [0,1,8,9,17,21,23,24,25,34,39,41,43,46,59,84,85,93,139,147,149,193,240,245,228]

    if speaker_id is not None:
        speaker_id = int(speaker_id)
    preset = args["--preset"]

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "deepvoice3"

    _frontend = getattr(frontend, hparams.frontend)
    import train
    train._frontend = _frontend
    from train import plot_alignment, build_model
    # Model
    model = build_model()
    model = nn.DataParallel(model)

    # Load checkpoints separately
    if checkpoint_postnet_path is not None and checkpoint_seq2seq_path is not None:
        checkpoint = _load(checkpoint_seq2seq_path)
        model.seq2seq.load_state_dict(checkpoint["state_dict"])
        checkpoint = _load(checkpoint_postnet_path)
        model.postnet.load_state_dict(checkpoint["state_dict"])
        checkpoint_name = splitext(basename(checkpoint_seq2seq_path))[0]
    else:
        checkpoint = _load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        checkpoint_name = splitext(basename(checkpoint_path))[0]

    model.module.seq2seq.decoder.max_decoder_steps = max_decoder_steps

    os.makedirs(dst_dir, exist_ok=True)
    metadata = []
    batch = 0
    epoch = 0
    dataset = InferDataset(text_list_file_path)
    data_loader = data_utils.DataLoader(dataset, batch_size=hparams.batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)
    start = time.time()

    # for speaker_id in range(hparams.n_speakers):
    #     for text_ids, text in data_loader:
    #         linear_outputs, alignments, dones = batch_tts(model, text_ids, 0, speaker_id)
    #         for idx, linear_output in enumerate(linear_outputs):
    #             stop_frame = 0
    #             for done in dones:
    #                 if done[idx] > 0.9:
    #                     break
    #                 stop_frame += 1
    #             linear_output = linear_output.cpu().data.numpy()
    #             linear_output = linear_output[:stop_frame * hparams.downsample_step * hparams.outputs_per_step,:]
    #             alignment = alignments[idx].cpu().data.numpy()
    #             alignment = alignment[:stop_frame,:len(text[idx])]

    #             waveform = audio.inv_spectrogram(linear_output.T)
    #             dst_wav_path = join(dst_dir, "speaker_id_{}_index_{}.wav".format(
    #                 speaker_id, idx))
    #             dst_alignment_path = join(
    #                 dst_dir, "speaker_id_{}_index_{}_alignment.png".format(
    #                 speaker_id, idx))
    #             audio.save_wav(waveform, dst_wav_path)
    #             plot_alignment(alignment.T, dst_alignment_path,
    #                            info="{}, {}".format(hparams.builder, basename(checkpoint_path)))

    while epoch < n_epochs:
        for text_ids, text in data_loader:
            # Generate data for a random speaker
            speaker_id = random.randint(0, hparams.n_speakers-1)
            # while speaker_id in bad_speakers:
                # speaker_id = random.randint(0, hparams.n_speakers)

            linear_outputs, alignments, dones = batch_tts(model, text_ids, 0, speaker_id)
            for idx, linear_output in enumerate(linear_outputs):
                stop_frame = 0
                # Find when to cut off the audio based on stop token prediction
                for done in dones:
                    if done[idx] > 0.9:
                        break
                    stop_frame += 1

                linear_output = linear_output.cpu().data.numpy()
                linear_output = linear_output[:stop_frame * hparams.downsample_step * hparams.outputs_per_step,:]
                alignment = alignments[idx].cpu().data.numpy()
                alignment = alignment[:stop_frame,:len(text[idx])]

                # Filter out bad audio with alignment loss 
                soft_max = guided_attention(
                    alignment.shape[0], 
                    alignment.shape[0],
                    alignment.shape[1],
                    alignment.shape[1],
                    0.2
                )
                attn_loss = (alignment * soft_max).mean()

                if attn_loss > 0.00025:
                    continue

                waveform = audio.inv_spectrogram(linear_output.T)
                dst_wav_path = join(dst_dir, "batch{}_index{}_speaker_id_{}_checkpoint_{}.wav".format(
                    batch, idx, speaker_id, checkpoint_name))
                dst_alignment_path = join(
                    dst_dir, "batch{}_index{}_speaker_id_{}_checkpoint_{}_alignment.png".format(
                    batch, idx, speaker_id, checkpoint_name))
                audio.save_wav(waveform, dst_wav_path)
                plot_alignment(alignment.T, dst_alignment_path,
                               info="{}, {}".format(hparams.builder, basename(checkpoint_path)))

                # Save metadata
                metadata.append([dst_wav_path, text[idx]])
            metadata_df = pd.DataFrame(metadata)
            metadata_df.to_csv("{}/{}_metadata.csv".format(dst_dir, batch), encoding='utf-8', index=False, header=["wav_filename", "transcript"])
            batch += 1
        epoch += 1

    end = time.time()
    print("Total duration: {}".format(end-start))
        
    sys.exit(0)
