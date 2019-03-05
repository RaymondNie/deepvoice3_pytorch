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
    --max_speaker_id=<max_id>         Will create audio samples for speaker ids from 0 to max_id
    --speakers_per_utterance=<int>    How many speakers per utterance
    --output-html                     Output html for blog post.
    -h, --help               Show help message.
"""
from docopt import docopt
from torch import nn

import sys
import os
from os.path import dirname, join, basename, splitext
import audio

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

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
_frontend = None  # to be set later

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
    
    # linear_output = linear_outputs[0].cpu().data.numpy()
    # spectrogram = audio._denormalize(linear_output)
    # alignment = alignments[0].cpu().data.numpy()
    # mel = mel_outputs[0].cpu().data.numpy()
    # mel = audio._denormalize(mel)
    
    # # Predicted audio signal
    # waveform = audio.inv_spectrogram(linear_output.T)
    
    # Jasper conversions
    linear_outputs = linear_outputs.data.cpu().numpy()
    mel_outputs = mel_outputs.data.cpu().numpy()

    linear_outputs = audio._denormalize(linear_outputs) + 20
    mel_outputs = audio._denormalize(mel_outputs) + 20

    mel_to_mag = audio.jasper_inverse_mel(mel_outputs, 16000, 512, 64)
    mag_to_mag = audio.jasper_get_mag_spec(linear_outputs)

    mel_signal = audio.jasper_griffin_lim(mel_to_mag.T)
    mag_signal = audio.jasper_griffin_lim(mag_to_mag.T)

    return mel_signal, mag_signal, alignment, spectrogram, mel


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
    if speaker_id is not None:
        speaker_id = int(speaker_id)
    max_speaker_id = args["--max_speaker_id"]
    if max_speaker_id is not None:
        max_speaker_id = int(max_speaker_id)
    speakers_per_utterance = int(args["--speakers_per_utterance"])
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

    with open(text_list_file_path, "rb") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            generated_speech_count = 0
            fail_count = 0
            sample_audio = [] # Get 10 audio samples 
            while generated_speech_count < speakers_per_utterance and fail_count < 10:
                # Generate a random speaker_id and try to generate audio
                rand_speaker_id = random.randint(0,max_speaker_id)
                text = line.decode("utf-8")[:-1]
                words = nltk.word_tokenize(text)
                waveform, alignment, _, _ = tts(
                    model, text, p=replace_pronunciation_prob, speaker_id=rand_speaker_id, fast=True)
                soft_mask = guided_attention(alignment.shape[0], alignment.shape[0], alignment.shape[1], alignment.shape[1], 0.2)
                attn_loss = (alignment * soft_mask).mean()
                # Filter out poorly generated audio
                duration = waveform.shape[0] / 22050 # divide by sample rate
                if attn_loss < 0.00025 or duration < 16.00:
                    print("{}_{}{}_speaker_id_{}.wav, {}".format(
                        idx, checkpoint_name, file_name_suffix, rand_speaker_id, attn_loss))
                    generated_speech_count += 1

                    dst_wav_path = join(dst_dir, "{}_{}{}_speaker_id_{}.wav".format(
                        idx, checkpoint_name, file_name_suffix, rand_speaker_id))
                    dst_alignment_path = join(
                        dst_dir, "{}_{}{}_speaker_id_{}_alignment.png".format(idx, checkpoint_name,
                                                                file_name_suffix, rand_speaker_id))
                    metadata.append([dst_wav_path, text])
                    plot_alignment(alignment.T, dst_alignment_path,
                                   info="{}, {}".format(hparams.builder, basename(checkpoint_path)))
                    audio.save_wav(waveform, dst_wav_path)
                    name = splitext(basename(text_list_file_path))[0]
                else:
                    fail_count += 1

        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv("{}/metadata.csv".format(dst_dir), encoding='utf-8', index=False, header=None)
        # print("Finished! Check out {} for generated audio samples.".format(dst_dir))
        sys.exit(0)
