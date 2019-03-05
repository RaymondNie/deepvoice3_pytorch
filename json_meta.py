'''
Started in 1945h, Mar 10, 2018
First done in 2103h, Mar 11, 2018
Test done in 2324h, Mar 11, 2018
Modified for HTK labeling in 1426h, Apr 21, 2018
by engiecat(github)

This makes r9y9/deepvoice3_pytorch compatible with json format of carpedm20/multi-speaker-tacotron-tensorflow and keithito/tacotron.
The json file is given per speaker, generated in the format of 
	(if completely aligned)
		(path-to-the-audio):aligned text

	(if partially aligned)
		(path-to-the-audio):[candidate sentence - not aligned,recognized words]

	(if non-aligned)
		(path-to-the-audio):[recognized words]
is given per speaker.

(e.g. python preprocess.py json_meta "./datasets/LJSpeech_1_0/alignment.json,./datasets/GoTBookRev/alignment.json" "./datasets/LJ+GoTBookRev" --preset=./presets/deepvoice3_vctk.json )

usage: 
    python preprocess.py [option] <json_paths> <output_data_path>


options:
    --preset     Path of preset parameters (json).
    -h --help    show this help message and exit


'''

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import audio
from nnmnkwii.io import hts
from hparams import hparams
from os.path import exists
import librosa
import json

import scipy.io.wavfile as wave
import python_speech_features as psf

def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    
    json_paths = in_dir.split(',')
    json_paths = [json_path.replace("'", "").replace('"',"") for json_path in json_paths]
    num_speakers = len(json_paths)
    is_aligned = {}
    
    speaker_id=0
    for json_path in json_paths:
        # Loads json metadata info
        if json_path.endswith("json"):
            with open(json_path, encoding='utf8') as f:
                content = f.read()
            info = json.loads(content)
        elif json_path.endswith("csv"):
            with open(json_path) as f:
                info = {}
                for line in f:
                    path, text = line.strip().split('|')
                    info[path] = text
        else:
            raise Exception(" [!] Unknown metadata format: {}".format(json_path))

        print(" [*] Loaded - {}".format(json_path))
        # check audio file existence
        base_dir = os.path.dirname(json_path)
        new_info = {}
        for path in info.keys():
            if not os.path.exists(path):
                new_path = os.path.join(base_dir, path)
                if not os.path.exists(new_path):
                    print(" [!] Audio not found: {}".format([path, new_path]))
                    continue
            else:
                new_path = path
            
            new_info[new_path] = info[path]
        
        info = new_info
        
        # ignore_recognition_level check
        for path in info.keys():
            is_aligned[path] = True
            if isinstance(info[path], list):
                if hparams.ignore_recognition_level == 1 and len(info[path]) == 1 or \
                        hparams.ignore_recognition_level == 2:
                    # flag the path to be 'non-aligned' text
                    is_aligned[path] = False
                info[path] = info[path][0]
        
        # Reserve for future processing
        queue_count = 0
        for audio_path, text in info.items():
            if isinstance(text, list):
                if hparams.ignore_recognition_level == 0:
                    text = text[-1]
                else:
                    text = text[0]
            if hparams.ignore_recognition_level > 0 and not is_aligned[audio_path]:
                continue
            if hparams.min_text > len(text):
                continue
            if num_speakers == 1:
                # Single-speaker
                futures.append(executor.submit(
                    partial(_process_utterance_single, out_dir, text, audio_path)))
            else:
                # Multi-speaker
                futures.append(executor.submit(
                    partial(_process_utterance_jasper, out_dir, text, audio_path, speaker_id)))
            queue_count += 1
        print(" [*] Appended {} entries in the queue".format(queue_count))
        
        # increase speaker_id
        speaker_id += 1
    
    # Show ignore_recognition_level description
    ignore_description = {
        0: "use all",
        1: "ignore only unmatched_alignment",
        2: "fully ignore recognition",
    }
    print(" [!] Skip recognition level: {} ({})". \
            format(hparams.ignore_recognition_level,
                   ignore_description[hparams.ignore_recognition_level]))
    
    if num_speakers == 1:
        print(" [!] Single-speaker mode activated!")
    else:
        print(" [!] Multi-speaker({}) mode activated!".format(num_speakers))
    
    # Now, Do the job!
    results = [future.result() for future in tqdm(futures)]
    # Remove entries with None (That has been filtered due to bad htk alginment (if process_only_htk_aligned is enabled in hparams)
    results = [result for result in results if result != None]
    return results
    

def start_at(labels):
    has_silence = labels[0][-1] == "pau"
    if not has_silence:
        return labels[0][0]
    for i in range(1, len(labels)):
        if labels[i][-1] != "pau":
            return labels[i][0]
    assert False


def end_at(labels):
    has_silence = labels[-1][-1] == "pau"
    if not has_silence:
        return labels[-1][1]
    for i in range(len(labels) - 2, 0, -1):
        if labels[i][-1] != "pau":
            return labels[i][1]
    assert False


def _process_utterance(out_dir, text, wav_path, speaker_id=None):

    # check whether singlespeaker_mode
    if speaker_id is None:
        return _process_utterance_single(out_dir,text,wav_path)
    # modified version of VCTK _process_utterance
    sr = hparams.sample_rate

    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)
    
    lab_path = wav_path.replace("wav48/", "lab/").replace(".wav", ".lab")
    if not exists(lab_path):
        lab_path = os.path.splitext(wav_path)[0]+'.lab'

    # Trim silence from hts labels if available
    if exists(lab_path):
        labels = hts.load(lab_path)
        b = int(start_at(labels) * 1e-7 * sr)
        e = int(end_at(labels) * 1e-7 * sr)
        wav = wav[b:e]
        wav, _ = librosa.effects.trim(wav, top_db=25)
    else:
        if hparams.process_only_htk_aligned:
            return None
        wav, _ = librosa.effects.trim(wav, top_db=15)

    if hparams.rescaling:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    n_frames = spectrogram.shape[1]
    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
    # Write the spectrograms to disk: 
    # Get filename from wav_path
    wav_name = os.path.basename(wav_path)
    wav_name = os.path.splitext(wav_name)[0]
    
    # case if wave files across different speakers have the same naming format.
    # e.g. Recording0.wav
    spectrogram_filename = 'spec-{}-{}.npy'.format(speaker_id, wav_name)
    mel_filename = 'mel-{}-{}.npy'.format(speaker_id, wav_name)
    np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
    # Return a tuple describing this training example:
    return (spectrogram_filename, mel_filename, n_frames, text, speaker_id)
    
def _process_utterance_single(out_dir, text, wav_path):
    # modified version of LJSpeech _process_utterance

    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)
    sr = hparams.sample_rate
    # Added from the multispeaker version
    lab_path = wav_path.replace("wav48/", "lab/").replace(".wav", ".lab")
    if not exists(lab_path):
        lab_path = os.path.splitext(wav_path)[0]+'.lab'

    # Trim silence from hts labels if available
    if exists(lab_path):
        labels = hts.load(lab_path)
        b = int(start_at(labels) * 1e-7 * sr)
        e = int(end_at(labels) * 1e-7 * sr)
        wav = wav[b:e]
        wav, _ = librosa.effects.trim(wav, top_db=25)
    else:
        if hparams.process_only_htk_aligned:
            return None
        wav, _ = librosa.effects.trim(wav, top_db=15)
    # End added from the multispeaker version
    
    if hparams.rescaling:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    n_frames = spectrogram.shape[1]

    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
    # Write the spectrograms to disk: 
    # Get filename from wav_path
    wav_name = os.path.basename(wav_path)
    wav_name = os.path.splitext(wav_name)[0]
    spectrogram_filename = 'spec-{}.npy'.format(wav_name)
    mel_filename = 'mel-{}.npy'.format(wav_name)
    np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example:
    return (spectrogram_filename, mel_filename, n_frames, text)

def _process_utterance_jasper(out_dir, text, wav_path, speaker_id):
    # Jasper spec params
    window_size_ms = 20e-3
    window_stride_ms = 10e-3
    sample_freq, signal = wave.read(wav_path)

    n_window_size = int(window_size_ms * sample_freq)
    n_window_stride = int(window_stride_ms * sample_freq)

    signal = signal / np.abs(signal).max() * hparams.rescaling_max

    # Get mel spectrograms
    mel_features = psf.logfbank(
        signal=signal,
        samplerate=sample_freq,
        winlen=window_size_ms,
        winstep=window_stride_ms,
        nfilt=hparams.num_mels,
        nfft=512,
        lowfreq=0, 
        highfreq=sample_freq/2,
        preemph=0.97,
        winfunc=np.hanning
    )

    # Getting linear spectrograms
    frames = psf.sigproc.framesig(sig=signal,
                                  frame_len=n_window_size,
                                  frame_step=n_window_stride,
                                  winfunc=np.hanning)
    features = psf.sigproc.logpowspec(frames, NFFT=512)
    features *= 2
    n_frames = features.shape[0]

    # mel features in not in db but features is
    mel_features = np.exp(mel_features)
    mel_features = 20 * np.log10(mel_features) - hparams.ref_level_db
    features -= hparams.ref_level_db

    # Write the spectrograms to disk: 
    # Get filename from wav_path
    wav_name = os.path.basename(wav_path)
    wav_name = os.path.splitext(wav_name)[0]
    
    # case if wave files across different speakers have the same naming format.
    # e.g. Recording0.wav
    spectrogram_filename = 'spec-{}-{}.npy'.format(speaker_id, wav_name)
    mel_filename = 'mel-{}-{}.npy'.format(speaker_id, wav_name)

    features = audio._normalize(features)
    mel_features = audio._normalize(mel_features)

    np.save(os.path.join(out_dir, spectrogram_filename), features, allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename), mel_features, allow_pickle=False)

    return (spectrogram_filename, mel_filename, n_frames, text, speaker_id)
