import librosa
import librosa.filters
import math
import numpy as np
from scipy import signal
from hparams import hparams
from scipy.io import wavfile

import lws


def load_wav(path):
    return librosa.core.load(path, sr=hparams.sample_rate)[0]


def save_wav(wav, path):
    wav = wav * 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, hparams.sample_rate, wav.astype(np.int16))


def preemphasis(x):
    from nnmnkwii.preprocessing import preemphasis
    return preemphasis(x, hparams.preemphasis)


def inv_preemphasis(x):
    from nnmnkwii.preprocessing import inv_preemphasis
    return inv_preemphasis(x, hparams.preemphasis)


def spectrogram(y):
    D = _lws_processor().stft(preemphasis(y)).T
    S = _amp_to_db(np.abs(D)) - hparams.ref_level_db
    return _normalize(S)


def inv_spectrogram(spectrogram):
    '''Converts spectrogram to waveform using librosa'''
    S = _db_to_amp(_denormalize(spectrogram) + hparams.ref_level_db)  # Convert back to linear
    processor = _lws_processor()
    D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
    y = processor.istft(D).astype(np.float32)
    return inv_preemphasis(y)

def melspectrogram(y):
    D = _lws_processor().stft(preemphasis(y)).T
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hparams.ref_level_db
    if not hparams.allow_clipping_in_normalization:
        assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0
    return _normalize(S)


def _lws_processor():
    return lws.lws(hparams.fft_size, hparams.hop_size, mode="speech")


# Conversions:


_mel_basis = None


def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _build_mel_basis():
    if hparams.fmax is not None:
        assert hparams.fmax <= hparams.sample_rate // 2
    return librosa.filters.mel(hparams.sample_rate, hparams.fft_size,
                               fmin=hparams.fmin, fmax=hparams.fmax,
                               n_mels=hparams.num_mels)


def _amp_to_db(x):
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize(S):
    return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)


def _denormalize(S):
    return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db

def jasper_inverse_mel(log_mel, fs, n_fft, n_mels, power=2.0, htk=True):
    mel_spec = 10 ** (log_mel / 20)
    mel_basis = librosa.filters.mel(fs, n_fft, n_mels=n_mels, htk=htk)
    mag_spec = np.dot(mel_spec, mel_basis)
    mag_spec = np.power(mag_spec, 1. / power)
    return mag_spec

def jasper_griffin_lim(mag, n_iters=50, n_fft=512):
    phase = np.exp(2j * np.pi * np.random.rand(*mag.shape))
    complex_spec = mag * phase
    signal = librosa.istft(complex_spec, hop_length=160, win_length=320)
    for _ in range(n_iters):
        _, phase = librosa.magphase(librosa.stft(signal, n_fft=n_fft, hop_length=160, win_length=320))
        complex_spec = mag * phase
        signal= librosa.istft(complex_spec, hop_length=160, win_length=320)
    return signal

def jasper_get_mag_spec(spec):
    return (10 ** (spec / 20) * 512) ** 0.5
