import matplotlib.pyplot as plt
import librosa.display
import librosa
import numpy as np
import copy
import scipy
from config2 import Tacotron2Config

config=Tacotron2Config()

def melspectrogram2wav(mel):
    '''# Generate wave file from spectrogram'''
    # transpose
    #mel = mel.T

    # de-noramlize
    mel = (np.clip(mel, 0, 1) * config.max_db) - config.max_db + config.ref_db

    # to amplitude
    mel = np.power(10.0, mel * 0.05)
    m = _mel_to_linear_matrix(config.sr, config.n_fft, config.n_mels)
    mag = np.dot(m, mel)

    # wav reconstruction
    wav = griffin_lim(mag)

    # de-preemphasis
    wav = scipy.signal.lfilter([1], [1, -config.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)

def _mel_to_linear_matrix(sr, n_fft, n_mels):
    m = librosa.filters.mel(sr, n_fft, n_mels)
    m_t = np.transpose(m)
    p = np.matmul(m, m_t)
    d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
    return np.matmul(m_t, np.diag(d))

def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.
    '''
    X_best = copy.deepcopy(spectrogram)
    for i in range(config.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, config.n_fft, config.hop_length, win_length=config.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y


def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, config.hop_length, win_length=config.win_length, window="hann")



def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data
