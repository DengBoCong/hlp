#提取音频的mel-spectrogram（得到声谱和幅度谱）
import os
import librosa
import numpy as np
sr = 24000 # Sample rate.
n_fft = 2048 # fft points (samples)
frame_shift = 0.0125 # seconds
frame_length = 0.05 # seconds
hop_length = int(sr*frame_shift) # samples.
win_length = int(sr*frame_length) # samples.
n_mels = 512 # Number of Mel banks to generate
power = 1.2 # Exponent for amplifying the predicted magnitude
n_iter = 100 # Number of inversion iterations
preemphasis = .97 # or None
max_db = 100
ref_db = 20
top_db = 15
def get_spectrograms(fpath):
    '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
        sound_file: A string. The full path of a sound file.
    Returns:
        mel: A 2d array of shape (T, n_mels) <- Transposed
        mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
    '''
    # Loading sound file
    y, x = librosa.load(fpath, sr=sr)

    # Trimming
    y, _ = librosa.effects.trim(y, top_db=top_db)

    # Preemphasis
    y = np.append(y[0], y[1:] - preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,n_fft=n_fft,hop_length=hop_length,
                              win_length=win_length)
    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(x, n_fft, n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
    mag = np.clip((mag - ref_db + max_db) / max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)
    return(mel,mag)

path="E:\LJSpeech-1.1\LJSpeech-1.1\wavs"
dirs = os.listdir(path)
for file in dirs:
    get_spectrograms(file)