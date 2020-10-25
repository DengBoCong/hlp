import os

import tensorflow as tf

sr = 22050  # Sample rate.
n_fft = 2048  # fft points (samples)
frame_shift = 0.0125  # seconds
frame_length = 0.05  # seconds
hop_length = int(sr * frame_shift)  # samples.
win_length = int(sr * frame_length)  # samples.
n_mels = 80  # Number of Mel banks to generate
power = 1.2  # Exponent for amplifying the predicted magnitude
n_iter = 100  # Number of inversion iterations
preemphasis = .97  # or None
max_db = 100
ref_db = 20
top_db = 15
import librosa
import numpy as np
from model.plot import plot_mel


def get_spectrograms(fpath):
    '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
      sound_file: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
 '''
    # Loading sound file
    y, sr = librosa.load(fpath, sr=None)

    # Trimming
    y, _ = librosa.effects.trim(y, top_db=top_db)

    # Preemphasis
    y = np.append(y[0], y[1:] - preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels)  # (n_mels, 1+n_fft//2)
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

    return mel, mag


mel_list = []


def Dataset_wave(path):
    x = tf.constant(0, shape=(1, 500, 80))
    dirs = os.listdir(path)
    for file in dirs:
        # y, sr = librosa.load(path+file, sr=None)
        # melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=512, n_mels=80)
        # logmelspec = librosa.power_to_db(melspec)
        logmelspec, sr = get_spectrograms(path + file)
        # logmelspec = np.exp(logmelspec)
        # logmelspec = 10*logmelspec

        # print("logmelspec::",logmelspec.shape)
        # logmelspec = tf.transpose(logmelspec, [1, 0])
        print(logmelspec)
        # if i==0:
        #     print(logmelspec)
        mel_list.append(logmelspec.tolist())
        # logmelspec = tf.keras.preprocessing.sequence.pad_sequences(logmelspec, maxlen=1000, padding='post', value=0.0)
        # if i==0:
        #     print(logmelspec)
        # logmelspec = tf.transpose(logmelspec, [1, 0])
        #
        # logmelspec = tf.expand_dims(logmelspec, 0)
        # if i == 0:
        #     x = logmelspec
        #     i = i + 1
        #     continue
        # x = tf.concat([x, logmelspec], 0)
        # print("logmelspec",logmelspec.shape)
    # print(mel_list)
    mel_numpy = tf.keras.preprocessing.sequence.pad_sequences(mel_list, maxlen=1000, padding='post', value=0.0,
                                                              dtype='float32')
    print(mel_numpy)
    inputs = tf.convert_to_tensor(mel_numpy)
    # print(inputs)
    plot_mel(inputs)
    return inputs

#
# wav = melspectrogram2wav(logmelspec)
# wave.write('4.wav', rate=sr, data=wav)
