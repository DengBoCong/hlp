import os
import librosa
import numpy as np
import copy
from scipy import signal


def get_padded_spectros(filepath, r, preemphasis, n_fft,
                        hop_length, win_length, sampling_rate,
                        n_mel, ref_db, max_db):
    filename = os.path.basename(filepath)
    mel_spectro, spectro = get_spectros(filepath, preemphasis, n_fft,
                                        hop_length, win_length, sampling_rate,
                                        n_mel, ref_db, max_db)

    # print(mel_spectro.shape, spectro.shape)

    t = mel_spectro.shape[0]  # frame数
    nb_paddings = r - (t % r) if t % r != 0 else 0  # 需要填充的frame数
    mel_spectro = np.pad(mel_spectro,
                         [[0, nb_paddings], [0, 0]],
                         mode="constant")
    spectro = np.pad(spectro,
                     [[0, nb_paddings], [0, 0]],
                     mode="constant")

    print('[PAD] mel_spectro, spectro: ', mel_spectro.shape, spectro.shape)
    return filename, mel_spectro.reshape((-1, n_mel * r)), spectro  # mel谱r帧一组


def get_spectros(filepath, preemphasis, n_fft,
                 hop_length, win_length,
                 sampling_rate, n_mel,
                 ref_db, max_db):
    waveform, sampling_rate = librosa.load(filepath,
                                           sr=sampling_rate)

    # 去掉两头的silence
    waveform, _ = librosa.effects.trim(waveform)

    # 使用预加重过滤低频，加重高频
    waveform = np.append(waveform[0],
                         waveform[1:] - preemphasis * waveform[:-1])

    # 短时傅里叶变换, shape=(1 + n_fft/2, n_frames)，每个频率f在每帧t的幅度和相位，复数表示
    stft_matrix = librosa.stft(y=waveform,
                               n_fft=n_fft,
                               hop_length=hop_length,
                               win_length=win_length)
    print('stft_matrix: ', stft_matrix.shape)

    # 每个频率每帧的幅度, 线性谱, shape=(1 + n_fft/2, n_frames)
    spectro = np.abs(stft_matrix)
    print('spectro:', spectro.shape)

    # 梅尔FBank（filter-bank）矩阵, 线性转换矩阵， shape=(n_mels, 1 + n_fft/2)
    mel_transform_matrix = librosa.filters.mel(sampling_rate, n_fft, n_mel)
    print('mel_transform_matrix:', mel_transform_matrix.shape)

    # 点积，能量, mel谱, shape=(n_mels, n_frames)
    mel_spectro = np.dot(mel_transform_matrix, spectro)
    print('mel_spectro:', mel_spectro.shape)

    # 取对数，使用分贝decidel刻度
    mel_spectro = 20 * np.log10(np.maximum(1e-5, mel_spectro))
    spectro = 20 * np.log10(np.maximum(1e-5, spectro))

    # 规范化
    mel_spectro = np.clip((mel_spectro - ref_db + max_db) / max_db, 1e-8, 1)
    spectro = np.clip((spectro - ref_db + max_db) / max_db, 1e-8, 1)

    # 转置使得时间为第一维，频率为第二维
    mel_spectro = mel_spectro.T.astype(np.float32)
    spectro = spectro.T.astype(np.float32)

    print('mel_spectro, spectro: ', mel_spectro.shape, spectro.shape)

    return mel_spectro, spectro


def get_griffin_lim(spectrogram, n_fft, hop_length,
                    win_length, window_type, n_iter):

    spectro = copy.deepcopy(spectrogram)
    for i in range(n_iter):
        estimated_wav = spectro_inversion(spectro, hop_length,
                                          win_length, window_type)
        est_stft = librosa.stft(estimated_wav, n_fft,
                                hop_length,
                                win_length=win_length)
        phase = est_stft / np.maximum(1e-8, np.abs(est_stft))
        spectro = spectrogram * phase
    estimated_wav = spectro_inversion(spectro, hop_length,
                                      win_length, window_type)
    result = np.real(estimated_wav)

    return result


def spectro_inversion(spectrogram, hop_length, win_length, window_type):
    return librosa.istft(spectrogram, hop_length, win_length=win_length, window=window_type)


def from_spectro_to_waveform(spectro, n_fft, hop_length,
                             win_length, n_iter, window_type,
                             max_db, ref_db, preemphasis):
    # transpose
    spectro = spectro.T

    # de-noramlize
    spectro = (np.clip(spectro, 0, 1) * max_db) - max_db + ref_db

    # to amplitude
    spectro = np.power(10.0, spectro * 0.05)

    # wav reconstruction
    waveform = get_griffin_lim(spectro, n_fft, hop_length,
                               win_length,
                               window_type, n_iter)

    # de-preemphasis
    waveform = signal.lfilter([1], [1, -preemphasis], waveform)

    # trim
    waveform, _ = librosa.effects.trim(waveform)

    return waveform.astype(np.float32)
