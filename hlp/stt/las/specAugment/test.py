# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 08:21:18 2020

@author: 九童
"""
"""SpecAugment test"""

import argparse
import librosa
from hlp.stt.las.specAugment import spec_augment
import numpy as np

parser = argparse.ArgumentParser(description='Spec Augment')
parser.add_argument('--audio-path', default='../data/number/wav_test/0_jackson_1.wav',
                    help='The audio file.')
parser.add_argument('--time-warp-para', default=80,
                    help='time warp parameter W')
parser.add_argument('--frequency-mask-para', default=100,
                    help='frequency mask parameter F')
parser.add_argument('--time-mask-para', default=27,
                    help='time mask parameter T')
parser.add_argument('--masking-line-number', default=1,
                    help='masking line number')

args = parser.parse_args()
audio_path = args.audio_path
time_warping_para = args.time_warp_para
time_masking_para = args.frequency_mask_para
frequency_masking_para = args.time_mask_para
masking_line_number = args.masking_line_number

if __name__ == "__main__":
    audio, sampling_rate = librosa.load(audio_path)

    mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                     sr=sampling_rate,
                                                     n_mels=256,
                                                     hop_length=128,
                                                     fmax=8000)

    # reshape spectrogram shape to [batch_size, time, frequency, 1]
    shape = mel_spectrogram.shape

    mel_spectrogram = np.reshape(mel_spectrogram, (-1, shape[0], shape[1], 1))

    # Show Raw mel-spectrogram
    spec_augment.visualization_spectrogram(mel_spectrogram=mel_spectrogram,
                                           title="Raw Mel Spectrogram")

    # Show time warped & masked spectrogram
    spec_augment.visualization_spectrogram(mel_spectrogram=spec_augment.spec_augment(mel_spectrogram),
                                           title="tensorflow Warped & Masked Mel Spectrogram")
