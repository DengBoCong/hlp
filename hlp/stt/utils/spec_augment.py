import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow_addons.image import sparse_image_warp


def sparse_warp(mel_spectrogram, time_warping_para=80):
    fbank_size = tf.shape(mel_spectrogram)
    n, v = fbank_size[1], fbank_size[2]

    # Image warping control point setting.
    # Source
    pt = tf.random.uniform([], time_warping_para, n - time_warping_para, tf.int32)  # radnom point along the time axis
    src_ctr_pt_freq = tf.range(v // 2)  # control points on freq-axis
    src_ctr_pt_time = tf.ones_like(src_ctr_pt_freq) * pt  # control points on time-axis
    src_ctr_pts = tf.stack((src_ctr_pt_time, src_ctr_pt_freq), -1)
    src_ctr_pts = tf.cast(src_ctr_pts, dtype=tf.float32)

    # Destination
    w = tf.random.uniform([], -time_warping_para, time_warping_para, tf.int32)  # distance
    dest_ctr_pt_freq = src_ctr_pt_freq
    dest_ctr_pt_time = src_ctr_pt_time + w
    dest_ctr_pts = tf.stack((dest_ctr_pt_time, dest_ctr_pt_freq), -1)
    dest_ctr_pts = tf.cast(dest_ctr_pts, dtype=tf.float32)

    # warp
    source_control_point_locations = tf.expand_dims(src_ctr_pts, 0)  # (1, v//2, 2)
    dest_control_point_locations = tf.expand_dims(dest_ctr_pts, 0)  # (1, v//2, 2)

    warped_image, _ = sparse_image_warp(mel_spectrogram,
                                        source_control_point_locations,
                                        dest_control_point_locations)
    return warped_image


def frequency_masking(mel_spectrogram, v, frequency_masking_para=27, frequency_mask_num=2):
    fbank_size = tf.shape(mel_spectrogram)
    n, v = fbank_size[1], fbank_size[2]

    for i in range(frequency_mask_num):
        f = tf.random.uniform([], minval=0, maxval=frequency_masking_para, dtype=tf.int32)
        v = tf.cast(v, dtype=tf.int32)
        f0 = tf.random.uniform([], minval=0, maxval=v - f, dtype=tf.int32)

        # warped_mel_spectrogram[f0:f0 + f, :] = 0
        mask = tf.concat((tf.ones(shape=(1, n, v - f0 - f, 1)),
                          tf.zeros(shape=(1, n, f, 1)),
                          tf.ones(shape=(1, n, f0, 1)),
                          ), 2)
        mel_spectrogram = mel_spectrogram * mask

    return tf.cast(mel_spectrogram, dtype=tf.float32)


def time_masking(mel_spectrogram, tau, time_masking_para=100, time_mask_num=2):
    fbank_size = tf.shape(mel_spectrogram)
    n, v = fbank_size[1], fbank_size[2]

    for i in range(time_mask_num):
        t = tf.random.uniform([], minval=0, maxval=time_masking_para, dtype=tf.int32)
        t0 = tf.random.uniform([], minval=0, maxval=tau - t, dtype=tf.int32)

        # mel_spectrogram[:, t0:t0+t] = 0
        mask = tf.concat((tf.ones(shape=(1, n - t0 - t, v, 1)),
                          tf.zeros(shape=(1, t, v, 1)),
                          tf.ones(shape=(1, t0, v, 1)),), 1)
        mel_spectrogram = mel_spectrogram * mask

    return tf.cast(mel_spectrogram, dtype=tf.float32)


def spec_augment(mel_spectrogram):
    v = mel_spectrogram.shape[0]
    tau = mel_spectrogram.shape[1]

    warped_mel_spectrogram = sparse_warp(mel_spectrogram)

    warped_frequency_spectrogram = frequency_masking(warped_mel_spectrogram, v=v)

    warped_frequency_time_sepctrogram = time_masking(warped_frequency_spectrogram, tau=tau)

    return warped_frequency_time_sepctrogram


def _plot_spectrogram(mel_spectrogram, title):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram[0, :, :, 0], ref=np.max),
                             y_axis='mel', fmax=8000,
                             x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description='Spec Augment')
    parser.add_argument('--audio-path', default='../data/data_thchs30/data/A2_0.wav',
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

    audio, sampling_rate = librosa.load(audio_path)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                     sr=sampling_rate,
                                                     n_mels=256,
                                                     hop_length=128,
                                                     fmax=8000)

    # reshape spectrogram shape to [batch_size, time, frequency, 1]
    shape = mel_spectrogram.shape
    mel_spectrogram = np.reshape(mel_spectrogram, (-1, shape[0], shape[1], 1))

    _plot_spectrogram(mel_spectrogram=mel_spectrogram,
                      title="Raw Mel Spectrogram")

    _plot_spectrogram(
        mel_spectrogram=spec_augment(mel_spectrogram),
        title="tensorflow Warped & Masked Mel Spectrogram")
