import numpy as np
import tensorflow as tf

from preprocess import read_data, label_2_float

# 数据生成器
def generator(wav_name_list, batch_size, sample_rate, peak_norm, voc_mode, bits, mu_law, wave_path, voc_pad, hop_length,
              voc_seq_len, preemphasis, n_fft, n_mels, win_length, max_db, ref_db, top_db):
    # generator只能进行一次生成，故需要while True来进行多个epoch的数据生成
    while True:
        # 每epoch将所有数据进行一次shuffle
        #order = np.random.choice(len(wav_name_list), len(wav_name_list), replace=False)
        #audio_data_path_list = [wav_name_list[i] for i in order]
        audio_data_path_list = wav_name_list
        batchs = len(wav_name_list)//batch_size
        for idx in range(batchs):
            #逐步取音频名
            wav_name_list2 = audio_data_path_list[idx * batch_size: (idx + 1) * batch_size]

            #取音频数据
            input_mel, input_sig = read_data(
                wave_path, sample_rate, peak_norm, voc_mode, bits, mu_law, wav_name_list2, preemphasis, n_fft, n_mels,
                hop_length, win_length, max_db, ref_db, top_db
            )

            dataset = collate_vocoder(input_mel, input_sig, voc_seq_len, hop_length, voc_pad, voc_mode, bits)
            # input_mel = tf.convert_to_tensor(input_mel[0])
            # input_sig = tf.convert_to_tensor(input_sig[0])
        yield dataset

def collate_vocoder(input_mel: tf.Tensor, input_sig: tf.Tensor, voc_seq_len, hop_length, voc_pad, voc_mode, bits):
    mel_win = voc_seq_len // hop_length + 2 * voc_pad
    max_offsets = [tf.shape(x)[-1] - 2 - (mel_win + 2 * voc_pad) for x in input_mel]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + voc_pad) * hop_length for offset in mel_offsets]

    mels = [x[:][mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(input_mel)]
    #mels = [x[:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(input_mel)]

    labels = [x[sig_offsets[i]:sig_offsets[i] + voc_seq_len + 1] for i, x in enumerate(input_sig)]

    mels = np.stack(mels).astype(np.float32)
    labels = np.stack(labels).astype(np.int64)

    mels = tf.convert_to_tensor(mels)
    labels = tf.convert_to_tensor(labels)

    x = labels[:, :voc_seq_len]
    y = labels[:, 1:]
    bits = 16 if voc_mode == 'MOL' else bits

    x = label_2_float(tf.cast(x, dtype=float), bits)

    if voc_mode == 'MOL':
        y = label_2_float(tf.cast(y, dtype=float), bits)
    dataset = [x, y, mels]
    return dataset

