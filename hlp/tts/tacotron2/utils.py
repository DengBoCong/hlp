import tensorflow as tf
from dataset.dataset_txt import Dataset_txt
from dataset.dataset_wav import Dataset_wave


def process_text(path):
    # 将文字转化为token
    input_ids, en_tokenizer = Dataset_txt(path)
    vocab_inp_size = len(en_tokenizer.word_index) + 1
    input_ids = tf.convert_to_tensor(input_ids)
    return input_ids, vocab_inp_size


# 对声音的处理
# path = r"./wavs/"
def process_wave(path):
    # 提取Mel频谱
    mel_gts = Dataset_wave(path)
    mel_gts = tf.cast(mel_gts, tf.float32)
    return mel_gts


# tf.data
def tf_data(batch_size, input_ids, mel_gts):
    BUFFER_SIZE = len(input_ids)
    steps_per_epoch = BUFFER_SIZE // batch_size
    dataset = tf.data.Dataset.from_tensor_slices((input_ids, mel_gts)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # input_ids, mel_gts = next(iter(dataset))
    return dataset, steps_per_epoch
