import io
import os
import sys
import time
from pathlib import Path
import tensorflow as tf
import model.Seq2Seq.model as model
import config.getConfig as gConfig
from common.data_utils import preprocess_sentence


def create_dataset(path, num_examples):
    is_exist = Path(path)
    if not is_exist.exists():
        file = open(path, 'w', encoding='utf-8')
        file.write('吃饭 了 吗' + '\t' + '吃 了')
        file.close()
    size = os.path.getsize(path)
    lines = io.open(path, encoding='utf-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
    return zip(*word_pairs)


def max_length(tensor):
    return max(len(t) for t in tensor)


def read_data(path, num_examples):
    input_lang, target_lang = create_dataset(path, num_examples)
    input_tensor, input_token = tokenize(input_lang)
    target_tensor, target_token = tokenize(target_lang)
    return input_tensor, input_token, target_tensor, target_token


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=gConfig.vocab_inp_size, oov_token=3)
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=gConfig.max_length_inp, padding='post')

    return tensor, lang_tokenizer


input_tensor, input_token, target_tensor, target_token = read_data(gConfig.data, gConfig.max_train_data_size)


def train():
    steps_per_epoch = len(input_tensor) // gConfig.BATCH_SIZE
    print(steps_per_epoch)
    enc_hidden = model.encoder.initialize_hidden_state()
    checkpoint_dir = gConfig.train_data
    # 这里需要检查一下是否有模型的目录，没有的话就创建，有的话就跳过
    is_exist = Path(checkpoint_dir)
    if not is_exist.exists():
        os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt = tf.io.gfile.listdir(checkpoint_dir)
    if ckpt:
        model.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    BUFFER_SIZE = len(input_tensor)
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(gConfig.BATCH_SIZE, drop_remainder=True)
    checkpoint_dir = gConfig.train_data
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    while True:
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = model.train_step(inp, targ, target_token, enc_hidden)
            total_loss += batch_loss
            print(batch_loss.numpy())
        model.checkpoint.save(file_prefix=checkpoint_prefix)
        sys.stdout.flush()
