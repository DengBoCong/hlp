import os
import jieba
from pathlib import Path
import tensorflow as tf
import config.get_config as _config
import model.seq2seq.model as model
import common.data_utils as _data
import model.seq2seq.trainer as seq


def predict(sentence):
    checkpoint_dir = _config.train_data

    # 这里需要检查一下是否有模型的目录，没有的话就创建，有的话就跳过
    is_exist = Path(checkpoint_dir)
    if not is_exist.exists():
        os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt = tf.io.gfile.listdir(checkpoint_dir)
    if ckpt:
        model.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    else:
        print('请先训练再进行测试体验，训练轮数建议一百轮以上!')
        return

    # 在这里，我们需要对输入的语句进行文本处理
    sentence = " ".join(jieba.cut(sentence))
    sentence = _data.preprocess_sentence(sentence)
    inputs = [_data.input_token.word_index.get(i, 3) for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=_config.max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    result = ''

    hidden = [tf.zeros((1, _config.units))]
    enc_out, enc_hidden = model.encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([_data.target_token.word_index['start']], 0)

    for t in range(_config.max_length_tar):
        predictions, dec_hidden, attention_weights = model.decoder(dec_input, dec_hidden, enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        if _data.target_token.index_word.get(predicted_id) == 'end':
            break

        # 把相关结果拼接起来
        result += _data.target_token.index_word.get(predicted_id, '') + ''

        dec_input = tf.expand_dims([predicted_id], 0)

    return result
