import os
import jieba
import tensorflow as tf
import config.get_config as _config
import common.data_utils as _data
from common.data_utils import load_dataset


def predict(sentence, model):
    """
    seq2seq的预测方法，每次一个输入一个输出，输出的就是处理好的完整的句子
    :param sentence:
    :param model:
    :return: answer
    """
    # 在这里，我们需要对输入的语句进行文本处理
    _, input_token, _, target_token = load_dataset()
    sentence = " ".join(jieba.cut(sentence))
    sentence = _data.preprocess_sentence(sentence)
    inputs = [input_token.word_index.get(i, 3) for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=_config.max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    result = ''

    # 初始化隐藏层，并使用encoder得到隐藏层和decoder的输入给decoder使用
    hidden = [tf.zeros((1, _config.units))]
    enc_out, enc_hidden = model.encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_token.word_index['start']], 0)
    print(inputs.shape)
    for t in range(_config.max_length_tar):
        predictions, dec_hidden, attention_weights = model.decoder(dec_input, dec_hidden, enc_out)
        print(predictions.shape)
        return 'ni'
        predicted_id = tf.argmax(predictions[0]).numpy()
        # 这里就是做一下判断，当预测结果解码是end，说明预测结束
        if target_token.index_word.get(predicted_id) == 'end':
            break
        # 把相关结果拼接起来
        result += target_token.index_word.get(predicted_id, '')
        # 这里就是更新一下decoder的输入，因为要将前面预测的结果反过来
        # 作为输入丢到decoder里去
        dec_input = tf.expand_dims([predicted_id], 0)

    return result
