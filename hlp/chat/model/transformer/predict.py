import jieba
import tensorflow as tf
import config.get_config as _config
import common.data_utils as _data
from model.transformer.model import model


def predict(sentence):
    sentence = " ".join(jieba.cut(sentence))
    sentence = _data.preprocess_sentence(sentence)
    inputs = [_data.input_token.word_index.get(i, 3) for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=_config.max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    dec_input = tf.expand_dims([_data.target_token.word_index['start']], 0)

    result = ''
    for t in range(_config.max_length_tar):
        predictions = model(inputs=[inputs, dec_input], training=False)
        print(predictions.shape)
        predictions = predictions[:, -1:, :]
        print(predictions.shape)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        print(predicted_id[0][0].numpy())
        if _data.target_token.index_word.get(predicted_id[0][0].numpy()) == 'end':
            break
        result += _data.target_token.index_word.get(predicted_id, '') + ''
        dec_input = tf.concat([dec_input, predicted_id], axis=-1)

    return result
