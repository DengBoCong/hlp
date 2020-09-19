import jieba
import tensorflow as tf
import common.data_utils as _data
import config.get_config as _config

def predict(sentence, model):
    _, input_token, _, target_token = _data.load_dataset()
    sentence = ' '.join(jieba.cut(sentence))
    sentence = _data.preprocess_sentence(sentence)
    inputs = [input_token.word_index.get(i, 3) for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],  maxlen=_config.max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    dec_input = tf.expand_dims([target_token.word_index['start']], 0)
    result = ''

    for t in range(_config.max_length_tar):
        predictions = model(inputs=[inputs, dec_input], training=False)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        if target_token.index_word.get(predicted_id[0][0].numpy()) == 'end':
            break
        result += target_token.index_word.get(predicted_id[0][0].numpy(), '')
        dec_input = tf.concat([dec_input, predicted_id], axis=-1)

    return result