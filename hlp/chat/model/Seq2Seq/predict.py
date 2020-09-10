import os
from pathlib import Path
import tensorflow as tf
import config.getConfig as gConfig
import model.Seq2Seq.model as model
from common.data_utils import preprocess_sentence
import model.Seq2Seq.trainer as seq


def predict(sentence):
    checkpoint_dir = gConfig.train_data
    # print(checkpoint_dir)
    pExist = Path(checkpoint_dir)
    if not pExist.exists():
        os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt = tf.io.gfile.listdir(checkpoint_dir)
    if ckpt:
        model.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    else:
        print('请先训练再进行测试体验，训练轮数建议一百轮以上!')
        return
    sentence = preprocess_sentence(sentence)
    inputs = [seq.input_token.word_index.get(i, 3) for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=gConfig.max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, gConfig.units))]
    enc_out, enc_hidden = model.encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([seq.target_token.word_index['start']], 0)

    for t in range(gConfig.max_length_tar):
        predictions, dec_hidden, attention_weights = model.decoder(dec_input, dec_hidden, enc_out)

        predicted_id = tf.argmax(predictions[0]).numpy()

        if seq.target_token.index_word[predicted_id] == 'end':
            break
        result += seq.target_token.index_word[predicted_id] + ' '

        dec_input = tf.expand_dims([predicted_id], 0)

    return result
