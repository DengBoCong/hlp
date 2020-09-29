"""
对输出的句子进行翻译
"""
from model import network
import tensorflow as tf
from config import get_config as _config
from common import self_attention
import nmt


def predict_index(inp_sentence, transformer, input_pre, target_pre):
    """对输入句子进行翻译并返回编码的句子列表"""
    inp_sequence = input_pre.encode_sentence(inp_sentence)
    encoder_input = tf.expand_dims(inp_sequence, 0)

    decoder_input = [target_pre.tokenizer.word_index[_config.start_word]]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(_config.max_target_length):
        enc_padding_mask, combined_mask, dec_padding_mask = self_attention.create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, _ = transformer(encoder_input,
                                             output,
                                             False,
                                             enc_padding_mask,
                                             combined_mask,
                                             dec_padding_mask)

        # 从 seq_len 维度选择最后一个词
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # 如果 predicted_id 等于结束标记，就返回结果
        if predicted_id == [target_pre.tokenizer.word_index[_config.end_word]]:
            return tf.squeeze(output, axis=0)

        # 连接 predicted_id 与输出，作为解码器的输入传递到解码器。
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)[1:]


def translate(sentence, transformer, input_pre, target_pre):
    """对句子(经过预处理未经过编码)进行翻译,未进行检查点的判断"""
    predict_idx = predict_index(sentence, transformer, input_pre, target_pre)
    predicted_sentence = target_pre.decode_sequence(predict_idx)
    return predicted_sentence


if __name__ == '__main__':
    translate('i love you')