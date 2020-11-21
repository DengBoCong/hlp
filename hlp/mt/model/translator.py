"""
对输出的句子进行翻译
"""
import tensorflow as tf
from config import get_config as _config
from model import transformer as _transformer
from common import preprocess
from utils import beamsearch
from common import tokenize


def _predict_index(inp_sentence, transformer, beam_search_container, input_tokenizer, target_tokenizer):
    """对输入句子进行翻译并返回编码的句子列表"""
    sentence = preprocess.preprocess_sentences([inp_sentence], language=_config.source_lang)

    inp_sequence, _ = tokenize.encode_sentences(sentence, input_tokenizer, language=_config.source_lang)
    inp_sequence = tf.squeeze(inp_sequence)
    inp_sequence = tf.expand_dims(inp_sequence, 0)

    # start_token  shape:(1,)
    start_token = tokenize.get_start_token(_config.start_word, target_tokenizer, language=_config.target_lang)
    end_token, _ = tokenize.encode_sentences([_config.end_word], target_tokenizer, language=_config.target_lang)
    end_token = tf.squeeze(end_token)

    decoder_input = tf.expand_dims(start_token, 0)  # shape --> (1,1) 即(batch_size,sentence_length)

    beam_search_container.reset(inputs=inp_sequence, dec_input=decoder_input)
    inputs, decoder_input = beam_search_container.expand_beam_size_inputs()
    for i in range(_config.max_target_length):
        enc_padding_mask, combined_mask, dec_padding_mask = _transformer.create_masks(inputs, decoder_input)

        # predictions.shape == (batch_size, s.eq_len, vocab_size)
        predictions, _ = transformer(inputs,
                                     decoder_input,
                                     False,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)

        # 从 seq_len 维度选择最后一个词
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
        predictions = tf.squeeze(predictions, axis=1)
        beam_search_container.add(predictions=predictions, end_sign=end_token)
        if beam_search_container.beam_size == 0:
            break
        # predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        inputs, decoder_input = beam_search_container.expand_beam_size_inputs()
    beam_search_result = beam_search_container.get_result()

    return beam_search_result


def translate(sentence, transformer, tokenizer_source, tokenizer_target, beam_size=_config.BEAM_SIZE):
    """对句子(经过预处理未经过编码)进行翻译,未进行检查点的判断"""
    beam_search_container = beamsearch.BeamSearch(
        beam_size=beam_size,
        max_length=_config.max_target_length,
        worst_score=0)
    predict_idxes = _predict_index(sentence, transformer, beam_search_container, tokenizer_source, tokenizer_target)
    predicted_sentences = []
    # 从容器中抽取序列，生成最终结果
    for i in range(len(predict_idxes)):
        predict_idx = predict_idxes[i].numpy()
        predict_idx = tf.squeeze(predict_idx)
        predict_sentence = tokenize.decode_sentence(predict_idx, tokenizer_target, language=_config.target_lang)
        # text[0] = text[0].replace('start', '').replace('end', '').replace(' ', '')
        predict_sentence = predict_sentence.replace(_config.start_word, '')\
            .replace(_config.end_word, '').strip()
        predicted_sentences.append(predict_sentence)
    # predicted_sentence = preprocess.decode_sentence(predict_idx, target_tokenizer, _config.zh_tokenize_type)
    return predicted_sentences
