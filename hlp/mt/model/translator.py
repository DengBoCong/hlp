"""
对输出的句子进行翻译
"""
import tensorflow as tf

from hlp.mt.config import get_config as _config
from hlp.mt.model import transformer as _transformer
from hlp.mt.model import checkpoint
from hlp.mt.common import preprocess, text_vectorize
from hlp.utils import beamsearch
from hlp.utils import optimizers as _optimizers


def _predict_index(inp_sentence, transformer, beam_search_container, input_tokenizer, target_tokenizer):
    """对输入句子进行翻译并返回编码的句子列表"""
    sentence = preprocess.preprocess_sentences([inp_sentence], language=_config.source_lang)

    inp_sequence, _ = text_vectorize.encode_sentences(sentence, input_tokenizer, language=_config.source_lang)
    inp_sequence = tf.squeeze(inp_sequence)
    inp_sequence = tf.expand_dims(inp_sequence, 0)

    # start_token  shape:(1,)
    start_token = text_vectorize.get_start_token(_config.start_word, target_tokenizer, language=_config.target_lang)
    end_token, _ = text_vectorize.encode_sentences([_config.end_word], target_tokenizer, language=_config.target_lang)
    end_token = tf.squeeze(end_token)

    decoder_input = tf.expand_dims(start_token, 0)  # shape --> (1,1) 即(batch_size,sentence_length)

    beam_search_container.reset(inputs=inp_sequence, dec_input=decoder_input)
    inputs, decoder_input = beam_search_container.get_search_inputs()
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
        beam_search_container.expand(predictions=predictions, end_sign=end_token)
        if beam_search_container.beam_size == 0:
            break
        # predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        inputs, decoder_input = beam_search_container.get_search_inputs()
    beam_search_result = beam_search_container.get_result()

    return beam_search_result


def translate(sentence, transformer, tokenizer_source, tokenizer_target, beam_size=_config.BEAM_SIZE):
    """对句子(经过预处理未经过编码)进行翻译,未进行检查点的判断"""
    beam_search_container = beamsearch.BeamSearch(
        beam_size=beam_size,
        max_length=_config.max_target_length,
        worst_score=0)

    # 对检查点进行恢复
    learning_rate = _optimizers.CustomSchedule(_config.d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # 采用checkpoint_ensembling,获取需要使用的检查点路径列表
    checkpoints_path = checkpoint.get_checkpoints_path()
    if _config.checkpoint_ensembling == "False":
        checkpoints_path = checkpoints_path[-1:]

    # 使用列表中检查点进行预测
    for checkpoint_path in checkpoints_path:
        checkpoint.load_checkpoint(transformer, optimizer, checkpoint_path=checkpoint_path)

    predict_idxes = _predict_index(sentence, transformer, beam_search_container, tokenizer_source, tokenizer_target)
    predicted_sentences = []
    # 从容器中抽取序列，生成最终结果
    for i in range(len(predict_idxes)):
        predict_idx = predict_idxes[i].numpy()
        predict_idx = tf.squeeze(predict_idx)
        predict_sentence = text_vectorize.decode_sentence(predict_idx, tokenizer_target, language=_config.target_lang)
        # text[0] = text[0].replace('start', '').replace('end', '').replace(' ', '')
        predict_sentence = predict_sentence.replace(_config.start_word, '')\
            .replace(_config.end_word, '').strip()
        predicted_sentences.append(predict_sentence)
    # predicted_sentence = preprocess.decode_sentence(predict_idx, target_tokenizer, _config.zh_tokenize_type)
    return predicted_sentences
