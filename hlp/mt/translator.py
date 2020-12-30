"""
对输出的句子进行翻译
"""
import tensorflow as tf
import copy

import hlp.mt.common.text_vectorize
from hlp.mt.config import get_config as _config
from hlp.mt.model import transformer as _transformer
from hlp.mt.model import checkpoint
from hlp.mt.common import text_vectorize
from hlp.utils import beamsearch
from hlp.mt.common import text_split
from hlp.mt import preprocess


def _checkpoint_ensembling(checkpoints_path, model, inputs, decoder_input):
    """
    使用路径中的检查点得到此步的predictions
    @param checkpoints_path: 使用的检查点路径列表
    @param model: 模型
    @param inputs: 输入
    @param decoder_input: 解码器输入
    @param enc_padding_mask: 编码器遮挡
    @param combined_mask: 遮挡
    @param dec_padding_mask: 解码器遮挡
    @return:使用多个检查点模型后的平均predictions
    """
    # 首先使用首个检查点模型得到结果
    enc_padding_mask, combined_mask, dec_padding_mask = _transformer.create_masks(inputs, decoder_input)
    checkpoint_path = checkpoints_path[0]
    checkpoint.load_checkpoint(model, tf.keras.optimizers.Adam(), checkpoint_path=checkpoint_path)
    predictions, _ = model(inputs, decoder_input, False, enc_padding_mask, combined_mask, dec_padding_mask)
    # 从 seq_len 维度选择最后一个词
    predictions = tf.squeeze(predictions[:, -1:, :], axis=1)  # (batch_size, vocab_size)
    predictions_sum = copy.deepcopy(predictions)
    if len(checkpoints_path) > 1:
        for i in range(len(checkpoints_path)-1):  # 分别读取n个检查点模型并预测得到predictions进行累加
            checkpoint_path = checkpoints_path[i+1]
            checkpoint.load_checkpoint(model, tf.keras.optimizers.Adam(), checkpoint_path=checkpoint_path)
            predictions, _ = model(inputs, decoder_input, False, enc_padding_mask, combined_mask, dec_padding_mask)
            predictions = tf.squeeze(predictions[:, -1:, :], axis=1)  # (batch_size, vocab_size)
            predictions_sum = tf.add(predictions_sum, predictions)
    predictions_avg = tf.divide(predictions_sum, len(checkpoints_path))

    return predictions_avg


def _predict_index(checkpoints_path, inp_sentence, model, beam_search_container, input_tokenizer, target_tokenizer):
    """对输入句子进行翻译并返回编码的句子列表"""
    input_mode = hlp.mt.common.text_vectorize.get_tokenizer_mode(_config.source_lang)
    target_mode = hlp.mt.common.text_vectorize.get_tokenizer_mode(_config.target_lang)

    sentence = text_split.preprocess_sentences([inp_sentence], _config.source_lang, input_mode)

    inp_sequence, _ = text_vectorize.encode_sentences(sentence, input_tokenizer,
                                                      language=_config.source_lang, mode=input_mode)
    inp_sequence = tf.squeeze(inp_sequence)
    inp_sequence = tf.expand_dims(inp_sequence, 0)

    # start_token  shape:(1,)
    start_token = text_vectorize.encode_start_token(_config.start_word, target_tokenizer, language=_config.target_lang)
    end_token, _ = text_vectorize.encode_sentences([_config.end_word], target_tokenizer,
                                                   language=_config.target_lang, mode=target_mode)
    end_token = tf.squeeze(end_token)

    decoder_input = tf.expand_dims(start_token, 0)  # shape --> (1,1) 即(batch_size,sentence_length)

    beam_search_container.reset(inputs=inp_sequence, dec_input=decoder_input)
    inputs, decoder_input = beam_search_container.get_search_inputs()
    if len(checkpoints_path) == 1:  # 如果只使用一个检查点，则不使用checkpoint_ensembling
        checkpoint_path = checkpoints_path[0]
        checkpoint.load_checkpoint(model, tf.keras.optimizers.Adam(), checkpoint_path=checkpoint_path)
    for i in range(_config.max_target_length):
        if len(checkpoints_path) == 1:  # 如果只使用一个检查点，则不使用checkpoint_ensembling
            enc_padding_mask, combined_mask, dec_padding_mask = _transformer.create_masks(inputs, decoder_input)
            predictions, _ = model(inputs, decoder_input, False, enc_padding_mask, combined_mask, dec_padding_mask)
            predictions = tf.squeeze(predictions[:, -1:, :], axis=1)  # (batch_size, vocab_size)
        else:
            predictions = _checkpoint_ensembling(checkpoints_path, model, inputs, decoder_input)

        beam_search_container.expand(predictions=predictions, end_sign=end_token)
        if beam_search_container.beam_size == 0:
            break
        inputs, decoder_input = beam_search_container.get_search_inputs()
    beam_search_result = beam_search_container.get_result()

    return beam_search_result


def translate(sentence, model, tokenizer_source, tokenizer_target, beam_size=_config.BEAM_SIZE):
    """对句子(经过预处理未经过编码)进行翻译,未进行检查点的判断"""
    beam_search_container = beamsearch.BeamSearch(
        beam_size=beam_size,
        max_length=_config.max_target_length,
        worst_score=0)

    # 采用checkpoint_ensembling,获取需要使用的检查点路径列表
    checkpoints_path = checkpoint.get_checkpoints_path()
    if _config.checkpoint_ensembling == "False":
        checkpoints_path = checkpoints_path[-1:]

    predict_idxes = _predict_index(checkpoints_path, sentence, model, beam_search_container, tokenizer_source, tokenizer_target)

    predicted_sentences = []
    target_mode = hlp.mt.common.text_vectorize.get_tokenizer_mode(_config.target_lang)
    # 从容器中抽取序列，生成最终结果
    for i in range(len(predict_idxes)):
        predict_idx = predict_idxes[i].numpy()
        predict_idx = tf.squeeze(predict_idx)
        predict_sentence = text_vectorize.decode_sentence(predict_idx, tokenizer_target,
                                                          _config.target_lang, target_mode)
        predict_sentence = predict_sentence.replace(_config.start_word, '') \
            .replace(_config.end_word, '').strip()
        predicted_sentences.append(predict_sentence)
    return predicted_sentences
