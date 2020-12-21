import os

import tensorflow as tf
from pathlib import Path

from hlp.mt.config import get_config as _config
from hlp.mt.common import text_vectorize
from hlp.mt.lm import language_model


def sentence_rescore(sentences, model, tokenizer):
    """给句子列表打分

    @param sentences: 需要进行打分的句子列表
    @param model: 打分使用的语言模型实例
    @param tokenizer: 字典
    """
    language = _config.lm_language
    mode = _config.lm_tokenize_type

    scores_list = []
    for i, sentence in enumerate(sentences):
        score = 0
        sequence = text_vectorize.encode_sentences([sentence], tokenizer, language, mode)
        seq_input = sequence[:, :-1]
        seq_real = sequence[:, 1:]
        prediction = model(seq_input)  # (1, seq_len, vocab_size)
        for j in range(prediction.shape[0]):
            score += prediction[seq_real[0][j]]
        scores_list.append(score)

    return scores_list


def load_checkpoint(model, checkpoint_path=None):
    """从检查点加载模型

    @param model: 模型
    @param checkpoint_path:检查点路径,若为None，则默认使用保存的最新的检查点
    """
    if checkpoint_path is None:
        checkpoint_dir = _config.lm_checkpoint_path
        is_exist = Path(checkpoint_dir)
    else:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        is_exist = Path(checkpoint_path)

    ckpt = tf.train.Checkpoint(language_model=model, optimizer=tf.keras.optimizers.Adam())
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=_config.max_checkpoints_num)
    if not is_exist.exists():
        ValueError("路径 %s 不存在" % checkpoint_path)
    elif checkpoint_path is None:
        if language_model.check_point():
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('已恢复至最新检查点！')
    else:
        ckpt.restore(checkpoint_path)
