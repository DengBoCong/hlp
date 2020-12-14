import os

import tensorflow as tf
import numpy

from hlp.mt.config import get_config as _config
from hlp.mt.model import nmt_model, checkpoint
from hlp.mt.model import transformer as _transformer
from hlp.mt.common import text_vectorize


def _model_build(model, inp, tar):
    tar_inp = tar[:, :-1]
    enc_padding_mask, combined_mask, dec_padding_mask = _transformer.create_masks(inp, tar_inp)
    predictions, _ = model(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)


def _get_sample_dataset():
    """从保存的文件中读取样例数据进行模型build"""
    input_path = _config.encoded_sequences_path_prefix + _config.source_lang
    target_path = _config.encoded_sequences_path_prefix + _config.target_lang
    input_tensor = tf.cast(numpy.loadtxt(input_path, dtype='int32', max_rows=_config.BATCH_SIZE), tf.int32)
    target_tensor = tf.cast(numpy.loadtxt(target_path, dtype='int32', max_rows=_config.BATCH_SIZE), tf.int32)
    return input_tensor, target_tensor


def average_checkpoints_test():
    """
    对检查点本身进行平均的示例
    需要先进行训练保存几个检查点
    """
    # 模型相关配置
    transformer, optimizer, _, _ = nmt_model.load_model()
    trackables = {'transformer': transformer, 'optimizer': optimizer}
    model_key = 'transformer'

    # 模型build加载一个batch数据
    input_tensor, target_tensor = _get_sample_dataset()
    _model_build(transformer, input_tensor, target_tensor)

    # 检查点路径及输出平均检查点路径
    model_dir = _config.checkpoint_path
    output_dir = model_dir + '_avg_ckpts'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    path = checkpoint.average_checkpoints(model_dir, output_dir, trackables, max_count=8, model_key=model_key)
    print(path)


if __name__ == '__main__':
    average_checkpoints_test()