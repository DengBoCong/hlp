import tensorflow as tf
from config import get_config as _config
import numpy as np


def load_checkpoint(transformer, optimizer):
    """获取检查点"""
    # 加载检查点
    checkpoint_path = _config.checkpoint_path
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=_config.max_checkpoints_num)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        # ckpt.restore('./checkpoints/en_zh/ckpt-10')
        print('已恢复至最新的检查点！')


# TODO:完成检查点平均方法
def average_checkpoints(checkpoints_dir
                        , output_dir=_config.checkpoint_path+'_avg_ckpts'
                        , max_count=8):
    """

    @param checkpoints_dir: 用来生成平均检查点的检查点路径
    @param output_dir:输出平均检查点的路径
    @param max_count:最大用来生成平均检查点的检查点数量
    """
    # 获取检查点列表
    checkpoint_state = tf.train.get_checkpoint_state(checkpoints_dir)
    if checkpoint_state is None:
        raise ValueError("No checkpoints found in %s" % checkpoints_dir)
    checkpoints_path = checkpoint_state.all_model_checkpoint_paths
    if len(checkpoints_path) > max_count:
        checkpoints_path = checkpoints_path[-max_count:]
    # 生成检查点变量列表
    var_list = tf.contrib.framework.list_variables(checkpoints_path[0])

    # 对checkpoint里的每个参数求平均
    var_values, var_dtypes = {}, {}

    for (name, shape) in var_list:
        var_values[name] = np.zeros(shape)

    for ckpt in checkpoints_path:
        reader = tf.contrib.framework.load_checkpoint(ckpt)
        for name in var_values:
            tensor = reader.get_tensor(name)
            var_dtypes[name] = tensor.dtype
            var_values[name] += tensor

    for name in var_values:
        var_values[name] /= len(checkpoints_path)

    # 将平均后的参数保存在一个新的checkpoint里面
    tf_vars = [tf.get_variable(name, dtype=var_dtypes[name], initializer=var_values[name]) for name in var_values]
