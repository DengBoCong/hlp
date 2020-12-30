import os

from tensorflow.python.training.tracking import graph_view
import tensorflow as tf
import numpy

from hlp.mt.config import get_config as _config
from hlp.mt.model import nmt_model, checkpoint
from hlp.mt.model import transformer as _transformer


def load_checkpoint(transformer, optimizer, checkpoint_path=_config.checkpoint_path):
    """
    获取检查点
    @param transformer: 模型实例
    @param optimizer: 优化器
    @param checkpoint_path:检查点的路径
    """
    # 加载检查点
    checkpoint_dir = os.path.dirname(checkpoint_path)
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=_config.max_checkpoints_num)
    if ckpt_manager.latest_checkpoint:
        # ckpt.restore(ckpt_manager.latest_checkpoint)
        ckpt.restore(checkpoint_path)
        # print('已恢复至最新的检查点！')
        print('正在使用检查点:'+checkpoint_path)


def get_checkpoints_path(model_dir=_config.checkpoint_path):
    """
    获取检查点路径列表
    @param model_dir:
    @return:
    """
    checkpoint_state = tf.train.get_checkpoint_state(model_dir)
    if checkpoint_state is None:
        raise ValueError("未在目录：%s 中发现检查点！" % model_dir)
    return checkpoint_state.all_model_checkpoint_paths


def average_checkpoints(model_dir,
                        output_dir,
                        trackables,
                        max_count=8,
                        model_key="model"):
    """

    @param model_dir: 需要平均的检查点的文件夹路径
    @param output_dir: 将得到的检查点输出的文件夹路径
    @param trackables: 检查点所保存的对象的字典
    @param max_count: 最多使用几个检查点进行平均
    @param model_key: 字典中模型对应的key
    @return:
    """
    if model_dir == output_dir:
        raise ValueError("输入与输出需是不同文件夹")
    model = trackables.get(model_key)
    if model is None:
        raise ValueError("模型的key:%s 并没有在字典 %s 中找到" % (model_key, trackables))

    # 取检查点路径列表
    checkpoint_state = tf.train.get_checkpoint_state(model_dir)
    if checkpoint_state is None:
        raise ValueError(" %s 文件夹中没有检查点" % model_dir)
    checkpoints_path = checkpoint_state.all_model_checkpoint_paths
    if len(checkpoints_path) > max_count:
        checkpoints_path = checkpoints_path[-max_count:]

    _average_checkpoints_into_layer(checkpoints_path, model, model_key)

    last_step = _get_step_from_checkpoint_prefix(checkpoints_path[-1])
    checkpoint = tf.train.Checkpoint(**trackables)
    new_checkpoint_manager = tf.train.CheckpointManager(checkpoint, output_dir, max_to_keep=None)
    new_checkpoint_manager.save(checkpoint_number=last_step)
    return output_dir


def _average_checkpoints_into_layer(checkpoints, layer, layer_prefix):
    """将检查点平均并将平均值放到模型中
    @param checkpoints: 检查点路径的列表
    @param layer: 模型实例
    @param layer_prefix:模型的key
    """
    if not checkpoints:
        raise ValueError("至少应有一个检查点")
    if not layer.built:
        raise ValueError("使用此方法前应对模型进行build")

    # 将模型的变量都重置为0
    for variable in layer.variables:
        variable.assign(tf.zeros_like(variable))

    # 得到一个检查点中变量名到层中变量的字典
    _, names_to_variables = _get_variables_name_mapping(layer, root_key=layer_prefix)

    num_checkpoints = len(checkpoints)
    tf.get_logger().info("正在平均 %d 个检查点...", num_checkpoints)
    for checkpoint_path in checkpoints:
        tf.get_logger().info("正在读取检查点 %s...", checkpoint_path)
        reader = tf.train.load_checkpoint(checkpoint_path)
        for path in reader.get_variable_to_shape_map().keys():
            if not path.startswith(layer_prefix) or ".OPTIMIZER_SLOT" in path:
                continue
            variable = names_to_variables[path]
            value = reader.get_tensor(path)
            variable.assign_add(value / num_checkpoints)


def _get_step_from_checkpoint_prefix(prefix):
    """Extracts the training step from the checkpoint file prefix."""
    return int(prefix.split("-")[-1])


def _get_variables_name_mapping(root, root_key=None):
    """ 返回一个检查点中变量名到层中变量的字典
    @param root: 模型（层）实例
    @param root_key: 模型（层）的key，即在检查点中的key
    @return: 返回一个检查点中变量名到层中变量的字典
    """
    named_variables, _, _ = graph_view.ObjectGraphView(root).serialize_object_graph()
    variables_to_names = {}
    names_to_variables = {}
    for saveable_object in named_variables:
        variable = saveable_object.op
        # 判断是否是张量，暂时去掉
        # if not hasattr(variable, "ref"):
        #     continue
        name = saveable_object.name
        if root_key is not None:
            name = "%s/%s" % (root_key, name)
        variables_to_names[variable.experimental_ref()] = name
        names_to_variables[name] = variable
    return variables_to_names, names_to_variables


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