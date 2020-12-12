import os

import tensorflow as tf
from tensorflow.python.training.tracking import graph_view

from hlp.mt.config import get_config as _config
from hlp.mt.model import nmt_model
from hlp.mt import trainer


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
        print('翻译模型使用检查点:'+checkpoint_path)


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


def checkpoint_ensembling(model, model_dir):
    """
    @param model:使用的模型对象
    @param model_dir:包含用来生成检查点的目录
    """

    pass


def average_checkpoints(model_dir,
                        output_dir,
                        trackables,
                        max_count=8,
                        model_key="model"):
    """Averages object-based checkpoints.

    Args:
    model_dir: The directory containing checkpoints.包含检查点的目录
    output_dir: The directory that will contain the averaged checkpoint.平均后检查点的保存路径
    trackables: A dictionary containing the trackable objects included in the
      checkpoint.可追踪的对象的字典
    max_count: The maximum number of checkpoints to average.用来生成平均检查点的最大检查点数量
    model_key: The key in :obj:`trackables` that references the model.模型的key？

    Returns:
    The path to the directory containing the averaged checkpoint. 包含平均检查点的路径

    Raises:
    ValueError: if :obj:`output_dir` is the same as :obj:`model_dir`.
    ValueError: if a model is not found in :obj:`trackables` or is not already
      built.
    ValueError: if no checkpoints are found in :obj:`model_dir`.

    See Also:
    :func:`opennmt.utils.average_checkpoints_into_layer`
    """
    if model_dir == output_dir:
        raise ValueError("Model and output directory must be different")
    model = trackables.get(model_key)
    if model is None:
        raise ValueError("%s not found in trackables %s" % (model_key, trackables))

    # 取检查点路径列表
    checkpoint_state = tf.train.get_checkpoint_state(model_dir)
    if checkpoint_state is None:
        raise ValueError("No checkpoints found in %s" % model_dir)
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
    """Updates the layer weights with their average value in the checkpoints.

    Args:
      checkpoints: A non empty list of checkpoint paths.
      layer: A ``tf.keras.layers.Layer`` instance.
      layer_prefix: The name/scope that prefixes the layer variables names in the
        checkpoints.

    Raises:
      ValueError: if :obj:`checkpoints` is empty.
      ValueError: if :obj:`layer` is not already built.

    See Also:
      :func:`opennmt.utils.average_checkpoints`
    """
    if not checkpoints:
        raise ValueError("There should be at least one checkpoint")
    if not layer.built:
        raise ValueError("The layer should be built before calling this function")

    # Reset the layer variables to 0.
    for variable in layer.variables:
        variable.assign(tf.zeros_like(variable))

    # Get a map from variable names in the checkpoint to variables in the layer.
    _, names_to_variables = _get_variables_name_mapping(layer, root_key=layer_prefix)

    num_checkpoints = len(checkpoints)
    tf.get_logger().info("Averaging %d checkpoints...", num_checkpoints)
    for checkpoint_path in checkpoints:
        tf.get_logger().info("Reading checkpoint %s...", checkpoint_path)
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
    """Returns mapping between variables and their name in the object-based
    representation.

    Args:
    root: The root layer.
    root_key: Key that was used to save :obj:`root`, if any.

    Returns:
    A dict mapping variables ref to names and a dict mapping variables name to
    variables.
    """
    # TODO: find a way to implement this function using public APIs.
    named_variables, _, _ = graph_view.ObjectGraphView(root).serialize_object_graph()
    variables_to_names = {}
    names_to_variables = {}
    for saveable_object in named_variables:
        variable = saveable_object.op
        if not hasattr(variable, "ref"):  # Ignore non Tensor-like objects.
            continue
        name = saveable_object.name
        if root_key is not None:
            name = "%s/%s" % (root_key, name)
        variables_to_names[variable.ref()] = name
        names_to_variables[name] = variable
    return variables_to_names, names_to_variables


def main():
    transformer = nmt_model.get_model(2894, 1787)
    learning_rate = trainer.CustomSchedule(_config.d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    trackables = {'transformer': transformer, 'optimizer': optimizer}
    model_key = 'transformer'
    model_dir = _config.checkpoint_path
    output_dir = _config.checkpoint_path + '_avg_ckpts'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    trainer.train(transformer,
                  validation_data=_config.validate_from_txt,
                  validation_split=1 - _config.train_size,
                  validation_freq=_config.validation_freq)
    path = average_checkpoints(model_dir, output_dir, trackables, max_count=8, model_key=model_key)
    print(path)


if __name__ == '__main__':
    main()
