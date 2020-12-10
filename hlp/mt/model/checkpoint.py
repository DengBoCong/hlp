import tensorflow as tf
from config import get_config as _config
import numpy as np
from tensorflow.python.training.tracking import graph_view
from model import nmt_model
from model import trainer
import os


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
def average_checkpoints_v1(checkpoints_dir
                           , output_dir=_config.checkpoint_path + '_avg_ckpts'
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


def average_checkpoints(model_dir,
                        output_dir,
                        trackables,
                        max_count=8,
                        model_key="model"):
    """Averages object-based checkpoints.

    Args:
    model_dir: The directory containing checkpoints.
    output_dir: The directory that will contain the averaged checkpoint.
    trackables: A dictionary containing the trackable objects included in the
      checkpoint.
    max_count: The maximum number of checkpoints to average.
    model_key: The key in :obj:`trackables` that references the model.

    Returns:
    The path to the directory containing the averaged checkpoint.

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

    checkpoint_state = tf.train.get_checkpoint_state(model_dir)
    if checkpoint_state is None:
        raise ValueError("No checkpoints found in %s" % model_dir)
    checkpoints_path = checkpoint_state.all_model_checkpoint_paths
    if len(checkpoints_path) > max_count:
        checkpoints_path = checkpoints_path[-max_count:]

    average_checkpoints_into_layer(checkpoints_path, model, model_key)

    last_step = get_step_from_checkpoint_prefix(checkpoints_path[-1])
    checkpoint = tf.train.Checkpoint(**trackables)
    new_checkpoint_manager = tf.train.CheckpointManager(checkpoint, output_dir, max_to_keep=None)
    new_checkpoint_manager.save(checkpoint_number=last_step)
    return output_dir


def average_checkpoints_into_layer(checkpoints, layer, layer_prefix):
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
    _, names_to_variables = get_variables_name_mapping(layer, root_key=layer_prefix)

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


def get_step_from_checkpoint_prefix(prefix):
    """Extracts the training step from the checkpoint file prefix."""
    return int(prefix.split("-")[-1])


def get_variables_name_mapping(root, root_key=None):
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
    trainer.train(transformer
                  , validation_data=_config.validate_from_txt
                  , validation_split=1 - _config.train_size
                  , validation_freq=_config.validation_freq)
    path = average_checkpoints(model_dir, output_dir, trackables, max_count=8, model_key=model_key)
    print(path)


if __name__ == '__main__':
    main()
