
import tensorflow as tf
from math import ceil
from hlp.stt.ds2.model import DS2
from hlp.stt.ds2.util import get_config, get_dataset_information, compute_metric
from hlp.stt.utils.load_dataset import load_data
from hlp.stt.utils.generator import test_generator


if __name__ == "__main__":
    configs = get_config()
    dataset_information = get_dataset_information(configs["preprocess"]["dataset_information_path"])

    # 获取模型配置，加载模型
    conv_layers = configs["model"]["conv_layers"]
    filters = configs["model"]["conv_filters"]
    kernel_size = configs["model"]["conv_kernel_size"]
    strides = configs["model"]["conv_strides"]
    bi_gru_layers = configs["model"]["bi_gru_layers"]
    gru_units = configs["model"]["gru_units"]
    fc_units = configs["model"]["fc_units"]
    dense_units = dataset_information["vocab_size"] + 2

    model = DS2(conv_layers, filters, kernel_size, strides, bi_gru_layers, gru_units, fc_units, dense_units)
    optimizer = tf.keras.optimizers.Adam()

    # 加载模型检查点
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=configs["checkpoint"]['directory'],
        max_to_keep=configs["checkpoint"]['max_to_keep']
    )
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)

    dataset_name = configs["preprocess"]["dataset_name"]
    test_data_path = configs["test"]["data_path"]
    num_examples = configs["test"]["num_examples"]

    # 加载测试集数据(audio_data_path_list, text_list)
    test_data = load_data(dataset_name, test_data_path, num_examples)

    batch_size = configs["test"]["batch_size"]
    batchs = ceil(len(test_data[0]) / batch_size)
    audio_feature_type = configs["other"]["audio_feature_type"]
    max_input_length = dataset_information["max_input_length"]

    # 构建测试数据生成器
    test_data_generator = test_generator(
        test_data,
        batchs,
        batch_size,
        audio_feature_type,
        max_input_length
    )

    # 获取index_word
    index_word = dataset_information["index_word"]
    text_process_mode = configs["preprocess"]["text_process_mode"]

    # 计算指标并打印
    wers, lers, norm_lers = compute_metric(model, test_data_generator, batchs, text_process_mode, index_word)
    print("WER:")
    print("平均WER:", wers)
    print("LER:")
    print("平均LER:", lers)
    print("规范化平均LER:", norm_lers)
