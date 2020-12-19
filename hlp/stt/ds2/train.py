import os
import time
from math import ceil

import tensorflow as tf
from matplotlib import pyplot as plt

from hlp.stt.ds2.model import DS2
from hlp.stt.ds2.util import get_config, get_dataset_info, compute_ctc_input_length, compute_metric, can_stop
from hlp.stt.utils.generator import train_generator, test_generator
from hlp.stt.utils.load_dataset import load_data
from hlp.stt.utils.text_process import split_and_encode


def _train_step(model, optimizer, input_tensor, target_tensor, input_length, target_length):
    # 单次训练
    with tf.GradientTape() as tape:
        y_pred = model(input_tensor)
        y_true = target_tensor
        input_length = compute_ctc_input_length(input_tensor.shape[1], y_pred.shape[1], input_length)
        loss = tf.keras.backend.ctc_batch_cost(y_true=y_true,
                                               y_pred=y_pred,
                                               input_length=input_length,
                                               label_length=target_length)

        """
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        """
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def train(model, optimizer,
          train_data_generator, train_batches, epochs,
          valid_data_generator, valid_batches, valid_epoch_freq,
          stop_early_limits, text_process_mode, index_word, manager, save_epoch_freq):
    # 构建history
    history = {"loss": [], "wers": [], "norm_lers": []}

    for epoch in range(1, epochs + 1):
        total_loss = 0
        print("Epoch %d/%d" % (epoch, epochs))
        epoch_start = time.time()
        for batch, (input_tensor, target_tensor, input_length, target_length) in zip(range(1, train_batches + 1),
                                                                                     train_data_generator):
            batch_start = time.time()
            batch_loss = _train_step(model, optimizer, input_tensor, target_tensor, input_length, target_length)
            total_loss += batch_loss
            batch_end = time.time()

            print("Batch %d/%d batch_time: %dms - batch_loss: %.4f" % (
                batch, train_batches, (batch_end - batch_start) * 1000, batch_loss))

        epoch_end = time.time()
        print("batches: %d - epoch_time: %ds - %dms/batch - loss: %.4f\n" % (
            train_batches, epoch_end - epoch_start, (epoch_end - epoch_start) * 1000 / train_batches,
            total_loss / train_batches))

        # 将损失写入history
        history["loss"].append(total_loss / train_batches)

        # 验证并将相关指标写入history
        if epoch % valid_epoch_freq == 0 or epoch == epochs:
            wers, norm_lers = compute_metric(model, valid_data_generator, valid_batches,
                                             text_process_mode, index_word)
            history["wers"].append(wers)
            history["norm_lers"].append(norm_lers)
            print("平均WER:", wers)
            print("规范化平均LER:", norm_lers)
            if len(history["wers"]) >= stop_early_limits:
                if can_stop(history["wers"][-stop_early_limits:]) \
                        or can_stop(history["lers"][-stop_early_limits:]) \
                        or can_stop(history["norm_lers"][-stop_early_limits:]):
                    print("指标反弹，停止训练！")
                    break

        #  每save_epoch_freq轮保存检查点
        if epoch % save_epoch_freq == 0:
            manager.save()

    return history


def _plot_history(history, valid_epoch_freq, history_img_path):
    # 绘制loss
    plt.subplot(2, 1, 1)
    epoch1 = [i for i in range(1, 1 + len(history["loss"]))]
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(epoch1, history["loss"], "--*b")
    plt.xticks(epoch1)

    # 绘制metric(wers、lers、norm_lers)
    plt.subplot(2, 1, 2)
    epoch2 = [i * valid_epoch_freq for i in range(1, 1 + len(history["wers"]))]
    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.plot(epoch2, history["wers"], "--*r", label="wers")
    plt.plot(epoch2, history["norm_lers"], "--*y", label="norm_lers")
    plt.xticks(epoch2)

    plt.legend()
    plt.savefig(history_img_path)
    plt.show()


if __name__ == "__main__":
    # 获取训练配置和语料信息
    configs = get_config()
    print("加载数据集信息...")
    dataset_info = get_dataset_info(configs["preprocess"]["dataset_info_path"])

    epochs = configs["train"]["train_epochs"]
    data_path = configs["train"]["data_path"]
    num_examples = configs["train"]["num_examples"]
    dataset_name = configs["preprocess"]["dataset_name"]

    # 加载训练数据
    print("加载训练数据集...")
    train_audio_path_list, train_text_list = load_data(dataset_name, data_path, num_examples)

    valid_data_path = configs["valid"]["data_path"]
    # 是否含有验证valid数据集,若有则加载，若没有，则将train数据按比例切分一部分为valid数据
    if valid_data_path:
        print("加载验证数据集...")
        valid_num_examples = configs["valid"]["num_examples"]
        valid_audio_data_path_list, valid_text_list = load_data(dataset_name,
                                                                valid_data_path,
                                                                valid_num_examples)
    else:
        print("从训练集中划分验证数据...")
        valid_percent = configs["valid"]["valid_percent"]
        pos = ceil(len(train_audio_path_list) * valid_percent / 100)
        valid_audio_data_path_list, valid_text_list = train_audio_path_list[-pos:], train_text_list[-pos:]
        train_audio_path_list, train_text_list = train_audio_path_list[:-pos], train_text_list[:-pos]

    # 构建train_data和valid_data
    text_process_mode = configs["preprocess"]["text_process_mode"]
    word_index = dataset_info["word_index"]
    train_text_int_sequences_list = split_and_encode(train_text_list, text_process_mode, word_index)
    train_data = (train_audio_path_list, train_text_int_sequences_list)
    valid_data = (valid_audio_data_path_list, valid_text_list)

    audio_feature_type = configs["other"]["audio_feature_type"]
    max_input_length = dataset_info["max_input_length"]
    max_label_length = dataset_info["max_label_length"]

    # 构建训练数据生成器
    train_batch_size = configs["train"]["batch_size"]
    train_batches = ceil(len(train_audio_path_list) / train_batch_size)
    train_data_generator = train_generator(train_data,
                                           train_batches,
                                           train_batch_size,
                                           audio_feature_type,
                                           max_input_length,
                                           max_label_length)

    # 构建验证数据生成器
    valid_batch_size = configs["valid"]["batch_size"]
    valid_batches = ceil(len(valid_audio_data_path_list) / valid_batch_size)
    valid_data_generator = test_generator(valid_data,
                                          valid_batches,
                                          valid_batch_size,
                                          audio_feature_type,
                                          max_input_length)

    # 加载模型
    conv_layers = configs["model"]["conv_layers"]
    filters = configs["model"]["conv_filters"]
    kernel_size = configs["model"]["conv_kernel_size"]
    strides = configs["model"]["conv_strides"]
    bi_gru_layers = configs["model"]["bi_gru_layers"]
    gru_units = configs["model"]["gru_units"]
    fc_units = configs["model"]["fc_units"]
    dense_units = dataset_info["vocab_size"] + 2

    model = DS2(conv_layers, filters, kernel_size, strides, bi_gru_layers, gru_units, fc_units, dense_units)
    optimizer = tf.keras.optimizers.Adam()

    # 加载检查点
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    if not os.path.exists(configs["checkpoint"]['directory']):
        os.mkdir(configs["checkpoint"]['directory'])
    manager = tf.train.CheckpointManager(checkpoint,
                                         directory=configs["checkpoint"]['directory'],
                                         max_to_keep=configs["checkpoint"]['max_to_keep'])
    save_epoch_freq = configs["checkpoint"]["save_epoch_freq"]
    if manager.latest_checkpoint:
        print("加载检查点...")
        checkpoint.restore(manager.latest_checkpoint)

    valid_epoch_freq = configs["valid"]["valid_epoch_freq"]
    stop_early_limits = configs["valid"]["stop_early_limits"]
    index_word = dataset_info["index_word"]

    # 训练
    print("开始训练...")
    history = train(model, optimizer,
                    train_data_generator, train_batches, epochs,
                    valid_data_generator, valid_batches, valid_epoch_freq,
                    stop_early_limits, text_process_mode, index_word, manager, save_epoch_freq)

    # 绘制history并保存
    history_img_dir = configs["other"]["history_img_dir"]
    if not os.path.exists(os.path.dirname(history_img_dir)):
        os.makedirs(os.path.dirname(history_img_dir), exist_ok=True)
    history_img_path = history_img_dir + dataset_name + time.strftime("%Y%m%d%H%M%S", time.localtime()) + ".png"
    _plot_history(history, valid_epoch_freq, history_img_path)

    print("训练结束.")
