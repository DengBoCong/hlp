import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from hlp.stt.utils.load_dataset import load_data
from hlp.stt.utils.audio_process import wav_to_feature
from hlp.stt.utils.utils import compute_ctc_input_length
from hlp.stt.utils.utils import can_stop
from hlp.stt.utils.utils import wers
from hlp.stt.utils.utils import lers
from hlp.stt.utils.utils import record


def train(model: tf.keras.Model, optimizer: tf.keras.optimizers.Adam,
          epochs: int, checkpoint: tf.train.CheckpointManager, train_data_path: str, max_len: int,
          vocab_size: int, batch_size: int, buffer_size: int, checkpoint_save_freq: int,
          dict_path: str = "", valid_data_split: float = 0.0, valid_data_path: str = "",
          train_length_path: str = "", valid_length_path: str = "", stop_early_limits: int = 0,
          max_train_data_size: int = 0, max_valid_data_size: int = 0, history_img_path: str = ""):
    """
    训练模块
    :param model: 模型
    :param optimizer: 优化器
    :param checkpoint: 检查点管理器
    :param epochs: 训练周期
    :param train_data_path: 文本数据路径
    :param max_len: 文本序列最大长度
    :param vocab_size: 词汇大小
    :param buffer_size: Dataset加载缓存大小
    :param batch_size: Dataset加载批大小
    :param dict_path: 字典路径，若使用phoneme则不用传
    :param valid_data_split: 用于从训练数据中划分验证数据
    :param valid_data_path: 验证数据文本路径
    :param max_train_data_size: 最大训练数据量
    :param train_length_path: 训练样本长度保存路径
    :param valid_length_path: 验证样本长度保存路径
    :param stop_early_limits: 不增长停止个数
    :param max_valid_data_size: 最大验证数据量
    :param checkpoint_save_freq: 检查点保存频率
    :param history_img_path: 历史指标数据图表保存路径
    :return: 返回历史指标数据
    """
    tokenizer, train_dataset, valid_dataset, steps_per_epoch, valid_steps_per_epoch = \
        load_data(train_data_path=train_data_path, max_len=max_len, vocab_size=vocab_size,
                  batch_size=batch_size, buffer_size=buffer_size, dict_path=dict_path,
                  valid_data_split=valid_data_split, valid_data_path=valid_data_path,
                  train_length_path=train_length_path, valid_length_path=valid_length_path,
                  max_train_data_size=max_train_data_size, max_valid_data_size=max_valid_data_size)
    history = {"loss": [], "wers": [], "norm_lers": []}

    if steps_per_epoch == 0:
        print("训练数据量过小，小于batch_size，请添加数据后重试")
        exit(0)

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        start_time = time.time()
        total_loss = 0

        for (batch, (audio_feature, sentence, length)) in enumerate(train_dataset.take(steps_per_epoch)):
            batch_start = time.time()

            batch_loss = _train_step(model, optimizer, sentence, length, audio_feature)
            total_loss += batch_loss

            print('\r{}/{} [Batch {} Loss {:.4f} {:.1f}s]'.format(
                (batch + 1), steps_per_epoch, batch + 1, batch_loss.numpy(), (time.time() - batch_start)), end="")

        print(' - {:.0f}s/step - loss: {:.4f}'.format((time.time() - start_time) / steps_per_epoch,
                                                      total_loss / steps_per_epoch))
        history["loss"].append(total_loss / steps_per_epoch)

        if (epoch + 1) % checkpoint_save_freq == 0:
            checkpoint.save()

            if valid_steps_per_epoch == 0:
                print("验证数据量过小，小于batch_size，请添加数据后重试")
                exit(0)

            valid_loss, valid_wer, valid_ler = _valid_step(model=model, dataset=valid_dataset,
                                                           steps_per_epoch=valid_steps_per_epoch, tokenizer=tokenizer)
            history["wers"].append(valid_wer)
            history["norm_lers"].append(valid_ler)

            if stop_early_limits != 0 and len(history["wers"]) >= stop_early_limits:
                if can_stop(history["wers"][-stop_early_limits:]) \
                        or can_stop(history["norm_lers"][-stop_early_limits:]):
                    print("指标反弹，停止训练！")
                    break
    _plot_history(history=history, valid_epoch_freq=checkpoint_save_freq, history_img_path=history_img_path)
    return history


def recognize(model: tf.keras.Model, audio_feature_type: str,
              record_path: str, max_length: int, dict_path: str):
    """
    语音识别模块
    :param model: 模型
    :param audio_feature_type: 特征类型
    :param record_path: 录音保存路径
    :param max_length: 最大音频补齐长度
    :param dict_path: 字典保存路径
    :return: 无返回值
    """
    while True:
        try:
            record_duration = int(input("请设定录音时长(秒, 负数结束，0则继续输入音频路径):"))
        except BaseException:
            print("录音时长只能为int数值")
        else:
            if record_duration < 0:
                break
            if not os.path.exists(record_path):
                os.makedirs(record_path)
            # 录音
            if record_duration == 0:
                record_path = input("请输入音频路径：")
            else:
                record_path = record_path + time.strftime("%Y_%m_%d_%H_%M_%S_", time.localtime(time.time())) + ".wav"
                record(record_path, record_duration)

            # 加载录音数据并预测
            audio_feature = wav_to_feature(record_path, audio_feature_type)
            input_tensor = tf.keras.preprocessing.sequence.pad_sequences([audio_feature], padding='post',
                                                                         maxlen=max_length, dtype='float32')
            predictions = model(input_tensor)
            ctc_input_length = compute_ctc_input_length(input_tensor.shape[1], predictions.shape[1],
                                                        tf.convert_to_tensor([[len(audio_feature)]]))

            output = tf.keras.backend.ctc_decode(y_pred=predictions, input_length=tf.reshape(
                ctc_input_length, [ctc_input_length.shape[0]]), greedy=True)

            with open(dict_path, 'r', encoding='utf-8') as dict_file:
                json_string = dict_file.read().strip().strip("\n")
                tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string)

            sentence = tokenizer.sequences_to_texts(output[0][0].numpy())
            sentence = sentence[0].replace("<start>", '').replace("<end>", '').replace(' ', '')
            print("Output:", sentence)


def _train_step(model: tf.keras.Model, optimizer: tf.keras.optimizers.Adam,
                sentence: tf.Tensor, length: tf.Tensor, audio_feature: tf.Tensor):
    """
    训练步
    :param model: 模型
    :param sentence: sentence序列
    :param audio_feature: 音频特征序列
    :param length: 样本长度序列
    :param optimizer 优化器
    :return: batch损失和post_net输出
    """
    with tf.GradientTape() as tape:
        predictions = model(audio_feature)
        input_length = compute_ctc_input_length(audio_feature.shape[1],
                                                predictions.shape[1], length[:, 1:])
        loss = tf.keras.backend.ctc_batch_cost(y_true=sentence, y_pred=predictions,
                                               input_length=input_length, label_length=length[:, 0:1])
    batch_loss = tf.reduce_mean(loss)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return batch_loss


def _valid_step(model: tf.keras.Model, dataset: tf.data.Dataset,
                steps_per_epoch: int, tokenizer: tf.keras.preprocessing.text.Tokenizer):
    """
    验证模块
    :param model: 模型
    :param dataset: 验证数据dataset
    :param steps_per_epoch: 验证训练步
    :param tokenizer: 分词器
    :return: 损失、wer、ler
    """
    print("验证轮次")
    start_time = time.time()
    total_loss = 0
    aver_wers = 0
    aver_norm_lers = 0

    for (batch, (audio_feature, sentence, length)) in enumerate(dataset.take(steps_per_epoch)):
        batch_start = time.time()

        predictions = model(audio_feature)
        input_length = compute_ctc_input_length(audio_feature.shape[1],
                                                predictions.shape[1], length[:, 1:])
        loss = tf.keras.backend.ctc_batch_cost(y_true=sentence, y_pred=predictions,
                                               input_length=input_length, label_length=length[:, 0:1])
        output = tf.keras.backend.ctc_decode(y_pred=predictions, greedy=True,
                                             input_length=tf.reshape(input_length, [input_length.shape[0]]))

        results = tokenizer.sequences_to_texts(output[0][0].numpy())
        sentence = tokenizer.sequences_to_texts(sentence.numpy())

        _, aver_wer = wers(sentence, results)
        _, norm_aver_ler = lers(sentence, results)

        aver_wers += aver_wer
        aver_norm_lers += norm_aver_ler

        loss = tf.reduce_mean(loss)
        total_loss += loss
        print('\r{}/{} [Batch {} Loss {:.4f} {:.1f}s]'.format((batch + 1), steps_per_epoch, batch + 1,
                                                              loss.numpy(), (time.time() - batch_start)), end='')
    print(' - {:.0f}s/step - loss: {:.4f} - average_wer：{:.4f} - '
          'average_norm_ler：{:.4f}'.format((time.time() - start_time) / steps_per_epoch,
                                           total_loss / steps_per_epoch, aver_wers / steps_per_epoch,
                                           aver_norm_lers / steps_per_epoch))

    return total_loss / steps_per_epoch, aver_wers / steps_per_epoch, aver_norm_lers / steps_per_epoch


def _plot_history(history, valid_epoch_freq, history_img_path):
    """
    绘制各种指标数据
    :param history: 历史指标数据
    :param valid_epoch_freq: 验证频率
    :param history_img_path: 历史指标显示图片保存位置
    :return: 无返回值
    """
    plt.subplot(2, 1, 1)
    epoch1 = [i for i in range(1, 1 + len(history["loss"]))]
    epoch2 = [i * valid_epoch_freq for i in range(1, 1 + len(history["wers"]))]

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(epoch1, history["loss"], "--*b")
    plt.xticks(epoch1)

    # 绘制metric(valid_loss、wers、norm_lers)
    plt.subplot(2, 1, 2)
    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.plot(epoch2, history["wers"], "--*r", label="wers")
    plt.plot(epoch2, history["norm_lers"], "--*y", label="norm_lers")
    plt.xticks(epoch2)

    plt.legend()
    if not os.path.exists(history_img_path):
        os.makedirs(history_img_path, exist_ok=True)
    plt.savefig(history_img_path + time.strftime("%Y_%m_%d_%H_%M_%S_", time.localtime(time.time())))
    plt.show()
