import os
import time
import tensorflow as tf
from hlp.stt.utils.load_dataset import load_data
from hlp.stt.utils.audio_process import wav_to_feature
from hlp.stt.utils.utils import compute_ctc_input_length
from hlp.stt.utils.utils import can_stop
from hlp.stt.utils.utils import wers
from hlp.stt.utils.utils import lers
from hlp.stt.utils.utils import record
from hlp.utils.utils import load_tokenizer
from hlp.stt.utils.utils import plot_history


def train(model: tf.keras.Model, optimizer: tf.keras.optimizers.Adam, epochs: int,
          checkpoint: tf.train.CheckpointManager, train_data_path: str, batch_size: int,
          buffer_size: int, checkpoint_save_freq: int, dict_path: str = "",
          valid_data_split: float = 0.0, valid_data_path: str = "", train_length_path: str = "",
          valid_length_path: str = "", stop_early_limits: int = 0, max_train_data_size: int = 0,
          max_valid_data_size: int = 0, history_img_path: str = ""):
    """
    训练模块
    :param model: 模型
    :param optimizer: 优化器
    :param checkpoint: 检查点管理器
    :param epochs: 训练周期
    :param train_data_path: 文本数据路径
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
    train_dataset, valid_dataset, steps_per_epoch, valid_steps_per_epoch = \
        load_data(train_data_path=train_data_path, batch_size=batch_size, buffer_size=buffer_size,
                  valid_data_split=valid_data_split, valid_data_path=valid_data_path,
                  train_length_path=train_length_path, valid_length_path=valid_length_path,
                  max_train_data_size=max_train_data_size, max_valid_data_size=max_valid_data_size)

    tokenizer = load_tokenizer(dict_path=dict_path)

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
    plot_history(history=history, valid_epoch_freq=checkpoint_save_freq, history_img_path=history_img_path)
    return history


def evaluate(model: tf.keras.Model, data_path: str, batch_size: int, buffer_size: int,
             dict_path: str = "", length_path: str = "", max_data_size: int = 0):
    """
    评估模块
    :param model: 模型
    :param data_path: 文本数据路径
    :param buffer_size: Dataset加载缓存大小
    :param batch_size: Dataset加载批大小
    :param dict_path: 字典路径，若使用phoneme则不用传
    :param max_data_size: 最大训练数据量
    :param length_path: 训练样本长度保存路径
    :return: 返回历史指标数据
    """
    valid_dataset, _, valid_steps_per_epoch, _ = \
        load_data(train_data_path=data_path, batch_size=batch_size, buffer_size=buffer_size,
                  valid_data_split=0.0, valid_data_path="", train_length_path=length_path,
                  valid_length_path="", max_train_data_size=max_data_size, max_valid_data_size=0)

    tokenizer = load_tokenizer(dict_path=dict_path)

    _, _, _ = _valid_step(model=model, dataset=valid_dataset,
                          steps_per_epoch=valid_steps_per_epoch, tokenizer=tokenizer)


def recognize(model: tf.keras.Model, audio_feature_type: str, start_sign: str,
              unk_sign: str, end_sign: str, record_path: str, max_length: int, dict_path: str):
    """
    语音识别模块
    :param model: 模型
    :param audio_feature_type: 特征类型
    :param start_sign: 开始标记
    :param end_sign: 结束标记
    :param unk_sign: 未登录词
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
            audio_feature = audio_feature[:max_length, :]
            input_tensor = tf.keras.preprocessing.sequence.pad_sequences([audio_feature], padding='post',
                                                                         maxlen=max_length, dtype='float32')
            predictions = model(input_tensor)
            ctc_input_length = compute_ctc_input_length(input_tensor.shape[1], predictions.shape[1],
                                                        tf.convert_to_tensor([[len(audio_feature)]]))

            output = tf.keras.backend.ctc_decode(y_pred=predictions, input_length=tf.reshape(
                ctc_input_length, [ctc_input_length.shape[0]]), greedy=True)

            tokenizer = load_tokenizer(dict_path=dict_path)

            sentence = tokenizer.sequences_to_texts(output[0][0].numpy())
            sentence = sentence[0].replace(start_sign, '').replace(end_sign, '').replace(' ', '')
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


def _valid_step(model: tf.keras.Model, dataset: tf.data.Dataset, steps_per_epoch: int,
                tokenizer: tf.keras.preprocessing.text.Tokenizer):
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
