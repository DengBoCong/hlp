import time
import tensorflow as tf
from hlp.stt.utils.load_dataset import load_data
from hlp.utils.optimizers import loss_func_mask
from hlp.stt.utils.utils import wers
from hlp.stt.utils.utils import lers
from hlp.stt.utils.utils import load_tokenizer
from hlp.stt.utils.utils import plot_history
from hlp.utils.beamsearch import BeamSearchDecoder


def train(epochs: int, train_data_path: str, batch_size: int, buffer_size: int, checkpoint_save_freq: int,
          checkpoint: tf.train.CheckpointManager, model: tf.keras.Model, optimizer: tf.keras.optimizers.Adam,
          dict_path: str = "", valid_data_split: float = 0.0, valid_data_path: str = "",
          train_length_path: str = "", valid_length_path: str = "", max_train_data_size: int = 0,
          max_valid_data_size: int = 0, history_img_path: str = ""):
    """
    训练模块
    :param epochs: 训练周期
    :param train_data_path: 文本数据路径
    :param dict_path: 字典路径，若使用phoneme则不用传
    :param buffer_size: Dataset加载缓存大小
    :param batch_size: Dataset加载批大小
    :param checkpoint: 检查点管理器
    :param model: 模型
    :param optimizer: 优化器
    :param valid_data_split: 用于从训练数据中划分验证数据
    :param valid_data_path: 验证数据文本路径
    :param max_train_data_size: 最大训练数据量
    :param train_length_path: 训练样本长度保存路径
    :param valid_length_path: 验证样本长度保存路径
    :param max_valid_data_size: 最大验证数据量
    :param checkpoint_save_freq: 检查点保存频率
    :param history_img_path: 历史指标数据图表保存路径
    :return:
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
        total_loss = 0
        start_time = time.time()
        enc_hidden = model.initialize_hidden_state()
        dec_input = tf.expand_dims([tokenizer.word_index.get('<start>')] * batch_size, 1)

        print("Epoch {}/{}".format(epoch + 1, epochs))
        for (batch, (audio_feature, sentence, length)) in enumerate(train_dataset.take(steps_per_epoch)):
            batch_start = time.time()

            batch_loss = _train_step(model, optimizer, audio_feature, sentence, enc_hidden, dec_input)
            total_loss += batch_loss

            print('\r{}/{} [Batch {} Loss {:.4f} {:.1f}s]'.format(
                (batch + 1), steps_per_epoch, batch + 1, batch_loss.numpy(), (time.time() - batch_start)), end="")

        print(' - {:.0f}s/step - loss: {:.4f}'.format((time.time() - start_time) / steps_per_epoch,
                                                      total_loss / steps_per_epoch))

        if (epoch + 1) % checkpoint_save_freq == 0:
            checkpoint.save()

            if valid_steps_per_epoch == 0:
                print("验证数据量过小，小于batch_size，请添加数据后重试")
                exit(0)

            valid_loss, valid_wer, valid_ler = _valid_step(model=model, dataset=valid_dataset,
                                                           steps_per_epoch=valid_steps_per_epoch, tokenizer=tokenizer)
            history["wers"].append(valid_wer)
            history["norm_lers"].append(valid_ler)
            # norm_rates_lers, norm_aver_lers = compute_metric(model, val_data_generator,
            #                                                  val_batchs, val_batch_size)
            # print("平均字母错误率: ", norm_aver_lers)
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
    enc_hidden = model.initialize_hidden_state()
    dec_input = tf.expand_dims([tokenizer.word_index.get('<start>')] * batch_size, 1)

    _, _, _ = _valid_step(model=model, dataset=valid_dataset, steps_per_epoch=valid_steps_per_epoch,
                          tokenizer=tokenizer, enc_hidden=enc_hidden, dec_input=dec_input)


def _train_step(model: tf.keras.Model, optimizer: tf.keras.optimizers.Adam, audio_feature: tf.Tensor,
                sentence: tf.Tensor, enc_hidden: tf.Tensor, dec_input: tf.Tensor):
    """
    训练步
    :param model: 模型
    :param sentence: sentence序列
    :param audio_feature: 音频特征序列
    :param enc_hidden: encoder初始化隐藏层
    :param optimizer 优化器
    :param dec_input: 解码器输入
    :return: batch损失和post_net输出
    """
    loss = 0
    with tf.GradientTape() as tape:
        for t in range(1, sentence.shape[1]):
            predictions, _ = model(audio_feature, enc_hidden, dec_input)
            loss += loss_func_mask(sentence[:, t], predictions)

            if sum(sentence[:, t]) == 0:
                break

            dec_input = tf.expand_dims(sentence[:, t], 1)
    batch_loss = (loss / int(sentence.shape[0]))
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss


def _valid_step(model: tf.keras.Model, dataset: tf.data.Dataset, steps_per_epoch: int,
                tokenizer: tf.keras.preprocessing.text.Tokenizer, enc_hidden: tf.Tensor, dec_input: tf.Tensor):
    """
    验证模块
    :param model: 模型
    :param dataset: 验证数据dataset
    :param steps_per_epoch: 验证训练步
    :param tokenizer: 分词器
    :param enc_hidden: encoder初始化隐藏层
    :param dec_input: 解码器输入
    :return: 损失、wer、ler
    """
    print("验证轮次")
    start_time = time.time()
    total_loss = 0
    aver_wers = 0
    aver_norm_lers = 0

    for (batch, (audio_feature, sentence, length)) in enumerate(dataset.take(steps_per_epoch)):
        loss = 0
        batch_start = time.time()
        result = []

        for t in range(1, sentence.shape[1]):
            dec_input = dec_input[:, -1:]
            predictions, _ = model(audio_feature, enc_hidden, dec_input)
            predictions = tf.argmax(predictions, axis=-1)

            result.append(predictions)
            loss += loss_func_mask(sentence[:, t], predictions)

        batch_loss = (loss / int(sentence.shape[0]))
        print(result)
        exit(0)

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
