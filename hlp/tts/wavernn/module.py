import os
import time
import tensorflow as tf
from playsound import playsound
import scipy.io.wavfile as wave
import hlp.tts.utils.load_dataset as _dataset
from hlp.tts.utils.spec import melspectrogram2wav, spec_distance
from hlp.tts.utils.text_preprocess import text_to_phonemes, text_to_sequence_phoneme
import numpy as np

def train(epochs: int, train_data_path: str, max_len: int, vocab_size: int,
          batch_size: int, buffer_size: int, checkpoint_save_freq: int,
          checkpoint: tf.train.CheckpointManager, model: tf.keras.Model,
          optimizer: tf.keras.optimizers.Adam, tokenized_type: str = "phoneme",
          dict_path: str = "", valid_data_split: float = 0.0, valid_data_path: str = "",
          max_train_data_size: int = 0, max_valid_data_size: int = 0):
    """
    训练模块
    :param epochs: 训练周期
    :param train_data_path: 文本数据路径
    :param max_len: 文本序列最大长度
    :param vocab_size: 词汇大小
    :param tokenized_type: 分词类型，默认按音素分词，模式：phoneme(音素)/word(单词)/char(字符)
    :param dict_path: 字典路径，若使用phoneme则不用传
    :param buffer_size: Dataset加载缓存大小
    :param batch_size: Dataset加载批大小
    :param checkpoint: 检查点管理器
    :param model: 模型
    :param optimizer: 优化器
    :param valid_data_split: 用于从训练数据中划分验证数据
    :param valid_data_path: 验证数据文本路径
    :param max_train_data_size: 最大训练数据量
    :param max_valid_data_size: 最大验证数据量
    :param checkpoint_save_freq: 检查点保存频率
    :return:
    """
    train_dataset, valid_dataset, steps_per_epoch, valid_steps_per_epoch = \
        _dataset.load_data(train_data_path=train_data_path, max_len=max_len, vocab_size=vocab_size,
                           batch_size=batch_size, buffer_size=buffer_size, tokenized_type=tokenized_type,
                           dict_path=dict_path, valid_data_split=valid_data_split,
                           valid_data_path=valid_data_path, max_train_data_size=max_train_data_size,
                           max_valid_data_size=max_valid_data_size)

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        start_time = time.time()
        total_loss = 0

        for (batch, (mel, stop_token, sentence)) in enumerate(train_dataset.take(steps_per_epoch)):
            batch_start = time.time()
            batch_loss, mel_outputs = _train_step(model, optimizer, sentence, mel, stop_token)  # 训练一个批次，返回批损失
            total_loss += batch_loss

            print('\r{}/{} [Batch {} Loss {:.4f} {:.1f}s]'.format((batch + 1),
                                                                  steps_per_epoch, batch + 1, batch_loss.numpy(),
                                                                  (time.time() - batch_start)), end='')

        print(' - {:.0f}s/step - loss: {:.4f}'.format((time.time() - start_time) / steps_per_epoch,
                                                      total_loss / steps_per_epoch))

        if (epoch + 1) % checkpoint_save_freq == 0:
            checkpoint.save()
            _valid_step(model=model, dataset=valid_dataset, steps_per_epoch=valid_steps_per_epoch)

    return mel_outputs


def evaluate(model: tf.keras.Model, data_path: str, max_len: int,
             vocab_size: int, max_train_data_size: int, batch_size: int,
             buffer_size: int, tokenized_type: str = "phoneme"):
    """
    评估模块
    :param model: 模型
    :param data_path: 文本数据路径
    :param max_len: 文本序列最大长度
    :param vocab_size: 词汇大小
    :param tokenized_type: 分词类型，默认按音素分词，模式：phoneme(音素)/word(单词)/char(字符)
    :param buffer_size: Dataset加载缓存大小
    :param batch_size: Dataset加载批大小
    :param max_train_data_size: 最大训练数据量
    :return: 无返回值
    """
    dataset, _, steps_per_epoch, _ = \
        _dataset.load_data(train_data_path=data_path, max_len=max_len, vocab_size=vocab_size,
                           batch_size=batch_size, buffer_size=buffer_size, tokenized_type=tokenized_type,
                           max_train_data_size=max_train_data_size)

    j = 0
    score_sum = 0
    for (batch, (mel, stop_token, sentence)) in enumerate(dataset.take(steps_per_epoch)):
        for i in range(sentence.shape[0]):
            new_input_ids = sentence[i]
            new_input_ids = tf.expand_dims(new_input_ids, axis=0)
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model.inference(new_input_ids)
            mel2 = mel[i]
            mel2 = tf.expand_dims(mel2, axis=0)
            mel2 = tf.transpose(mel2, [0, 2, 1])
            score = spec_distance(mel_outputs_postnet, mel2)
            score_sum += score
            j = j + 1
            print('第{}个样本的欧式距离为：{}'.format(j, score))
    print("样本平均欧式距离为：", score_sum / j)


def generate(model: tf.keras.Model, max_db: int, ref_db: int, sr: int, max_len: int, wave_save_dir: str,
             n_fft: int, n_mels: int, pre_emphasis: float, n_iter: int, hop_length: int, cmu_dict_path: str,
             win_length: int, dict_path: str = "", tokenized_type: str = "phoneme"):
    """
    生成语音的方法
    :param model: 模型
    :param max_len: 句子序列最大长度
    :param wave_save_dir: 合成的音频保存目录
    :param n_fft: FFT窗口大小
    :param n_mels: 产生的梅尔带数
    :param hop_length: 帧移
    :param n_iter: 指针
    :param win_length: 每一帧音频都由window()加窗，窗长win_length，然后用零填充以匹配N_FFT
    :param max_db: 峰值分贝值
    :param ref_db: 参考分贝值
    :param sr: 采样率
    :param pre_emphasis: 预加重
    :param dict_path: 字典路径
    :param cmu_dict_path: 音素字典路径
    :param tokenized_type: 分词类型
    :return: 无返回值
    """
    if not os.path.exists(wave_save_dir):
        os.makedirs(wave_save_dir)

    i = 0
    # 抓取文本数据
    while True:
        i = i + 1
        b = str(i)
        print()
        seq = input("请输入您要合成的话，输入ESC结束：")
        if seq == 'ESC':
            break
        sequences_list = []
        sequences_list.append(text_to_phonemes(text=seq, cmu_dict_path=cmu_dict_path))
        if tokenized_type == "phoneme":
            input_ids = text_to_sequence_phoneme(texts=sequences_list, max_len=max_len)
        else:
            with open(dict_path, 'r', encoding="utf-8") as dict_file:
                json_string = dict_file.read().strip().strip("\n")
                tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
                input_ids = tokenizer.texts_to_sequences(sequences_list)
                input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids,
                                                                          max_len=max_len, padding="post")
        input_ids = tf.convert_to_tensor(input_ids)
        # 预测
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model.inference(input_ids)

        # 生成预测声音
        wav = melspectrogram2wav(mel_outputs_postnet[0].numpy(), max_db, ref_db, sr, n_fft,
                                 n_mels, pre_emphasis, n_iter, hop_length, win_length)
        name = wave_save_dir + '\\generated' + b + '.wav'
        wave.write(name, rate=sr, data=wav)
        playsound(name)
        print("已合成，路径：{}".format(name))
    print("合成结束")


def _loss_function(mel_out, mel_out_postnet, mel_gts, tar_token, stop_token):
    """
    损失函数
    :param mel_out: 模型输出的mel
    :param mel_out_postnet: postnet输出
    :param mel_gts: ground-true的mel
    :param tar_token: ground-true的stop_token
    :param stop_token: 输出的stop_token
    :return: 损失总和
    """
    mel_gts = tf.transpose(mel_gts, [0, 2, 1])
    mel_out = tf.transpose(mel_out, [0, 2, 1])
    mel_out_postnet = tf.transpose(mel_out_postnet, [0, 2, 1])
    binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    stop_loss = binary_crossentropy(tar_token, stop_token)
    mel_loss = tf.keras.losses.MeanSquaredError()(mel_out, mel_gts) + tf.keras.losses.MeanSquaredError()(
        mel_out_postnet, mel_gts) + stop_loss
    return mel_loss


def _train_step(model: tf.keras.Model, optimizer: tf.keras.optimizers.Adam,
                input_ids: tf.Tensor, mel_gts: tf.Tensor, stop_token: tf.Tensor):
    """
    训练步
    :param input_ids: sentence序列
    :param mel_gts: ground-true的mel
    :param model: 模型
    :param optimizer 优化器
    :param stop_token: ground-true的stop_token
    :return: batch损失和postnet输出
    """
    loss = 0
    with tf.GradientTape() as tape:
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model(input_ids, mel_gts)
        loss += _loss_function(mel_outputs, mel_outputs_postnet, mel_gts, stop_token, gate_outputs)
    batch_loss = loss
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)  # 计算损失对参数的梯度
    optimizer.apply_gradients(zip(gradients, variables))  # 优化器反向传播更新参数
    return batch_loss, mel_outputs_postnet


def _valid_step(model: tf.keras.Model, dataset: tf.data.Dataset, steps_per_epoch: int):
    """
    验证模块
    :param model: 模型
    :param dataset: 验证集dataset
    :param steps_per_epoch: 总的训练步数
    :return: 无返回值
    """
    print("验证轮次")
    start_time = time.time()
    total_loss = 0

    for (batch, (mel, stop_token, sentence)) in enumerate(dataset.take(steps_per_epoch)):
        batch_start = time.time()

        mel_outputs, mel_outputs_postnet, gate_outputs, _ = model(sentence, mel)
        batch_loss = _loss_function(mel_outputs, mel_outputs_postnet, mel, stop_token, gate_outputs)
        total_loss += batch_loss

        print('\r{}/{} [Batch {} Loss {:.4f} {:.1f}s]'.format((batch + 1),
                                                              steps_per_epoch, batch + 1, batch_loss.numpy(),
                                                              (time.time() - batch_start)), end='')
    print(' - {:.0f}s/step - loss: {:.4f}'.format((time.time() - start_time) / steps_per_epoch,
                                                  total_loss / steps_per_epoch))


def load_checkpoint(model: tf.keras.Model, checkpoint_dir: str, checkpoint_save_size: int):
    """
    恢复检查点
    """
    # 如果检查点存在就恢复，如果不存在就重新创建一个
    checkpoint = tf.train.Checkpoint(wavernn=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=checkpoint_save_size)

    if os.path.exists(checkpoint_dir):
        if ckpt_manager.latest_checkpoint:
            checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # if execute_type == "generate":
        #     print("没有检查点，请先执行train模式")
        #     exit(0)

    return ckpt_manager


def Discretized_Mix_Logistic_Loss(
        labels,
        logits,
        classes=65536,
        log_scale_min=None
):
    '''
    labels: [Batch, Time]
    logits: [Batch, Time, Dim]
    '''
    classes = tf.cast(classes, dtype=logits.dtype)

    if log_scale_min is None:
        log_scale_min = float(np.log(1e-14))
    if logits.get_shape()[-1] % 3 != 0:
        raise ValueError('The dimension of \'y\' must be a multiple of 3.')
    nr_mix = logits.get_shape()[-1] // 3

    logit_probs, means, log_scales = tf.split(logits, num_or_size_splits=3, axis=-1)  # [Batch, Time, Dim // 3]
    log_scales = tf.maximum(log_scales, log_scale_min)  # [Batch, Time]

    labels = tf.cast(tf.tile(
        tf.expand_dims(labels, axis=-1),
        [1, 1, means.get_shape()[-1]]
    ), dtype=logits.dtype)
    cetnered_labels = labels - means
    inv_stdv = tf.exp(-log_scales)

    plus_in = inv_stdv * (cetnered_labels + 1 / (classes - 1))
    cdf_plus = tf.math.sigmoid(plus_in)
    min_in = inv_stdv * (cetnered_labels - 1 / (classes - 1))
    cdf_min = tf.math.sigmoid(min_in)

    log_cdf_plus = plus_in - tf.math.softplus(plus_in)
    log_one_minus_cdf_min = -tf.math.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * cetnered_labels
    log_pdf_mid = mid_in - log_scales - 2.0 * tf.math.softplus(mid_in)

    inner_inner_cond = tf.cast(tf.greater(cdf_delta, 1e-5), dtype=logits.dtype)
    inner_inner_out = \
        inner_inner_cond * tf.math.log(tf.maximum(cdf_delta, 1e-12)) + \
        (1.0 - inner_inner_cond) * (log_pdf_mid - tf.math.log((classes - 1) / 2))
    inner_cond = tf.cast(tf.greater(labels, 0.999), dtype=logits.dtype)
    inner_out = \
        inner_cond * log_one_minus_cdf_min + \
        (1.0 - inner_cond) * inner_inner_out
    cond = tf.cast(tf.less(labels, -0.999), dtype=logits.dtype)
    log_probs = cond * log_cdf_plus + (1.0 - cond) * inner_out
    log_probs = log_probs + tf.math.log_softmax(logit_probs, -1)

    return -tf.reduce_mean(tf.math.reduce_logsumexp(log_probs, axis=-1, keepdims=True))
