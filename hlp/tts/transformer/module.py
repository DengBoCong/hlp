import os
import time
import tensorflow as tf
import scipy.io.wavfile as wave
import hlp.tts.utils.load_dataset as _dataset
from hlp.tts.utils.spec import melspectrogram2wav
from hlp.tts.utils.text_preprocess import text_to_phonemes
from hlp.tts.utils.text_preprocess import text_to_sequence_phoneme


def train(encoder: tf.keras.Model, decoder: tf.keras.Model, optimizer: tf.keras.optimizers.Adam,
          epochs: int, checkpoint: tf.train.CheckpointManager, train_data_path: str, max_len: int,
          vocab_size: int, batch_size: int, buffer_size: int, checkpoint_save_freq: int, num_mel: int,
          tokenized_type: str = "phoneme", dict_path: str = "", valid_data_split: float = 0.0,
          valid_data_path: str = "", max_train_data_size: int = 0, max_valid_data_size: int = 0):
    """
    训练模块
    :param encoder: 模型的encoder
    :param decoder: 模型的decoder
    :param optimizer: 优化器
    :param checkpoint: 检查点管理器
    :param epochs: 训练周期
    :param train_data_path: 文本数据路径
    :param max_len: 文本序列最大长度
    :param vocab_size: 词汇大小
    :param num_mel: 产生的梅尔带数
    :param buffer_size: Dataset加载缓存大小
    :param batch_size: Dataset加载批大小
    :param tokenized_type: 分词类型，默认按音素分词，模式：phoneme(音素)/word(单词)/char(字符)
    :param dict_path: 字典路径，若使用phoneme则不用传
    :param valid_data_split: 用于从训练数据中划分验证数据
    :param valid_data_path: 验证数据文本路径
    :param max_train_data_size: 最大训练数据量
    :param max_valid_data_size: 最大验证数据量
    :param checkpoint_save_freq: 检查点保存频率
    """
    train_dataset, valid_dataset, steps_per_epoch, valid_steps_per_epoch = \
        _dataset.load_data(train_data_path=train_data_path, max_len=max_len, vocab_size=vocab_size,
                           batch_size=batch_size, buffer_size=buffer_size, tokenized_type=tokenized_type,
                           dict_path=dict_path, valid_data_split=valid_data_split,
                           valid_data_path=valid_data_path, max_train_data_size=max_train_data_size,
                           max_valid_data_size=max_valid_data_size)

    if steps_per_epoch == 0:
        print("训练数据量过小，小于batch_size，请添加数据后重试")
        exit(0)

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        start_time = time.time()
        total_loss = 0

        for (batch, (mel, stop_token, sentence)) in enumerate(train_dataset.take(steps_per_epoch)):
            batch_start = time.time()
            mel = tf.transpose(mel, [0, 2, 1])
            mel_input = tf.concat([tf.zeros(shape=(mel.shape[0], 1, num_mel), dtype=tf.float32),
                                   mel[:, :-1, :]], axis=1)

            batch_loss, mel_outputs = _train_step(encoder, decoder, optimizer, sentence,
                                                  mel, mel_input, stop_token)
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

            _valid_step(encoder=encoder, decoder=decoder, dataset=valid_dataset,
                        num_mel=num_mel, steps_per_epoch=valid_steps_per_epoch)

    return mel_outputs


def generate(encoder: tf.keras.Model, decoder: tf.keras.Model, wave_save_dir: str,
             cmu_dict_path: str, max_mel_length: int,
             max_db: int, ref_db: int, sr: int, max_len: int,
             n_fft: int, num_mel: int, pre_emphasis: float, n_iter: int, hop_length: int,
             win_length: int, dict_path: str = "", tokenized_type: str = "phoneme"):
    """
    生成语音的方法
    :param encoder: 模型的encoder
    :param decoder: 模型的decoder
    :param max_len: 句子序列最大长度
    :param max_mel_length: 最长mel序列长度
    :param wave_save_dir: 合成的音频保存目录
    :param n_fft: FFT窗口大小
    :param num_mel: 产生的梅尔带数
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
    print("Agent: 你好！结束合成请输入ESC。")
    while True:
        sentence = input("Sentence: ")
        if sentence == "ESC":
            print("Agent: 再见！")
            exit(0)

        sentence = text_to_phonemes(text=sentence, cmu_dict_path=cmu_dict_path)
        if tokenized_type == "phoneme":
            input_ids = text_to_sequence_phoneme(texts=[sentence], max_len=max_len)
        else:
            with open(dict_path, 'r', encoding="utf-8") as dict_file:
                json_string = dict_file.read().strip().strip("\n")
                tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
                input_ids = tokenizer.texts_to_sequences([sentence])
                input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids,
                                                                          max_len=max_len, padding="post")
        input_ids = tf.convert_to_tensor(input_ids)
        mel_input = tf.zeros(shape=(1, 1, num_mel))
        enc_outputs, padding_mask = encoder(input_ids)
        for i in range(max_mel_length):
            mel_pred, post_net_pred, stop_token_pred = decoder(inputs=[enc_outputs, mel_input, padding_mask])
            # 由于数据量少的问题，暂时不开启stop_token
            # if stop_token_pred[0][0][0] > tf.constant(0.5):
            #     break
            mel_input = tf.concat([mel_input, post_net_pred[:, -1:, :]], axis=1)

        post_net_pred = tf.transpose(post_net_pred, [0, 2, 1])
        wav = melspectrogram2wav(post_net_pred[0].numpy(), max_db, ref_db, sr, n_fft,
                                 num_mel, pre_emphasis, n_iter, hop_length, win_length)
        name = wave_save_dir + '\\' + str(time.time()) + '.wav'
        wave.write(name, rate=sr, data=wav)
        print("已合成，路径：{}".format(name))

    print("合成结束")


def _train_step(encoder: tf.keras.Model, decoder: tf.keras.Model, optimizer, sentence,
                mel_target, mel_input, stop_token_target):
    """
    训练步
    :param encoder: 模型的encoder
    :param decoder: 模型的decoder
    :param sentence: sentence序列
    :param mel_target: ground-true的mel
    :param mel_input: ground-true的用作训练的mel，移动的一维
    :param optimizer 优化器
    :param stop_token_target: ground-true的stop_token
    :return: batch损失和post_net输出
    """
    with tf.GradientTape() as tape:
        enc_outputs, padding_mask = encoder(sentence)
        mel_pred, post_net_pred, stop_token_pred = decoder(inputs=[enc_outputs, mel_input, padding_mask])
        stop_token_pred = tf.squeeze(stop_token_pred, axis=-1)
        loss = _loss_function(mel_pred, post_net_pred, mel_target, stop_token_target, stop_token_pred)
    batch_loss = loss

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss, post_net_pred


def _valid_step(encoder: tf.keras.Model, decoder: tf.keras.Model, num_mel: int,
                dataset: tf.data.Dataset, steps_per_epoch: int):
    """
    验证模块
    :param encoder: 模型的encoder
    :param decoder: 模型的decoder
    :param num_mel: 产生的梅尔带数
    :param dataset: 验证集dataset
    :param steps_per_epoch: 总的训练步数
    :return: 无返回值
    """
    print("验证轮次")
    start_time = time.time()
    total_loss = 0

    for (batch, (mel, stop_token, sentence)) in enumerate(dataset.take(steps_per_epoch)):
        batch_start = time.time()
        mel = tf.transpose(mel, [0, 2, 1])
        mel_input = tf.concat([tf.zeros(shape=(mel.shape[0], 1, num_mel), dtype=tf.float32),
                               mel[:, :-1, :]], axis=1)

        enc_outputs, padding_mask = encoder(sentence)
        mel_pred, post_net_pred, stop_token_pred = decoder(inputs=[enc_outputs, mel_input, padding_mask])
        stop_token_pred = tf.squeeze(stop_token_pred, axis=-1)
        batch_loss = _loss_function(mel_pred, post_net_pred, mel, stop_token, stop_token_pred)
        total_loss += batch_loss

        print('\r{}/{} [Batch {} Loss {:.4f} {:.1f}s]'.format((batch + 1),
                                                              steps_per_epoch, batch + 1, batch_loss.numpy(),
                                                              (time.time() - batch_start)), end='')
    print(' - {:.0f}s/step - loss: {:.4f}'.format((time.time() - start_time) / steps_per_epoch,
                                                  total_loss / steps_per_epoch))


def _loss_function(mel_pred, post_net_pred, mel_target, stop_token_target, stop_token_pred):
    """
    损失函数
    :param mel_pred: 模型输出的mel
    :param post_net_pred: post_net输出
    :param mel_target: ground-true的mel
    :param stop_token_target: ground-true的stop_token
    :param stop_token_pred: 输出的stop_token
    :return: 损失总和
    """
    stop_loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=True)(stop_token_target, stop_token_pred)
    mel_loss = tf.keras.losses.MeanSquaredError()(mel_pred, mel_target)\
               + tf.keras.losses.MeanSquaredError()(post_net_pred, mel_target) + stop_loss

    return mel_loss


def load_checkpoint(encoder: tf.keras.Model, decoder: tf.keras.Model,
                    checkpoint_dir: str, execute_type: str, checkpoint_save_size: int):
    """
    恢复检查点
    """
    # 如果检查点存在就恢复，如果不存在就重新创建一个
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=checkpoint_save_size)

    if os.path.exists(checkpoint_dir):
        if ckpt_manager.latest_checkpoint:
            checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
        if execute_type == "generate":
            print("没有检查点，请先执行train模式")
            exit(0)

    return ckpt_manager
