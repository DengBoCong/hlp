import os
import time
import tensorflow as tf
from hlp.stt.utils.load_dataset import load_data


def train(encoder: tf.keras.Model, decoder: tf.keras.Model, optimizer: tf.keras.optimizers.Adam,
          epochs: int, checkpoint: tf.train.CheckpointManager, train_data_path: str, max_len: int,
          vocab_size: int, batch_size: int, buffer_size: int, checkpoint_save_freq: int,
          dict_path: str = "", valid_data_split: float = 0.0, valid_data_path: str = "",
          max_train_data_size: int = 0, max_valid_data_size: int = 0):
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
    :param buffer_size: Dataset加载缓存大小
    :param batch_size: Dataset加载批大小
    :param dict_path: 字典路径，若使用phoneme则不用传
    :param valid_data_split: 用于从训练数据中划分验证数据
    :param valid_data_path: 验证数据文本路径
    :param max_train_data_size: 最大训练数据量
    :param max_valid_data_size: 最大验证数据量
    :param checkpoint_save_freq: 检查点保存频率
    """
    _, train_dataset, valid_dataset, steps_per_epoch, valid_steps_per_epoch = \
        load_data(train_data_path=train_data_path, max_len=max_len, vocab_size=vocab_size,
                  batch_size=batch_size, buffer_size=buffer_size, dict_path=dict_path,
                  valid_data_split=valid_data_split, valid_data_path=valid_data_path,
                  max_train_data_size=max_train_data_size, max_valid_data_size=max_valid_data_size)

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        start_time = time.time()
        total_loss = 0

        for (batch, (audio_feature, sentence)) in enumerate(train_dataset.take(steps_per_epoch)):
            print(audio_feature)
            print(sentence)
            exit(0)
            batch_start = time.time()
            # audio_feature_input =

            batch_loss, mel_outputs = _train_step(encoder, decoder, optimizer, sentence,
                                                  mel, mel_input, stop_token)
            total_loss += batch_loss

            print('\r{}/{} [Batch {} Loss {:.4f} {:.1f}s]'.format(
                (batch + 1), steps_per_epoch, batch + 1, batch_loss.numpy(), (time.time() - batch_start)), end="")

        print(' - {:.0f}s/step - loss: {:.4f}'.format((time.time() - start_time) / steps_per_epoch,
                                                      total_loss / steps_per_epoch))

        if (epoch + 1) % checkpoint_save_freq == 0:
            checkpoint.save()
            _valid_step(encoder=encoder, decoder=decoder, dataset=valid_dataset,
                        num_mel=num_mel, steps_per_epoch=valid_steps_per_epoch)

    return mel_outputs


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
    mel_loss = tf.keras.losses.MeanSquaredError()(mel_pred, mel_target) \
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
