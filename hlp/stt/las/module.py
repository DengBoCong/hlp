import os
import time
import tensorflow as tf
from hlp.stt.utils.load_dataset import load_data
from hlp.utils.optimizers import loss_func_mask


def train(epochs: int, train_data_path: str, max_len: int, vocab_size: int,
          batch_size: int, buffer_size: int, checkpoint_save_freq: int,
          checkpoint: tf.train.CheckpointManager, model: tf.keras.Model,
          optimizer: tf.keras.optimizers.Adam,
          dict_path: str = "", valid_data_split: float = 0.0, valid_data_path: str = "",
          max_train_data_size: int = 0, max_valid_data_size: int = 0):
    """
    训练模块
    :param epochs: 训练周期
    :param train_data_path: 文本数据路径
    :param max_len: 文本序列最大长度
    :param vocab_size: 词汇大小
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
    tokenizer, train_dataset, valid_dataset, steps_per_epoch, valid_steps_per_epoch = \
        load_data(train_data_path=train_data_path, max_len=max_len, vocab_size=vocab_size,
                  batch_size=batch_size, buffer_size=buffer_size, dict_path=dict_path,
                  valid_data_split=valid_data_split, valid_data_path=valid_data_path,
                  max_train_data_size=max_train_data_size, max_valid_data_size=max_valid_data_size)

    for epoch in range(epochs):
        start = time.time()
        enc_hidden = model.initialize_hidden_state()
        total_loss = 0
        batch_start = time.time()

        print("Epoch {}/{}".format(epoch + 1, epochs))
        for (batch, (audio_feature, sentence)) in enumerate(train_dataset.take(steps_per_epoch)):
            batch_loss = _train_step(audio_feature, sentence,
                                     enc_hidden, tokenizer, model, optimizer, batch_size)

            total_loss += batch_loss

            print('Epoch {} Batch {} Loss {:.4f} - {:.4f} sec'.format(epoch + 1, batch, batch_loss.numpy(),
                                                                      time.time() - batch_start))
            batch_start = time.time()

        print('Epoch {} Loss {:.4f} - {:.4f} sec'.format(epoch + 1, total_loss / steps_per_epoch, time.time() - start))

        if (epoch + 1) % checkpoint_save_freq == 0:
            checkpoint.save()
            # norm_rates_lers, norm_aver_lers = compute_metric(model, val_data_generator,
            #                                                  val_batchs, val_batch_size)
            # print("平均字母错误率: ", norm_aver_lers)



def _train_step(audio_feature, sentence, enc_hidden, tokenizer, model, las_optimizer, train_batch_size):
    loss = 0
    dec_input = tf.expand_dims([tokenizer.word_index.get('<start>')] * train_batch_size, 1)
    with tf.GradientTape() as tape:
        # 解码器输入符号
        for t in range(1, sentence.shape[1]):
            print(t)
            predictions, _ = model(audio_feature, enc_hidden, dec_input)

            loss += loss_func_mask(sentence[:, t], predictions)  # 根据预测计算损失
            print("loss===={}".format(loss))
            # 使用导师驱动，下一步输入符号是训练集中对应目标符号
            dec_input = sentence[:, t]
            dec_input = tf.expand_dims(dec_input, 1)

    batch_loss = (loss / int(sentence.shape[1]))
    print("batch_loss===={}".format(batch_loss))
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)  # 计算损失对参数的梯度
    las_optimizer.apply_gradients(zip(gradients, variables))  # 优化器反向传播更新参数
    return batch_loss


def load_checkpoint(model: tf.keras.Model, checkpoint_dir: str,
                    execute_type: str, checkpoint_save_size: int):
    """
    恢复检查点
    """
    # 如果检查点存在就恢复，如果不存在就重新创建一个
    checkpoint = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=checkpoint_save_size)

    if os.path.exists(checkpoint_dir):
        if ckpt_manager.latest_checkpoint:
            checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
        if execute_type == "recognize":
            print("没有检查点，请先执行train模式")
            exit(0)

    return ckpt_manager
