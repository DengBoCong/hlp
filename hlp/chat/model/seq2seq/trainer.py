import io
import os
import sys
import time
from pathlib import Path
import tensorflow as tf
import model.seq2seq.model as model
import config.get_config as _config
from common.data_utils import load_dataset


def train():
    """
    seq2seq的训练方法，这里面主要就是包括数据预处理、模型保存点目录检查等
    相关的工作，按照batch_size分批进行训练，最后整合各batch的loss进行进展
    状态信息的输出
    :return:
    """
    input_tensor, input_token, target_tensor, target_token = load_dataset()
    print('训练开始，正在准备数据中...')
    steps_per_epoch = len(input_tensor) // _config.BATCH_SIZE
    checkpoint_dir = _config.seq2seq_train_data
    # 这里需要检查一下是否有模型的目录，没有的话就创建，有的话就跳过
    is_exist = Path(checkpoint_dir)
    if not is_exist.exists():
        os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt = tf.io.gfile.listdir(checkpoint_dir)
    if ckpt:
        model.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    # 这里一系列的操作是加速模型的学习
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).cache().shuffle(
        _config.BUFFER_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(_config.BATCH_SIZE, drop_remainder=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    start_time = time.time()

    for epoch in range(_config.epochs):
        print('当前训练epoch为：{}'.format(epoch + 1))
        enc_hidden = model.encoder.initialize_hidden_state()
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = model.train_step(inp, targ, target_token, enc_hidden)
            total_loss += batch_loss
            print('当前batch的损失：{:.4f}'.format(batch_loss.numpy()))
        step_loss = total_loss / steps_per_epoch
        step_time = (time.time() - start_time) / (epoch + 1)
        print('epochs平均耗时：{:.4f}s'.format(step_time))
        print('当前epoch的损失为：{:.4f}'.format(step_loss.numpy()))
        model.checkpoint.save(file_prefix=checkpoint_prefix)
        sys.stdout.flush()

    print('训练结束')
