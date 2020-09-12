import io
import os
import sys
import time
from pathlib import Path
import tensorflow as tf
import model.seq2seq.model as model
import config.get_config as _config
import common.data_utils as _data


def train():
    steps_per_epoch = len(_data.input_tensor) // _config.BATCH_SIZE

    checkpoint_dir = _config.train_data
    # 这里需要检查一下是否有模型的目录，没有的话就创建，有的话就跳过
    is_exist = Path(checkpoint_dir)
    if not is_exist.exists():
        os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt = tf.io.gfile.listdir(checkpoint_dir)
    if ckpt:
        model.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    BUFFER_SIZE = len(_data.input_tensor)
    dataset = tf.data.Dataset.from_tensor_slices((_data.input_tensor, _data.target_tensor)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(_config.BATCH_SIZE, drop_remainder=True)
    checkpoint_dir = _config.train_data
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    for epoch in range(_config.epochs):
        enc_hidden = model.encoder.initialize_hidden_state()
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = model.train_step(inp, targ, _data.target_token, enc_hidden)
            total_loss += batch_loss
        print(total_loss / steps_per_epoch)
        model.checkpoint.save(file_prefix=checkpoint_prefix)
        sys.stdout.flush()

    print('训练结束')
