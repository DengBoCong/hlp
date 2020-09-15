import os
import sys
import time
import jieba
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import common.layers as layers
import common.data_utils as _data
import config.get_config as _config
import model.transformer.model as model










def train():
    print('训练开始，正在准备数据中...')
    step_per_epoch = len(_data.input_tensor) // _config.BATCH_SIZE
    checkpoint_dir = _config.transformer_train_data
    is_exist = Path(checkpoint_dir)
    if not is_exist.exists():
        os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt = tf.io.gfile.listdir(checkpoint_dir)
    if ckpt:
        model.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    dataset = tf.data.Dataset.from_tensor_slices((_data.input_tensor, _data.target_tensor)).cache().shuffle(_config.BUFFER_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(_config.BATCH_SIZE, drop_remainder=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    # model.model.fit(dataset, epochs=EPOCHS)
    for epoch in range(_config.epochs):
        print("当前训练epoch为：{}".format(epoch + 1))
        start_time = time.time()
        model.train_loss.reset_states()
        model.train_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(dataset.take(step_per_epoch)):
            model.train_step(inp, tar)
        step_time = (time.time() - start_time)
        print('当前epoch损失：{:.4f}，精度：{:.4f}'.format(model.train_loss.result(), model.train_accuracy.result()))
        print('当前epoch耗时：{:.4f}'.format(step_time))
        model.checkpoint.save(file_prefix=checkpoint_prefix)
        sys.stdout.flush()
    print('训练结束')


