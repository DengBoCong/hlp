import tensorflow as tf
from model import transformer as _transformer
from config import get_config as _config
import time
from common import preprocess


# 自定义优化器（Optimizer）
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def _loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def _train_step(inp, tar, transformer, optimizer, train_loss, train_accuracy):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = _transformer.create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = _loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)
    return transformer, optimizer, train_loss, train_accuracy


def train(transformer, cache=True):
    """
    cache:若为True则将数据集都加载进内存进行训练，否则分批次加载内存训练
    """
    # optimizer, train_loss, train_accuracy
    learning_rate = CustomSchedule(_config.d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    if cache:
        # 若为True，则将全部数据集加载进内存
        train_dataset, val_dataset = preprocess.split_batch()

    # 检查点设置，如果检查点存在，则恢复最新的检查点。
    checkpoint_path = _config.checkpoint_path
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=100)
    if preprocess.check_point():
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('已恢复至最新检查点！')

    print("开始训练...")
    for epoch in range(_config.EPOCHS):
        print('Epoch {}/{}'.format(epoch + 1, _config.EPOCHS))
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        batch_sum = 0
        sample_sum = int((_config.num_sentences * (1 - _config.test_size)) // _config.BATCH_SIZE * _config.BATCH_SIZE)
        steps = _config.num_sentences//_config.BATCH_SIZE

        if cache:
            # cache为True,从内存中train_dataset取batch进行训练
            for (batch, (inp, tar)) in enumerate(train_dataset):
                transformer, optimizer, train_loss, train_accuracy = \
                    _train_step(inp, tar, transformer, optimizer, train_loss, train_accuracy)
                batch_sum = batch_sum + len(inp)
                print('\r{}/{} [Batch {} Loss {:.4f} Accuracy {:.4f}]'.format(batch_sum, sample_sum, batch
                                                                              , train_loss.result()
                                                                              , train_accuracy.result()), end='')
        else:
            # cache为True,从生成器中train_dataset取batch进行训练
            generator = preprocess.generate_batch_from_file(steps, _config.BATCH_SIZE)
            for (batch, (inp, tar)) in enumerate(generator):
                transformer, optimizer, train_loss, train_accuracy = \
                    _train_step(inp, tar, transformer, optimizer, train_loss, train_accuracy)
                batch_sum = batch_sum + len(inp)
                print('\r{}/{} [Batch {} Loss {:.4f} Accuracy {:.4f}]'.format(batch_sum, sample_sum, batch+1
                                                                              , train_loss.result()
                                                                              , train_accuracy.result()), end='')

        epoch_time = (time.time() - start)
        step_time = epoch_time * _config.BATCH_SIZE / sample_sum
        print(' - {:.0f}s - {:.0f}ms/step - loss: {:.4f} - Accuracy {:.4f}'.format(epoch_time, step_time * 1000
                                                                                   , train_loss.result()
                                                                                   , train_accuracy.result()))

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

    if (epoch + 1) % 5 != 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

    print('训练完毕！')





