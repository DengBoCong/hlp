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


def _train_epoch(dataset, transformer, optimizer, train_loss, train_accuracy, batch_sum, sample_sum):
    """
    对dataset进行训练并打印相关信息
    """
    for (batch, (inp, tar)) in enumerate(dataset):
        _train_step(inp, tar, transformer, optimizer, train_loss, train_accuracy)
        batch_sum = batch_sum + len(inp)
        print('\r{}/{} [batch {} loss {:.4f} accuracy {:.4f}]'.format(batch_sum, sample_sum, batch + 1
                                                                      , train_loss.result()
                                                                      , train_accuracy.result()), end='')
    print('\r{}/{} [==============================]'.format(sample_sum, sample_sum), end='')


def train(transformer, cache=True, min_delta=0.00003, patience=10):
    """
    cache:若为True则将数据集都加载进内存进行训练，否则分批次加载内存训练
    min_delta:增大或减小的阈值，只有大于这个部分才算作improvement
    patience：能够容忍多少个eval_accuracy都没有improvement
    stop-early对val_accuracy进行监控，若连续patience个验证集上未超过min_delta增幅，则停止训练
    """
    # stop-early参数初始化
    max_acc = 0
    patience_num = 0

    # 模型变量初始化
    learning_rate = CustomSchedule(_config.d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    history = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}
    if cache:
        # 若为True，则将全部数据集加载进内存
        train_dataset, val_dataset = preprocess.split_batch()

    # 检查点设置，如果检查点存在，则恢复最新的检查点。
    checkpoint_path = _config.checkpoint_path
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=_config.max_checkpoints_num)
    if preprocess.check_point():
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('已恢复至最新检查点！')

    # 训练相关参数初始化
    batch_sum_train = 0
    sample_sum_train = int((_config.num_sentences * (1 - _config.val_size)) // _config.BATCH_SIZE * _config.BATCH_SIZE)
    sample_sum_val = int((_config.num_sentences * _config.val_size) // _config.BATCH_SIZE * _config.BATCH_SIZE)
    steps = _config.num_sentences // _config.BATCH_SIZE

    print("开始训练...")
    for epoch in range(_config.EPOCHS):
        print('Epoch {}/{}'.format(epoch + 1, _config.EPOCHS))
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()
        # 训练部分
        if cache:
            # cache为True,从内存中train_dataset取batch进行训练
            _train_epoch(train_dataset, transformer, optimizer, train_loss, train_accuracy
                         , batch_sum_train, sample_sum_train)
        else:
            # cache为False,从生成器中取batch进行训练
            generator_train = preprocess.generate_batch_from_file(steps * (1 - _config.val_size), 0, _config.BATCH_SIZE)
            _train_epoch(generator_train, transformer, optimizer, train_loss, train_accuracy
                         , batch_sum_train, sample_sum_train)
        history['accuracy'].append(train_loss.result().numpy())
        history['loss'].append(train_accuracy.result().numpy())
        epoch_time = (time.time() - start)
        step_time = epoch_time * _config.BATCH_SIZE / sample_sum_train

        # 验证部分
        # 若到达所设置验证频率或最后一个epoch，使用验证集验证
        if (epoch + 1) % _config.validation_freq == 0 or (epoch + 1) == _config.EPOCHS:
            temp_loss = train_loss.result()
            temp_acc = train_accuracy.result()
            train_loss.reset_states()
            train_accuracy.reset_states()
            if cache:
                # cache为True,从内存中val_dataset取batch进行训练
                _train_epoch(val_dataset, transformer, optimizer, train_loss, train_accuracy
                             , sample_sum_train, sample_sum_train+sample_sum_val)
            else:
                # cache为False,从生成器中取batch进行训练
                generator_val = preprocess.generate_batch_from_file(steps, steps * (1 - _config.val_size)
                                                                    , _config.BATCH_SIZE)
                _train_epoch(generator_val, transformer, optimizer, train_loss, train_accuracy
                             , sample_sum_train, sample_sum_train+sample_sum_val)
            history['val_accuracy'].append(train_loss.result().numpy())
            history['val_loss'].append(train_accuracy.result().numpy())
            print(' - {:.0f}s - {:.0f}ms/step - loss: {:.4f} - accuracy {:.4f} - val_loss: {:.4f} - val_accuracy {:.4f}'
                  .format(epoch_time, step_time * 1000, temp_loss, temp_acc, train_loss.result(),
                          train_accuracy.result()))
            # stop-early判断
            if train_accuracy.result().numpy() >= (max_acc * (1 + min_delta)):
                max_acc = train_accuracy.result().numpy()
                patience_num = 0
            else:
                patience_num += 1
        else:
            print(' - {:.0f}s - {:.0f}ms/step - loss: {:.4f} - accuracy {:.4f}'
                  .format(epoch_time, step_time * 1000, train_loss.result(), train_accuracy.result()))

        if (epoch + 1) % _config.checkpoints_save_freq == 0:
            ckpt_save_path = ckpt_manager.save()
            print('检查点已保存至：{}'.format(ckpt_save_path))

        # 若连续patience个val_accuracy不达标，则停止训练
        if patience_num == patience:
            print('检测到连续%d个验证集增长不达标，停止训练' % patience)
            break

    if (epoch + 1) % _config.checkpoints_save_freq != 0:
        ckpt_save_path = ckpt_manager.save()
        print('检查点已保存至：{}'.format(ckpt_save_path))

    print('训练完毕！')
    return history
