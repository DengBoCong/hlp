import time

import tensorflow as tf

from hlp.mt.config import get_config as _config
from hlp.mt.lm import language_model, lm_preprocess
from hlp.utils import optimizers
from hlp.utils import train_history


def _train_step(sequences, lm, optimizer, train_loss, train_accuracy):
    """一个训练步
    @param sequences: 已编码的一个batch的数据集  shape --> (batch_size, seq_length)
    @param lm: 语言模型实例
    @param optimizer: 优化器
    """
    seq_input = sequences[:, :-1]
    seq_real = sequences[:, 1:]

    with tf.GradientTape() as tape:
        predictions = lm(seq_input)
        loss = optimizers.loss_func_mask(seq_real, predictions)

    gradients = tape.gradient(loss, lm.trainable_variables)
    optimizer.apply_gradients(zip(gradients, lm.trainable_variables))

    train_loss(loss)
    train_accuracy(seq_real, predictions)


def _train_epoch(dataset, model, optimizer, train_loss, train_accuracy, sample_sum):
    """
    对dataset进行训练并打印相关信息
    """
    trained_seq_sum = 0
    for batch, sequences in enumerate(dataset):
        _train_step(sequences, model, optimizer, train_loss, train_accuracy)
        trained_seq_sum += _config.lm_BATCH_SIZE
        print('\r{}/{} [batch {} loss {:.4f} accuracy {:.4f}]'.format(trained_seq_sum,
                                                                      sample_sum,
                                                                      batch + 1,
                                                                      train_loss.result()
                                                                      , train_accuracy.result()), end='')
    print('\r{}/{} [==============================]'.format(sample_sum, sample_sum), end='')


def train(epochs=_config.lm_EPOCHS, validation_split=0.0,
          min_delta=0.00003, patience=10, validation_freq=1):
    """训练
    @param epochs: 训练轮次
    @return: 训练过程history
    @param validation_split: 验证集划分比例
    @param min_delta: 增大或减小的阈值，只有大于这个部分才算作improvement
    @param patience: 能够容忍多少个val_accuracy都没有improvement
    @param validation_freq: 验证频率
    @return: history，包含训练过程中所有的指标
    """
    max_acc = 0
    patience_num = 0

    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    history = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}
    encoded_sequences_path_train = lm_preprocess.get_encoded_sequences_path(_config.lm_language, postfix='_train')

    tokenizer, vocab_size, max_sequence_length = lm_preprocess.train_preprocess()
    train_dataset, val_dataset = lm_preprocess.get_dataset(encoded_sequences_path_train, train_size=validation_split)

    lm = language_model.LanguageModel(vocab_size, _config.lm_d_embedding, _config.lm_BATCH_SIZE, _config.lm_d_rnn)

    # 检查点设置，如果检查点存在，则恢复最新的检查点。
    ckpt = tf.train.Checkpoint(language_model=lm, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, _config.lm_checkpoint_path, max_to_keep=_config.max_checkpoints_num)
    if language_model.check_point():
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('已恢复至最新检查点！')

    train_batch_sum = int((_config.lm_num_sentences*_config.lm_train_size)//_config.lm_BATCH_SIZE)
    val_batch_sum = int((_config.lm_num_sentences*(1-_config.lm_train_size))//_config.lm_BATCH_SIZE)
    train_seq_sum = int(train_batch_sum * _config.lm_BATCH_SIZE)
    val_seq_sum = int(val_batch_sum * _config.lm_BATCH_SIZE)

    print("开始训练...")
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()

        _train_epoch(train_dataset, lm, optimizer, train_loss, train_accuracy, train_seq_sum)

        history['accuracy'].append(train_accuracy.result().numpy())
        history['loss'].append(train_loss.result().numpy())

        epoch_time = (time.time() - start)
        step_time = epoch_time * _config.BATCH_SIZE / (_config.lm_num_sentences*_config.lm_train_size)

        # 验证部分
        # 若到达所设置验证频率或最后一个epoch，并且validate_from_txt为False和train_size不同时满足时使用验证集验证
        if (epoch + 1) % validation_freq == 0 or (epoch + 1) == _config.EPOCHS:
            temp_loss = train_loss.result()
            temp_acc = train_accuracy.result()
            train_loss.reset_states()
            train_accuracy.reset_states()

            _train_epoch(val_dataset, lm, optimizer, train_loss, train_accuracy, train_seq_sum+val_seq_sum)

            history['val_accuracy'].append(train_accuracy.result().numpy())
            history['val_loss'].append(train_loss.result().numpy())
            print(' - {:.0f}s - {:.0f}ms/step - loss: {:.4f} - accuracy {:.4f} - val_loss: {:.4f} - val_accuracy {:.4f}'
                  .format(epoch_time, step_time * 1000, temp_loss, temp_acc, train_loss.result(),
                          train_accuracy.result()))
            # stop-early判断
            if train_accuracy.result().numpy() >= (max_acc * (1 + min_delta)):
                max_acc = train_accuracy.result().numpy()
                patience_num = 0
            else:
                patience_num += 1

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

    train_history.show_and_save_history(history, _config.result_save_dir, _config.lm_validation_freq)
    return history


def main():
    train(validation_split=1-_config.train_size)


if __name__ == '__main__':
    main()
