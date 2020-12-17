import time

import tensorflow as tf

from hlp.mt.config import get_config as _config
from hlp.mt.lm import language_model, lm_preprocess
from hlp.utils import optimizers


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


def train(epochs=_config.lm_EPOCHS):
    """训练
    @param epochs: 训练轮次
    @return: 训练过程history
    """
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    history = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}
    encoded_sequences_path_train = lm_preprocess.get_encoded_sequences_path(_config.lm_language, postfix='_train')

    tokenizer, vocab_size, max_sequence_length = lm_preprocess.train_preprocess()
    train_dataset, val_dataset = lm_preprocess.get_dataset(encoded_sequences_path_train,
                                                           train_size=_config.lm_train_size)
    lm = language_model.LanguageModel(vocab_size, _config.lm_d_embedding, _config.lm_BATCH_SIZE, _config.lm_d_rnn)

    # 检查点设置，如果检查点存在，则恢复最新的检查点。
    ckpt = tf.train.Checkpoint(language_model=lm, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, _config.lm_checkpoint_path, max_to_keep=_config.max_checkpoints_num)
    if language_model.check_point():
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('已恢复至最新检查点！')

    batch_sum = int((_config.lm_num_sentences*_config.lm_train_size)//_config.lm_BATCH_SIZE)
    train_seq_sum = int(batch_sum * _config.lm_BATCH_SIZE)

    print("开始训练...")
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        trained_seq_sum = 0
        for batch, sequences in enumerate(train_dataset):
            _train_step(sequences, lm, optimizer, train_loss, train_accuracy)
            trained_seq_sum += _config.lm_BATCH_SIZE
            print('\r{}/{} [batch {} loss {:.4f} accuracy {:.4f}]'.format(trained_seq_sum, train_seq_sum, batch + 1
                                                                          , train_loss.result()
                                                                          , train_accuracy.result()), end='')
        print('\r{}/{} [==============================]'.format(train_seq_sum, train_seq_sum), end='')
        history['accuracy'].append(train_accuracy.result().numpy())
        history['loss'].append(train_loss.result().numpy())

        epoch_time = (time.time() - start)
        step_time = epoch_time * _config.BATCH_SIZE / (_config.lm_num_sentences*_config.lm_train_size)
        print(' - {:.0f}s - {:.0f}ms/step - loss: {:.4f} - accuracy {:.4f}'
              .format(epoch_time, step_time * 1000, train_loss.result(), train_accuracy.result()))

        if (epoch + 1) % _config.checkpoints_save_freq == 0:
            ckpt_save_path = ckpt_manager.save()
            print('检查点已保存至：{}'.format(ckpt_save_path))

    if (epoch + 1) % _config.checkpoints_save_freq != 0:
        ckpt_save_path = ckpt_manager.save()
        print('检查点已保存至：{}'.format(ckpt_save_path))

    return history


def main():
    train()


if __name__ == '__main__':
    main()
