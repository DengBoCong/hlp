import time
import tensorflow as tf

import hlp.mt.common.misc
import hlp.mt.common.text_vectorize
from hlp.mt import preprocess as _pre
from hlp.mt.common import load_dataset
from hlp.mt.config import get_config as _config
from hlp.mt.model import nmt_model
from hlp.mt.model import transformer as _transformer
from hlp.utils import optimizers as _optimizers
from hlp.utils import train_history
from hlp.mt.common.misc import check_and_create


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
        loss = _optimizers.loss_func_mask(tar_real, predictions)

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
        print('\r{}/{} [batch {} loss {:.4f} accuracy {:.4f}]'.format(batch_sum,
                                                                      sample_sum,
                                                                      batch + 1,
                                                                      train_loss.result(),
                                                                      train_accuracy.result()), end='')
    print('\r{}/{} [==============================]'.format(sample_sum, sample_sum), end='')


def _train(transformer, validation_data='False', validation_split=0.0,
           cache=True, min_delta=0.00003, patience=10, validation_freq=1):
    """
    @param transformer: 训练要使用的transformer模型
    @param validation_data: 为‘True’则从指定文本加载训练集，
    @param validation_split: 验证集划分比例
    @param cache: 若为True则将数据集都加载进内存进行训练，否则使用生成器分批加载
    @param min_delta: 增大或减小的阈值，只有大于这个部分才算作improvement
    @param patience: 能够容忍多少个val_accuracy都没有improvement
    @param validation_freq: 验证频率
    @return: history，包含训练过程中所有的指标
    """
    # stop-early参数初始化
    max_acc = 0
    patience_num = 0

    # 模型变量初始化
    train_size = 1 - validation_split
    learning_rate = _optimizers.CustomSchedule(_config.d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    history = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}

    # 检查点设置，如果检查点存在，则恢复最新的检查点。
    checkpoint_path = _config.checkpoint_path
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=_config.max_checkpoints_num)
    if check_and_create(_config.checkpoint_path):
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('已恢复至最新检查点！')

    # 训练相关参数初始化
    batch_sum_train = 0
    sample_sum_val_txt = 0
    if validation_data == 'True':
        # TODO: 此处有问题
        train_size = 0.999999
        sample_sum_val_txt = _config.num_validate_sentences
    else:
        print("从训练数据中划分验证数据.")
    sample_sum_train = int((_config.num_sentences * train_size) // _config.BATCH_SIZE * _config.BATCH_SIZE)
    sample_sum_val = int((_config.num_sentences * (1 - train_size)) // _config.BATCH_SIZE * _config.BATCH_SIZE)
    steps = _config.num_sentences // _config.BATCH_SIZE

    # 读取数据
    print("加载训练数据...")
    source_sequences_path_train = hlp.mt.common.text_vectorize.get_encoded_sequences_path(_config.source_lang,
                                                                                          postfix='_train')
    target_sequences_path_train = hlp.mt.common.text_vectorize.get_encoded_sequences_path(_config.target_lang,
                                                                                          postfix='_train')
    train_dataset, val_dataset = load_dataset.get_dataset(source_sequences_path_train, target_sequences_path_train,
                                                          cache, train_size, steps)
    if validation_data == 'True':  # 从文本中加载val_dataset
        print("加载验证数据...")
        source_sequences_path_val = hlp.mt.common.text_vectorize.get_encoded_sequences_path(_config.source_lang,
                                                                                            postfix='_val')
        target_sequences_path_val = hlp.mt.common.text_vectorize.get_encoded_sequences_path(_config.target_lang,
                                                                                            postfix='_val')
        val_dataset, _ = load_dataset.get_dataset(source_sequences_path_val, target_sequences_path_val,
                                                  cache, train_size, steps)
    print("开始训练...")
    for epoch in range(_config.EPOCHS):
        print('Epoch {}/{}'.format(epoch + 1, _config.EPOCHS))
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()
        # 训练部分
        _train_epoch(train_dataset, transformer, optimizer, train_loss, train_accuracy,
                     batch_sum_train, sample_sum_train)

        history['accuracy'].append(train_accuracy.result().numpy())
        history['loss'].append(train_loss.result().numpy())
        epoch_time = (time.time() - start)
        step_time = epoch_time * _config.BATCH_SIZE / sample_sum_train

        # 验证部分
        # 若到达所设置验证频率或最后一个epoch，并且validate_from_txt为False和train_size不同时满足时使用验证集验证
        if ((epoch + 1) % validation_freq == 0 or (epoch + 1) == _config.EPOCHS) \
                and (validation_data == 'True' or train_size != 1):
            temp_loss = train_loss.result()
            temp_acc = train_accuracy.result()
            train_loss.reset_states()
            train_accuracy.reset_states()

            _train_epoch(val_dataset, transformer, optimizer, train_loss, train_accuracy,
                         sample_sum_train, sample_sum_train + sample_sum_val + sample_sum_val_txt)

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

    train_history.show_and_save_history(history, _config.result_save_dir, validation_freq)
    print('训练完毕！')
    return history


def main():
    # 进行训练所需的数据处理
    # TODO: 每次训练都重复预处理，有问题。
    vocab_size_source, vocab_size_target = _pre.train_preprocess()

    # 创建模型及相关变量
    model = nmt_model.create_model(vocab_size_source, vocab_size_target)

    # 开始训练
    _train(model, validation_data=_config.validation_data,
           validation_split=1 - _config.train_size,
           validation_freq=_config.validation_freq)


if __name__ == '__main__':
    main()
