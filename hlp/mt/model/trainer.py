import tensorflow as tf
from model import network
from common import self_attention
from config import get_config as _config
import time
from common import preprocess


def train_step(inp, tar, transformer, optimizer, train_loss, train_accuracy):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = self_attention.create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = network.loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)
    return transformer, optimizer, train_loss, train_accuracy


def train(path_en, path_zh, transformer, optimizer, train_loss, train_accuracy, cache=True):
    """
    cache:若为True则将数据集都加载进内存进行训练，否则分批次加载内存训练
    """
    if cache:
        # 若为True，则将全部数据集加载进内存
        train_dataset, val_dataset = preprocess.split_batch(path_en, path_zh)

    # 检查点设置，如果检查点存在，则恢复最新的检查点。
    checkpoint_path = _config.checkpoint_path
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=100)
    if ckpt_manager.latest_checkpoint:
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

        # inp -> english, tar -> chinese
        if cache:
            # cache为True,从内存中train_dataset取batch进行训练
            for (batch, (inp, tar)) in enumerate(train_dataset):
                transformer, optimizer, train_loss, train_accuracy = \
                    train_step(inp, tar, transformer, optimizer, train_loss, train_accuracy)
                batch_sum = batch_sum + len(inp)
                print('\r{}/{} [Batch {} Loss {:.4f} Accuracy {:.4f}]'.format(batch_sum, sample_sum, batch
                                                                              , train_loss.result()
                                                                              , train_accuracy.result()), end='')
        else:
            # cache为True,从生成器中train_dataset取batch进行训练
            generator = preprocess.generate_batch_from_file(path_en, path_zh, steps, _config.BATCH_SIZE)
            for (batch, (inp, tar)) in enumerate(generator):
                transformer, optimizer, train_loss, train_accuracy = \
                    train_step(inp, tar, transformer, optimizer, train_loss, train_accuracy)
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





