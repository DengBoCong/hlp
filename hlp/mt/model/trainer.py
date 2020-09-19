import tensorflow as tf
import time
from model import layers
import config.get_config as _config
from common import preprocess


def train_step(inp, tar , transformer,loss_object,optimizer,train_loss,train_accuracy):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = layers.create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = layers.loss_function(tar_real, predictions ,loss_object)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


def train():

    # 加载数据集
    print("开始加载数据集..")
    input_tensor, target_tensor, inp_tokenizer, target_tokenizer = \
        preprocess.get_data(_config.path_to_file,_config.NUM_EXAMPLES)
    train_dataset, val_dataset = preprocess.split_batch(input_tensor,target_tensor,
                                                        _config.BUFFER_SIZE,_config.BATCH_SIZE,_config.TEST_SIZE)

    input_vocab_size = len(inp_tokenizer.word_index) + 2
    print("input_vocab_size:", input_vocab_size)
    target_vocab_size = len(target_tokenizer.word_index) + 2
    print("target_vocab_size:", target_vocab_size)

    # 设置transformer
    learning_rate = layers.CustomSchedule(_config.d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')

    transformer = layers.Transformer(_config.num_layers, _config.d_model, _config.num_heads, _config.dff,
                              input_vocab_size, target_vocab_size,
                              pe_input=input_vocab_size,
                              pe_target=target_vocab_size,
                              rate=_config.dropout_rate)

    # 检查点
    checkpoint_path = _config.checkpoint_path

    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # 如果检查点存在，则恢复最新的检查点。
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    # 开始训练
    for epoch in range(_config.EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> en, tar -> ch
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar, transformer, loss_object,
                       optimizer, train_loss, train_accuracy)

            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                            train_loss.result(),
                                                            train_accuracy.result()))

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
