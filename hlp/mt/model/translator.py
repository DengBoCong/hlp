import matplotlib.pyplot as plt
import tensorflow as tf
import config.get_config as _config
from common import preprocess
from model import layers


def evaluate(inp_sentence,inp_tokenizer,target_tokenizer,transformer,dic_keys):

    # 输入语句是英语，增加开始和结束标记
    inp_sentence = '<start> ' + inp_sentence + ' <end>'
    inp_sentence = [inp_tokenizer.word_index[i] for i in inp_sentence.split(' ')
                    if i in dic_keys]  # token编码 对不在词典中的单词不加入编码
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # 因为目标是英语，输入 transformer 的第一个词应该是
    # 中文的开始标记。
    decoder_input = [target_tokenizer.word_index['<start>']]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(_config.TARGET_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = layers.create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # 从 seq_len 维度选择最后一个词
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        # print('predicted_id：', predicted_id)
        # 如果 predicted_id 等于结束标记，就返回结果
        if predicted_id == [target_tokenizer.word_index['<end>']]:
            return tf.squeeze(output, axis=0), attention_weights

        # 连接 predicted_id 与输出，作为解码器的输入传递到解码器。
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def plot_attention_weights(attention, sentence, result, layer,inp_tokenizer,target_tokenizer):
    fig = plt.figure(figsize=(16, 8))

    sentence = inp_tokenizer.texts_to_sequences(sentence)

    attention = tf.squeeze(attention[layer], axis=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        # 画出注意力权重
        ax.matshow(attention[head][:-1, :], cmap='viridis')

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(sentence) + 2))
        ax.set_yticks(range(len(result)))

        ax.set_ylim(len(result) - 1.5, -0.5)

        ax.set_xticklabels(
            ['<start> '] + [inp_tokenizer.index_word[i] for i in sentence] + [' <end>'],
            fontdict=fontdict, rotation=90)

        ax.set_yticklabels([target_tokenizer.index_word[i] for i in result
                            if i < (len(target_tokenizer.index_word) + 2)],
                           fontdict=fontdict)

        ax.set_xlabel('Head {}'.format(head + 1))

    plt.tight_layout()
    plt.show()


# 翻译句子
def translate(sentence, plot=''):

    # 加载数据集
    # print("开始加载数据集..")
    input_tensor, target_tensor, inp_tokenizer, target_tokenizer = \
        preprocess.get_data(_config.path_to_file,_config.NUM_EXAMPLES)
    dic_keys = [i for i in inp_tokenizer.word_index]
    input_vocab_size = len(inp_tokenizer.word_index) + 2
    # print("input_vocab_size:", input_vocab_size)
    target_vocab_size = len(target_tokenizer.word_index) + 2
    # print("target_vocab_size:", target_vocab_size)

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

    result, attention_weights = evaluate(sentence,inp_tokenizer,target_tokenizer,transformer,dic_keys)
    # print('result:',result)
    predicted_sentence = [target_tokenizer.index_word[i.numpy()] for i in result
                          if i < (len(target_tokenizer.index_word) + 2) and i != [target_tokenizer.word_index['<start>']]]

    predicted_sentence = ''.join(predicted_sentence)
    # print('Input: {}'.format(sentence))
    # print('Predicted translation: {}'.format(predicted_sentence))

    # attention图
    if plot:
        plot_attention_weights(attention_weights, sentence, result, plot,inp_tokenizer,target_tokenizer)


    return predicted_sentence

