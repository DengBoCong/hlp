import tensorflow as tf


def encoder(units, vocab_size, embedding_dim, max_utterance, max_sentence):
    """
    SMN的Encoder，主要对utterances和responses进行基本的
    嵌入编码，以及对response的Word级语义进行建模
    Args:
        units: GRU单元数
        vocab_size: embedding词汇量
        embedding_dim: embedding维度
    Returns:
    """
    utterance_inputs = tf.keras.Input(shape=(max_utterance, max_sentence))
    response_inputs = tf.keras.Input(shape=(max_sentence,))

    embeddings = tf.keras.layers.Embedding(vocab_size, embedding_dim, name="encoder")
    utterance_embeddings = embeddings(utterance_inputs)
    response_embeddings = embeddings(response_inputs)

    # 这里对response进行GRU的Word级关系建模，这里用正交矩阵初始化内核权重矩阵，用于输入的线性变换。
    response_outputs = tf.keras.layers.GRU(units=units, return_sequences=True,
                                           kernel_initializer='orthogonal')(response_embeddings)

    # 将utterances的第一维和第二维进行调整，方便后面进行utterance-response配对
    # utterance_embeddings = tf.transpose(utterance_embeddings, perm=[1, 0, 2, 3])
    # # 同样的，为了后面求相似度矩阵，这里将第二维和第三维度进行调整
    # response_embeddings = tf.transpose(response_embeddings, perm=[0, 2, 1])
    # response_outputs = tf.transpose(response_outputs, perm=[0, 2, 1])

    return tf.keras.Model(inputs=[utterance_inputs, response_inputs],
                          outputs=[utterance_embeddings, response_embeddings, response_outputs])


def decoder(units, embedding_dim, max_utterance, max_sentence):
    utterance_inputs = tf.keras.Input(shape=(max_utterance, max_sentence, embedding_dim))
    response_inputs = tf.keras.Input(shape=(max_sentence, embedding_dim))
    response_gru = tf.keras.Input(shape=(max_sentence, units))
    a_matrix = tf.random.uniform(shape=(units, units), maxval=1, minval=-1)

    conv2d_layer = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding='valid',
                                          kernel_initializer='he_normal', activation='relu')
    max_polling2d_layer = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='valid')
    dense_layer = tf.keras.layers.Dense(50, activation='tanh', kernel_initializer='glorot_normal')

    # 这里需要做一些前提工作，因为我们要针对每个batch中的每个utterance进行运算，所
    # 以我们需要将batch中的utterance序列进行拆分，使得batch中的序列顺序一一匹配
    utterance_embeddings = tf.unstack(utterance_inputs, num=max_utterance, axis=1)
    matching_vectors = []
    for utterance_input in utterance_embeddings:
        # 求解第一个相似度矩阵，公式见论文
        matrix1 = tf.matmul(utterance_input, response_inputs, transpose_b=True)
        utterance_gru = tf.keras.layers.GRU(units, return_sequences=True,
                                            kernel_initializer='orthogonal')(utterance_input)
        matrix2 = tf.einsum("aij,jk->aik", utterance_gru, a_matrix)
        # matrix2 = tf.matmul(utterance_gru, a_matrix)
        matrix2 = tf.matmul(matrix2, response_gru, transpose_b=True)
        matrix = tf.stack([matrix1, matrix2], axis=3)

        conv_outputs = conv2d_layer(matrix)
        pooling_outputs = max_polling2d_layer(conv_outputs)
        flatten_outputs = tf.keras.layers.Flatten()(pooling_outputs)

        matching_vector = dense_layer(flatten_outputs)
        matching_vectors.append(matching_vector)
    vector = tf.stack(matching_vectors, axis=0)
    _, outputs = tf.keras.layers.GRU(units, return_state=True,
                                     kernel_initializer='orthogonal')(vector)

    return tf.keras.Model(inputs=[utterance_inputs, response_inputs, response_gru], outputs=outputs)


def smn(units, vocab_size, embedding_dim, max_utterance, max_sentence):
    utterances = tf.keras.Input(shape=(None, None))
    responses = tf.keras.Input(shape=(None,))
    print("utterances", utterances.shape)
    print("responses", responses.shape)

    utterances_embeddings, responses_embeddings, responses_gru = \
        encoder(units=units, vocab_size=vocab_size, embedding_dim=embedding_dim,
                max_utterance=max_utterance, max_sentence=max_sentence)(inputs=[utterances, responses])
    print("utterances_embeddings", utterances_embeddings.shape)
    print("responses_embeddings", responses_embeddings.shape)
    print("responses_gru", responses_gru.shape)
    dec_outputs = decoder(units=units, embedding_dim=embedding_dim, max_utterance=max_utterance,
                          max_sentence=max_sentence)(
        inputs=[utterances_embeddings, responses_embeddings, responses_gru])
    print("dec_outputs", dec_outputs)
    outputs = tf.keras.layers.Dense(2, kernel_initializer='glorot_normal')(dec_outputs)
    print("outputs", outputs)
    outputs = tf.nn.softmax(outputs)
    print("outputs", outputs)

    return tf.keras.Model(inputs=[utterances, responses], outputs=outputs)


if __name__ == '__main__':
    SMN = smn(512, 1000, 256, 10, 50)
