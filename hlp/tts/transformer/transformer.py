import tensorflow as tf


def encoder_pre_net(vocab_size: int, embedding_dim: int, units: int, dilation_rate: int = 1,
                    kernel_size: int = 5, strides: int = 1, use_bias: bool = True) -> tf.keras.Model:
    """
    encoder pre-net，由卷积网络组成，用于将phoneme序列输入到同一网络中
    Args:
        vocab_size: 词汇表大小
        embedding_dim: 嵌入层维数
        units: 隐藏层单元数
        dilation_rate: 指定用于扩张卷积的扩张率
        kernel_size: 卷积核大小
        strides: 步幅长度
        use_bias: 是否使用偏置
    Returns:
    """
    inputs = tf.keras.Input(shape=(None,))
    embeddings = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    conv1 = tf.keras.layers.Conv1D(filters=units, kernel_size=kernel_size,
                                   dilation_rate=dilation_rate, kernel_initializer='glorot_uniform',
                                   strides=strides, use_bias=use_bias, padding='valid', activation='relu')
    conv2 = tf.keras.layers.Conv1D(filters=units, kernel_size=kernel_size,
                                   dilation_rate=dilation_rate, kernel_initializer='glorot_uniform',
                                   strides=strides, use_bias=use_bias, padding='valid', activation='relu')
    conv3 = tf.keras.layers.Conv1D(filters=units, kernel_size=kernel_size,
                                   dilation_rate=dilation_rate, kernel_initializer='glorot_uniform',
                                   strides=strides, use_bias=use_bias, padding='valid', activation='relu')
    outputs = tf.keras.layers.Dropout(0.2)(conv1(embeddings))
    outputs = tf.keras.layers.Dropout(0.2)(conv2(outputs))
    outputs = tf.keras.layers.Dropout(0.2)(conv3(outputs))

    outputs = tf.keras.layers.Dense(units)(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs)
