import tensorflow as tf
from hlp.tts.utils.layers import PostNet
from hlp.tts.utils.layers import ConvDropBN
from hlp.tts.utils.layers import DecoderPreNet
from hlp.utils.layers import positional_encoding
from hlp.utils.layers import create_padding_mask
from hlp.utils.layers import create_look_ahead_mask
from hlp.utils.layers import transformer_encoder_layer
from hlp.utils.layers import transformer_decoder_layer


def encoder_pre_net(vocab_size: int, embedding_dim: int, encoder_pre_net_conv_num: int,
                    encoder_pre_net_filters: int, encoder_pre_net_kernel_size: int,
                    encoder_pre_net_activation: str, encoder_pre_net_dropout: float) -> tf.keras.Model:
    """
    :param vocab_size: 词汇大小
    :param embedding_dim: 嵌入层维度
    :param encoder_pre_net_conv_num: 卷积层数量
    :param encoder_pre_net_filters: 输出空间维数
    :param encoder_pre_net_kernel_size: 卷积核大小
    :param encoder_pre_net_activation: 激活方法
    :param encoder_pre_net_dropout: dropout采样率
    """
    inputs = tf.keras.Input(shape=(None,))
    outputs = tf.keras.layers.Embedding(vocab_size, embedding_dim)(inputs)

    for i in range(encoder_pre_net_conv_num):
        outputs = ConvDropBN(filters=encoder_pre_net_filters,
                             kernel_size=encoder_pre_net_kernel_size,
                             activation=encoder_pre_net_activation,
                             dropout_rate=encoder_pre_net_dropout)(outputs)
    outputs = tf.keras.layers.Dense(embedding_dim, activation="relu")(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def encoder(vocab_size: int, embedding_dim: int, encoder_pre_net_conv_num, num_layers: int,
            encoder_pre_net_filters: int, encoder_pre_net_kernel_size: int,
            encoder_pre_net_activation: str, encoder_units: int, num_heads: int,
            encoder_layer_dropout_rate: float = 0.1, encoder_pre_net_dropout: float = 0.1,
            dropout: float = 0.1) -> tf.keras.Model:
    """
    transformer tts的encoder层
    :param vocab_size: 词汇大小
    :param embedding_dim: 嵌入层维度
    :param encoder_pre_net_conv_num: 卷积层数量
    :param num_layers: encoder层数量
    :param encoder_pre_net_filters: 输出空间维数
    :param encoder_pre_net_kernel_size: 卷积核大小
    :param encoder_pre_net_activation: 激活方法
    :param encoder_units: 单元大小
    :param encoder_pre_net_dropout: pre_net的dropout采样率
    :param dropout: encoder的dropout采样率
    :param encoder_layer_dropout_rate: encoder_layer的dropout采样率
    :param num_heads: 头注意力数量
    """
    inputs = tf.keras.Input(shape=(None,))
    padding_mask = tf.keras.layers.Lambda(create_padding_mask,
                                          output_shape=(1, 1, None))(inputs)
    pos_encoding = positional_encoding(vocab_size, embedding_dim)
    alpha = tf.Variable(initial_value=1., trainable=True, dtype=tf.float32)

    pre_net = encoder_pre_net(vocab_size, embedding_dim, encoder_pre_net_conv_num,
                              encoder_pre_net_filters, encoder_pre_net_kernel_size,
                              encoder_pre_net_activation, encoder_pre_net_dropout)(inputs)
    pre_net *= tf.math.sqrt(tf.cast(embedding_dim, tf.float32))
    # 按论文结论，这里需要对位置编码增加一个可训练权重
    pre_net = pre_net + pos_encoding[:, :tf.shape(pre_net)[1], :] * alpha

    outputs = tf.keras.layers.Dropout(rate=dropout)(pre_net)

    for i in range(num_layers):
        outputs = transformer_encoder_layer(units=encoder_units, d_model=embedding_dim,
                                            dropout=encoder_layer_dropout_rate,
                                            name="transformer_encoder_layer_{}".format(i),
                                            num_heads=num_heads)([outputs, padding_mask])

    return tf.keras.Model(inputs=inputs, outputs=[outputs, padding_mask])


def decoder(vocab_size: int, embedding_dim: int, num_layers: int, decoder_units: int,
            num_heads: int, max_mel_length: int, num_mel: int, decoder_pre_net_layers_num: int,
            post_net_conv_num: int, post_net_filters: int, post_net_kernel_sizes: int,
            decoder_pre_net_dropout_rate: float = 0.1, dropout: float = 0.1,
            post_net_dropout: float = 0.1, post_net_activation: str = "tanh") -> tf.keras.Model:
    """
    :param vocab_size: 词汇大小
    :param embedding_dim: 嵌入层维度
    :param num_layers: encoder层数量
    :param decoder_units: 单元大小
    :param num_heads: 头注意力数量
    :param max_mel_length: 最长序列长度
    :param num_mel: 产生的梅尔带数
    :param decoder_pre_net_layers_num: pre_net层数
    :param decoder_pre_net_dropout_rate: pre_net的dropout采样率
    :param dropout: decoder的dropout采样率
    :param post_net_conv_num: post_net的卷积层数量
    :param post_net_filters: post_net卷积输出空间维数
    :param post_net_kernel_sizes: post_net卷积核大小
    :param post_net_dropout: post_net的dropout采样率
    :param post_net_activation: post_net卷积激活函数
    """
    enc_outputs = tf.keras.Input(shape=(None, None))
    dec_inputs = tf.keras.Input(shape=(None, num_mel))
    padding_mask = tf.keras.Input(shape=(1, 1, None))
    alpha = tf.Variable(initial_value=1., trainable=True, dtype=tf.float32)
    pos_encoding = positional_encoding(vocab_size, embedding_dim)

    look_ahead_mask = tf.keras.layers.Lambda(_combine_mask,
                                             output_shape=(1, None, None))(dec_inputs)

    decoder_pre_net = DecoderPreNet(embedding_dim, decoder_pre_net_layers_num,
                                    decoder_pre_net_dropout_rate)(dec_inputs)

    decoder_pre_net *= tf.math.sqrt(tf.cast(embedding_dim, tf.float32))
    decoder_pre_net = decoder_pre_net + alpha * pos_encoding[:, :tf.shape(decoder_pre_net)[1], :]
    dec_outputs = tf.keras.layers.Dropout(rate=dropout)(decoder_pre_net)

    for i in range(num_layers):
        dec_outputs = transformer_decoder_layer(
            units=decoder_units, d_model=embedding_dim, num_heads=num_heads,
            dropout=dropout, name="transformer_decoder_layer_{}".format(i)
        )(inputs=[dec_outputs, enc_outputs, look_ahead_mask, padding_mask])

    mel_outputs = tf.keras.layers.Dense(units=num_mel)(dec_outputs)
    stop_token_outputs = tf.keras.layers.Dense(units=1)(dec_outputs)
    stop_token_outputs = tf.math.sigmoid(stop_token_outputs)

    post_net_inputs = tf.transpose(mel_outputs, [0, 2, 1])
    outputs = PostNet(post_net_conv_num, post_net_conv_num, post_net_filters,
                      post_net_kernel_sizes, post_net_dropout, post_net_activation, num_mel)(post_net_inputs)

    outputs = outputs + post_net_inputs
    outputs = tf.transpose(outputs, [0, 2, 1])

    return tf.keras.Model(inputs=[enc_outputs, dec_inputs, padding_mask],
                          outputs=[mel_outputs, outputs, stop_token_outputs])


def _combine_mask(seq: tf.Tensor):
    """
    对input中的不能见单位进行mask，专用于mel序列
    :param seq: 输入序列
    :return: mask
    """
    look_ahead_mask = create_look_ahead_mask(seq)
    padding_mask = _create_decoder_padding_mask(seq)
    return tf.maximum(look_ahead_mask, padding_mask)


def _create_decoder_padding_mask(seq: tf.Tensor):
    """
    用于创建输入序列的扩充部分的mask，专用于mel序列
    :param seq: 输入序列
    :return: mask
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    seq = seq[:, :, 0]
    return seq[:, tf.newaxis, tf.newaxis, :]
