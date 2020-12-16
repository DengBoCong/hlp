import tensorflow as tf
from hlp.tts.utils.layers import ConvDropBN
from hlp.tts.utils.layers import DecoderPreNet
from hlp.utils.layers import positional_encoding
from hlp.utils.layers import create_padding_mask
from hlp.utils.layers import create_look_ahead_mask
from hlp.utils.layers import transformer_encoder_layer
from hlp.utils.layers import transformer_decoder_layer


def encoder_pre_net(vocab_size: int, embedding_dim: int, encoder_pre_net_conv_num: int,
                    encoder_pre_net_filters: int, encoder_pre_net_kernel_size: int,
                    encoder_pre_net_activation: str, encoder_pre_net_dropout: float):
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
            encoder_pre_net_activation: str, units: int, num_heads: int,
            encoder_layer_dropout_rate: float = 0.1, encoder_pre_net_dropout: float = 0.1,
            encoder_dropout_rate: float = 0.1):
    """
    transformer tts的encoder层
    :param vocab_size: 词汇大小
    :param embedding_dim: 嵌入层维度
    :param encoder_pre_net_conv_num: 卷积层数量
    :param num_layers: encoder层数量
    :param encoder_pre_net_filters: 输出空间维数
    :param encoder_pre_net_kernel_size: 卷积核大小
    :param encoder_pre_net_activation: 激活方法
    :param units: 单元大小
    :param encoder_pre_net_dropout: pre_net的dropout采样率
    :param encoder_dropout_rate: encoder的dropout采样率
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

    outputs = tf.keras.layers.Dropout(rate=encoder_dropout_rate)(pre_net)

    for i in range(num_layers):
        outputs = transformer_encoder_layer(units=units, d_model=embedding_dim,
                                            dropout=encoder_layer_dropout_rate,
                                            name="transformer_encoder_layer_{}".format(i),
                                            num_heads=num_heads)([outputs, padding_mask])

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def decoder(max_mel_length: int, num_mel: int, decoder_pre_net_units: int, decoder_pre_net_layers_num: int,
            d_model: int, decoder_pre_net_dropout_rate: float = 0.1):
    """
    :param max_mel_length: 最长序列长度
    :param num_mel: 产生的梅尔带数
    :param decoder_pre_net_units: pre_net全连接层单元数
    :param decoder_pre_net_layers_num: pre_net层数
    :param d_model: 位置数量和Embedding 维数相同
    :param decoder_pre_net_dropout_rate: pre_net的dropout采样率
    """
    inputs = tf.keras.Input(shape=(max_mel_length, num_mel))
    enc_outputs = tf.keras.Input(shape=(None, d_model))
    look_ahead_mask = tf.keras.layers.Lambda(_combine_mask,
                                             output_shape=(1, None, None))(inputs)
    padding_mask = tf.keras.layers.Lambda(_create_decoder_padding_mask, output_shape=(1, 1, None))(inputs)

    decoder_pre_net = DecoderPreNet(decoder_pre_net_units, decoder_pre_net_layers_num,
                                    decoder_pre_net_dropout_rate)(inputs)


def _combine_mask(seq: tf.Tensor):
    """
    对input中的不能见单位进行mask，专用于mel序列
    :param seq: 输入序列
    :return: mask
    """
    look_ahead_mask = create_look_ahead_mask(seq)
    padding_mask = create_padding_mask(seq)
    return tf.maximum(look_ahead_mask, padding_mask)


def _create_decoder_padding_mask(seq: tf.Tensor):
    """
    用于创建输入序列的扩充部分的mask，专用于mel序列
    :param seq: 输入序列
    :return: mask
    """
