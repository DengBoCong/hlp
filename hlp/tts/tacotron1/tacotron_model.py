from tensorflow.keras.layers import Input, Embedding, concatenate, RepeatVector, Dense, Reshape
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow.keras.initializers as k_init
from tensorflow.keras.layers import (Conv1D, Dense, Activation, MaxPooling1D, Add,
                                     Concatenate, Bidirectional, GRU, Dropout,
                                     BatchNormalization, Lambda, Dot, Multiply)
import tensorflow as tf

# 对文本字符和声音都使用相同的prenet，这好吗？
def pre_net(input_data):
    prenet = Dense(256)(input_data)
    prenet = Activation('relu')(prenet)
    prenet = Dropout(0.5)(prenet)

    prenet = Dense(128)(prenet)
    prenet = Activation('relu')(prenet)
    prenet = Dropout(0.5)(prenet)

    return prenet


# 多层带批规范化的1维卷积，无池化, C, B of CHBG
def conv1dbank(K, input_data):
    conv = Conv1D(filters=128, kernel_size=1,
                  strides=1, padding='same')(input_data)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    for k in range(2, K + 1):
        conv = Conv1D(filters=128, kernel_size=k,
                      strides=1, padding='same')(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)

    return conv


# 高速网络，H of CHBG
def get_highway_output(highway_input, nb_layers, activation="tanh", bias=-3):
    dim = K.int_shape(highway_input)[-1]  # dimension must be the same
    initial_bias = k_init.Constant(bias)
    for n in range(nb_layers):
        H = Dense(units=dim, bias_initializer=initial_bias)(highway_input)
        H = Activation("sigmoid")(H)
        carry_gate = Lambda(lambda x: 1.0 - x,
                            output_shape=(dim,))(H)
        transform_gate = Dense(units=dim)(highway_input)
        transform_gate = Activation(activation)(transform_gate)
        transformed = Multiply()([H, transform_gate])
        carried = Multiply()([carry_gate, highway_input])
        highway_output = Add()([transformed, carried])
    return highway_output


def encode_CBHG(input_data, K_CBHG):
    t = conv1dbank(K_CBHG, input_data)

    t = MaxPooling1D(pool_size=2, strides=1,
                              padding='same')(t)

    t = Conv1D(filters=128, kernel_size=3,
                        strides=1, padding='same')(t)
    t = BatchNormalization()(t)
    t = Activation('relu')(t)
    t = Conv1D(filters=128, kernel_size=3,
                        strides=1, padding='same')(t)
    t = BatchNormalization()(t)

    residual = Add()([input_data, t])

    highway_net = get_highway_output(residual, 4, activation='relu')

    CBHG_encoder = Bidirectional(GRU(128, return_sequences=True))(highway_net)

    return CBHG_encoder


def post_process_CBHG(input_data, K_CBHG):
    t = conv1dbank(K_CBHG, input_data)
    t = MaxPooling1D(pool_size=2, strides=1,
                              padding='same')(t)
    t = Conv1D(filters=256, kernel_size=3,
                        strides=1, padding='same')(t)
    t = BatchNormalization()(t)
    t = Activation('relu')(t)
    t = Conv1D(filters=80, kernel_size=3,
                        strides=1, padding='same')(t)
    t = BatchNormalization()(t)

    residual = Add()([input_data, t])

    highway_net = get_highway_output(residual, 4, activation='relu')

    CBHG_encoder = Bidirectional(GRU(128))(highway_net)

    return CBHG_encoder


def get_decoder_RNN_output(input_data):
    rnn1 = GRU(256, return_sequences=True)(input_data)

    inp2 = Add()([input_data, rnn1])
    rnn2 = GRU(256)(inp2)

    decoder_rnn = Add()([inp2, rnn2])

    return decoder_rnn


def get_attention_RNN():
    return GRU(256)


def get_attention_context(encoder_output, attention_rnn_output):

    attention_input = Concatenate(axis=-1)([encoder_output,
                                            attention_rnn_output])
    e = Dense(10, activation="tanh")(attention_input)
    energies = Dense(1, activation="relu")(e)
    attention_weights = Activation('softmax')(energies)
    context = Dot(axes=1)([attention_weights,
                           encoder_output])

    return context

#--------------------------------------------------------------------
class Prenet(tf.keras.Model):

    def __init__(self, config):
        super().__init__()
        self.rate = config.encoder_conv_dropout_rate
        self.prenet_units1 = config.prenet_units1
        self.prenet_units2 = config.prenet_units2

        self.fc1=tf.keras.layers.Dense(units=self.prenet_units1,activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(self.rate)
        self.fc2=tf.keras.layers.Dense(units=self.prenet_units2,activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(self.rate)

    def __call__(self, inputs):
        """Call logic."""
        x = self.fc1(inputs)
        x = self.dropout1(x)
        x = self.fc2(x)
        outputs = self.dropout2(x)

        return outputs
class CBHG_encoder(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.K_CBHG=config.k1
        self.nb_layers=config.nb_layers


    def conv1dbank(K, input_data):
        conv = Conv1D(filters=128, kernel_size=1,
                      strides=1, padding='same')(input_data)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)

        for k in range(2, K + 1):
            conv = Conv1D(filters=128, kernel_size=k,
                          strides=1, padding='same')(conv)
            conv = BatchNormalization()(conv)
            conv = Activation('relu')(conv)

        return conv


    def get_highway_output(highway_input, nb_layers, activation="tanh", bias=-3):
        dim = K.int_shape(highway_input)[-1]  # dimension must be the same
        initial_bias = k_init.Constant(bias)
        for n in range(nb_layers):
            H = Dense(units=dim, bias_initializer=initial_bias)(highway_input)
            H = Activation("sigmoid")(H)
            carry_gate = Lambda(lambda x: 1.0 - x,
                                output_shape=(dim,))(H)
            transform_gate = Dense(units=dim)(highway_input)
            transform_gate = Activation(activation)(transform_gate)
            transformed = Multiply()([H, transform_gate])
            carried = Multiply()([carry_gate, highway_input])
            highway_output = Add()([transformed, carried])
        return highway_output
    def call(self,input_data):
        t = conv1dbank(self.K_CBHG, input_data)

        t = MaxPooling1D(pool_size=2, strides=1,
                         padding='same')(t)

        t = Conv1D(filters=128, kernel_size=3,
                   strides=1, padding='same')(t)
        t = BatchNormalization()(t)
        t = Activation('relu')(t)
        t = Conv1D(filters=128, kernel_size=3,
                   strides=1, padding='same')(t)
        t = BatchNormalization()(t)

        residual = Add()([input_data, t])

        highway_net = get_highway_output(residual, 4, activation='relu')

        CBHG_encoder = Bidirectional(GRU(128, return_sequences=True))(highway_net)

        return CBHG_encoder
















class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, config):
        super(Encoder, self).__init__()
        self.embedding_dim = config.embedding_hidden_size
        self.vocab_size = vocab_size
        self.nb_char_max=config.nb_char_max
        #self.k1=config.k1

        # 定义嵌入层
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                   output_dim=self.embedding_dim,
                                                   input_length=self.nb_char_max
                                                   )
        # 定义pre-net
        self.prenet = Prenet(config)
        # 定义CBHG_encoder
        self.CBHG_encoder=CBHG_encoder(config)

    def call(self,input_data):
        embedded = self.embedding(input_data)
        prenet_encoding = self.prenet(embedded)

        cbhg_encoding = self.CBHG_encoder(prenet_encoding)
        return cbhg_encoding



class Decoder(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.mel_time_length = 1#config.MAX_MEL_TIME_LENGTH
        self.mag_time_length = 4#config.MAX_MAG_TIME_LENGTH
        self.n_mels = config.n_mels
        self.r = config.r
        self.k2 = config.k2
        self.n_fft = config.N_FFT
        self.prenet = Prenet(config)
        self.nb_char_max=config.nb_char_max

    def call(self, decode_mel_gts, cbhg_encoding):
        #Attention

        prenet_decoding = self.prenet(decode_mel_gts)


        attention_rnn_output = get_attention_RNN()(prenet_decoding)

        attention_rnn_output_repeated=tf.keras.layers.RepeatVector(self.nb_char_max)(attention_rnn_output)

        #attention_rnn_output = tf.expand_dims(attention_rnn_output,axis=1)


        attention_context = get_attention_context(cbhg_encoding,
                                                  attention_rnn_output_repeated)
        context_shape1 = int(attention_context.shape[1])
        context_shape2 = int(attention_context.shape[2])

        attention_rnn_output_reshaped = Reshape((context_shape1,
                                                 context_shape2))(attention_rnn_output)
        #decode
        input_of_decoder_rnn = concatenate(
            [attention_context, attention_rnn_output_reshaped])  # 连接上下文和状态
        input_of_decoder_rnn_projected = Dense(256)(input_of_decoder_rnn)

        output_of_decoder_rnn = get_decoder_RNN_output(
            input_of_decoder_rnn_projected)

        # mel_hat=TimeDistributed(Dense(n_mels*r))(output_of_decoder_rnn)
        mel_hat = Dense(self.mel_time_length * self.n_mels * self.r)(output_of_decoder_rnn)
        mel_output = Reshape((self.mel_time_length, self.n_mels * self.r))(mel_hat)  # mel谱

        def slice(x):
            return x[:, :, -self.n_mels:]

        mel_hat_last_frame = Lambda(slice)(mel_output)  # 最后一帧的mel谱
        # 从mel谱映映射到幅度谱
        post_process_output = post_process_CBHG(mel_hat_last_frame,
                                                self.k2)
        mags = Dense(self.mag_time_length * (1 + self.n_fft // 2))(post_process_output)
        mag_output = Reshape((self.mag_time_length, (1 + self.n_fft // 2)))(mags)  # 幅度谱
        return mag_output,mel_hat_last_frame,mel_output


class tacotron1(tf.keras.Model):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.encoder=Encoder(vocab_size, config)
        self.decoder=Decoder(config)
    def call(self,decode_mel_gts,input_data):
        encoder_output=self.encoder(input_data)
        mag_output,mel_hat_last_frame,mel_output=self.decoder(decode_mel_gts,encoder_output)
        return mag_output,mel_hat_last_frame,mel_output













#
# # TODO: 解码器应逐帧解码。解码器每步输入每帧mel谱，产生下一帧mel谱和幅度谱
# # 编码器输入为文本；解码器输入为mel谱；模型输出mel谱和幅度谱
# def get_tacotron_model(n_mels, r, k1, k2, nb_char_max,
#                        embedding_size, mel_time_length,
#                        mag_time_length, n_fft,
#                        vocabulary):
#     # Encoder:
#     enc_input_text = Input(shape=(nb_char_max,))  # 输入字符
#
#     embedded = Embedding(input_dim=len(vocabulary),
#                          output_dim=embedding_size,
#                          input_length=nb_char_max)(enc_input_text)
#     prenet_encoding = pre_net(embedded)
#
#     cbhg_encoding = encode_CBHG(prenet_encoding,
#                                 k1)
#
#     # Decoder-part1-Prenet:
#     dec_input_mel = Input(shape=(None, n_mels))  # mel谱，80
#     prenet_decoding = pre_net(dec_input_mel)
#
#     # Attention
#     attention_rnn_output = get_attention_RNN()(prenet_decoding)
#     attention_rnn_output_repeated = RepeatVector(
#         nb_char_max)(attention_rnn_output)
#     attention_context = get_attention_context(cbhg_encoding,
#                                               attention_rnn_output_repeated)  # 得到上下文向量
#
#     context_shape1 = int(attention_context.shape[1])
#     context_shape2 = int(attention_context.shape[2])
#     attention_rnn_output_reshaped = Reshape((context_shape1,
#                                              context_shape2))(attention_rnn_output)
#
#     # Decoder-part2:
#     input_of_decoder_rnn = concatenate(
#         [attention_context, attention_rnn_output_reshaped])  # 连接上下文和状态
#     input_of_decoder_rnn_projected = Dense(256)(input_of_decoder_rnn)
#
#     output_of_decoder_rnn = get_decoder_RNN_output(
#         input_of_decoder_rnn_projected)
#
#     # mel_hat=TimeDistributed(Dense(n_mels*r))(output_of_decoder_rnn)
#     mel_hat = Dense(mel_time_length * n_mels * r)(output_of_decoder_rnn)
#     mel_output = Reshape((mel_time_length, n_mels * r))(mel_hat)  # mel谱
#
#     def slice(x):
#         return x[:, :, -n_mels:]
#
#     mel_hat_last_frame = Lambda(slice)(mel_output)  # 最后一帧的mel谱
#     # 从mel谱映映射到幅度谱
#     post_process_output = post_process_CBHG(mel_hat_last_frame,
#                                             k2)
#     mags = Dense(mag_time_length * (1 + n_fft // 2))(post_process_output)
#     mag_output = Reshape((mag_time_length, (1 + n_fft // 2)))(mags)  # 幅度谱
#
#     model = Model(inputs=[enc_input_text, dec_input_mel],
#                   outputs=[mel_output, mag_output])
#     return model
