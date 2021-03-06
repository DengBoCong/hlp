import tensorflow as tf
from hlp.tts.utils.layers import ConvDropBN
from hlp.tts.utils.layers import DecoderPreNet
from hlp.tts.utils.layers import PostNet


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, encoder_conv_filters, encoder_conv_kernel_sizes,
                 encoder_conv_activation, encoder_lstm_units, encoder_conv_dropout_rate,
                 embedding_hidden_size, n_conv_encoder):
        super(Encoder, self).__init__()
        self.num_filters = encoder_conv_filters
        self.kernel_size = encoder_conv_kernel_sizes
        self.encoder_conv_activation = encoder_conv_activation

        self.lstm_unit = encoder_lstm_units

        self.rate = encoder_conv_dropout_rate
        self.vocab_size = vocab_size

        self.embedding_dim = embedding_hidden_size
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, mask_zero=True)

        self.conv_batch_norm = []
        for i in range(n_conv_encoder):
            conv = ConvDropBN(
                filters=self.num_filters,
                kernel_size=self.kernel_size,
                activation=self.encoder_conv_activation,
                dropout_rate=self.rate)
            self.conv_batch_norm.append(conv)

        # 双向LSTM
        self.forward_layer = tf.keras.layers.LSTM(units=self.lstm_unit, return_sequences=True)
        self.backward_layer = tf.keras.layers.LSTM(units=self.lstm_unit, return_sequences=True,
                                                   go_backwards=True)
        self.bi_lstm = tf.keras.layers.Bidirectional(layer=self.forward_layer,
                                                     backward_layer=self.backward_layer)

    def call(self, x):
        x = self.embedding(x)
        for conv in self.conv_batch_norm:
            x = conv(x)
        output = self.bi_lstm(x)
        return output


class LocationLayer(tf.keras.layers.Layer):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim1):
        super(LocationLayer, self).__init__()
        self.location_convolution = tf.keras.layers.Conv1D(
            filters=attention_n_filters,
            kernel_size=attention_kernel_size,
            padding="same",
            use_bias=False,
            name="location_conv")
        self.location_layer = tf.keras.layers.Dense(
            units=attention_dim1, use_bias=False, activation="tanh", name="location_layer")

    def call(self, attention_weights_cat):
        processed_attention = self.location_convolution(attention_weights_cat)
        processed_attention = self.location_layer(processed_attention)
        return processed_attention


class Attention(tf.keras.layers.Layer):
    def __init__(self, attention_dim, attention_filters, attention_kernel):
        super(Attention, self).__init__()
        self.attention_dim = attention_dim
        self.attention_location_n_filters = attention_filters
        self.attention_location_kernel_size = attention_kernel
        self.query_layer = tf.keras.layers.Dense(self.attention_dim, use_bias=False, activation="tanh")
        self.memory_layer = tf.keras.layers.Dense(self.attention_dim, use_bias=False, activation="tanh")
        self.V = tf.keras.layers.Dense(1, use_bias=False)
        self.location_layer = LocationLayer(self.attention_location_n_filters, self.attention_location_kernel_size,
                                            self.attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, memory, attention_weights_cat):
        processed_query = self.query_layer(tf.expand_dims(query, axis=1))
        processed_memory = self.memory_layer(memory)

        attention_weights_cat = tf.transpose(attention_weights_cat, (0, 2, 1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = tf.squeeze(self.V(tf.nn.tanh(processed_query + processed_attention_weights + processed_memory)), -1)
        return energies

    def __call__(self, attention_hidden_state, memory, attention_weights_cat):
        alignment = self.get_alignment_energies(
            attention_hidden_state, memory, attention_weights_cat)
        attention_weights = tf.nn.softmax(alignment, axis=1)
        attention_context = tf.expand_dims(attention_weights, 1)

        attention_context = tf.matmul(attention_context, memory)
        attention_context = tf.squeeze(attention_context, axis=1)
        return attention_context, attention_weights




class Decoder(tf.keras.layers.Layer):
    def __init__(self, attention_dim, attention_filters, attention_kernel, prenet_units,
                 n_prenet_layers, prenet_dropout_rate, n_conv_encoder, n_conv_postnet, postnet_conv_filters,
                 postnet_conv_kernel_sizes,
                 postnet_dropout_rate, postnet_conv_activation, n_mels, attention_rnn_dim, decoder_lstm_dim,
                 embedding_hidden_size,
                 gate_threshold, max_input_length, initial_hidden_size, decoder_lstm_rate):
        super(Decoder, self).__init__()
        self.attention_dim = attention_dim
        self.attention_rnn_dim = attention_rnn_dim
        self.decoder_lstm_dim = decoder_lstm_dim
        self.embedding_hidden_size = embedding_hidden_size
        self.gate_threshold = gate_threshold
        self.n_mels = n_mels
        self.max_input_length = max_input_length
        self.initial_hidden_size = initial_hidden_size
        self.prenet2 = DecoderPreNet(prenet_units, n_prenet_layers, prenet_dropout_rate)
        self.postnet = PostNet(n_conv_encoder, n_conv_postnet, postnet_conv_filters, postnet_conv_kernel_sizes,
                               postnet_dropout_rate, postnet_conv_activation, n_mels)

        # 两个单层LSTM
        self.decoder_lstms1 = tf.keras.layers.LSTMCell(self.decoder_lstm_dim, dropout=decoder_lstm_rate)
        self.decoder_lstms2 = tf.keras.layers.LSTMCell(self.decoder_lstm_dim, dropout=decoder_lstm_rate)

        # 线性变换投影成目标帧
        self.frame_projection = tf.keras.layers.Dense(
            units=self.n_mels, activation=None, name="frame_projection"
        )

        # 停止记号
        self.stop_projection = tf.keras.layers.Dense(
            units=1, activation='sigmoid', name="stop_projection"
        )

        # 用于注意力
        self.attention_layer = Attention(attention_dim, attention_filters, attention_kernel)

    def get_go_frame(self, memory):
        """ 用于第一步解码器输入
        参数
        ------
        memory: 解码器输出
        返回
        -------
        decoder_input: 全0张量
        """
        B = tf.shape(memory)[0]
        decoder_input = tf.zeros(shape=[B, self.n_mels], dtype=tf.float32)
        return decoder_input

    def initialize_decoder_states(self, memory):
        """ 初始化注意力rnn状态，解码器rnn状态，注意权重，注意力累积权重，注意力上下文，存储记忆
        并存储处理过的内存
        参数
        ------
        memory: 编码器输出
        """
        B = tf.shape(memory)[0]
        MAX_TIME = tf.shape(memory)[1]

        self.attention_hidden = tf.zeros(shape=[B, self.attention_rnn_dim], dtype=tf.float32)

        self.attention_cell = tf.zeros(shape=[B, self.attention_rnn_dim], dtype=tf.float32)

        self.decoder_hidden = tf.zeros(shape=[B, self.decoder_lstm_dim], dtype=tf.float32)

        self.decoder_cell = tf.zeros(shape=[B, self.decoder_lstm_dim], dtype=tf.float32)

        self.attention_weights = tf.zeros(shape=[B, MAX_TIME], dtype=tf.float32)
        self.attention_weights_cum = tf.zeros(shape=[B, MAX_TIME], dtype=tf.float32)
        self.attention_context = tf.zeros(shape=[B, self.initial_hidden_size], dtype=tf.float32)

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)

    def parse_decoder_inputs(self, decoder_inputs):
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = tf.transpose(decoder_inputs, (0, 2, 1))
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = tf.transpose(decoder_inputs, (1, 0, 2))
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        # (T_out, B) -> (B, T_out)
        alignments = tf.stack(alignments)
        alignments = tf.transpose(alignments, [1, 0, 2])
        # (T_out, B) -> (B, T_out)
        gate_outputs = tf.stack(gate_outputs)
        gate_outputs = tf.transpose(gate_outputs, [1, 0])
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = tf.stack(mel_outputs)
        mel_outputs = tf.transpose(mel_outputs, [1, 0, 2])
        mel_outputs = tf.reshape(mel_outputs, (mel_outputs.shape[0], -1, self.n_mels))
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = tf.transpose(mel_outputs, [0, 2, 1])
        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """    参数
               ------
               decoder_input: 先前的mel output
               -------
               """
        # 拼接
        cell_input = tf.concat((decoder_input, self.attention_context), -1)

        # 第一次过lstmcell
        cell_output, (self.attention_hidden, self.attention_cell) = self.decoder_lstms1(cell_input, (
            self.attention_hidden, self.attention_cell))

        # dropout
        self.attention_hidden = tf.keras.layers.Dropout(rate=0.1)(self.attention_hidden)

        # 拼接
        attention_weights_cat = tf.concat(
            ((tf.expand_dims(self.attention_weights, axis=1)), (tf.expand_dims(self.attention_weights_cum, axis=1))),
            axis=1)

        # 注意力
        self.attention_context, self.attention_weights = self.attention_layer(self.attention_hidden, self.memory,
                                                                              attention_weights_cat)
        self.attention_weights_cum += self.attention_weights

        # 拼接
        decoder_input = tf.concat((self.attention_hidden, self.attention_context), -1)

        # 第2次lstmcell
        decoder_output, (self.decoder_hidden, self.decoder_cell) = self.decoder_lstms2(decoder_input, (
            self.decoder_hidden, self.decoder_cell))

        # dropout
        self.decoder_hidden = tf.keras.layers.Dropout(rate=0.1)(self.decoder_hidden)

        # 拼接
        decoder_hidden_attention_context = tf.concat((self.decoder_hidden, self.attention_context), axis=1)

        # 投影梅尔频谱
        decoder_output = self.frame_projection(decoder_hidden_attention_context)

        # 投影stop_token
        gate_prediction = self.stop_projection(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def call(self, memory, decoder_inputs):
        """    参数
               memory: 编码器输出
               decoder_inputs: #用于教师强制
               """
        # go_frame
        decoder_input = self.get_go_frame(memory)
        decoder_input = tf.expand_dims((decoder_input), axis=0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = tf.concat((decoder_input, decoder_inputs), axis=0)
        decoder_inputs = self.prenet2(decoder_inputs)
        self.initialize_decoder_states(memory)
        mel_outputs, gate_outputs, alignments = [], [], []
        # 教师强制
        while len(mel_outputs) < decoder_inputs.shape[0] - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(decoder_input)
            # 拼接
            mel_outputs += [tf.squeeze(mel_output)]
            gate_outputs += [tf.squeeze(gate_output, axis=1)]
            alignments += [attention_weights]
            # 调整维度输出
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)
        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """    参数
               memory: 编码器输出
        """
        # go frame
        decoder_input = self.get_go_frame(memory)
        self.initialize_decoder_states(memory)
        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < self.max_input_length:
            # 通过pre_net
            decoder_input = self.prenet2(decoder_input)
            # 解码
            mel_output, gate_output, attention_weights = self.decode(decoder_input)
            # 拼接
            mel_outputs += [tf.squeeze(mel_output)]
            gate_outputs += [tf.squeeze(gate_output, axis=1)]
            alignments += [attention_weights]
            # 将自己预测的作为下一步的输入
            decoder_input = mel_output
        # 拓展维度
        mel_outputs = tf.expand_dims(mel_outputs, axis=1)
        # 变维度输出
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)
        return mel_outputs, gate_outputs, alignments


class Tacotron2(tf.keras.Model):
    def __init__(self, vocab_size, encoder_conv_filters, encoder_conv_kernel_sizes,
                 encoder_conv_activation, encoder_lstm_units, encoder_conv_dropout_rate,
                 embedding_hidden_size, n_conv_encoder, attention_dim, attention_filters, attention_kernel,
                 prenet_units,
                 n_prenet_layers, prenet_dropout_rate, n_conv_postnet, postnet_conv_filters,
                 postnet_conv_kernel_sizes,
                 postnet_dropout_rate, postnet_conv_activation, n_mels, attention_rnn_dim, decoder_lstm_dim,
                 gate_threshold, max_input_length, initial_hidden_size, decoder_lstm_rate):
        super(Tacotron2, self).__init__()
        self.encoder = Encoder(vocab_size, encoder_conv_filters, encoder_conv_kernel_sizes,
                               encoder_conv_activation, encoder_lstm_units, encoder_conv_dropout_rate,
                               embedding_hidden_size, n_conv_encoder)
        self.decoder = Decoder(attention_dim, attention_filters, attention_kernel, prenet_units,
                               n_prenet_layers, prenet_dropout_rate, n_conv_encoder, n_conv_postnet,
                               postnet_conv_filters,
                               postnet_conv_kernel_sizes,
                               postnet_dropout_rate, postnet_conv_activation, n_mels, attention_rnn_dim,
                               decoder_lstm_dim,
                               embedding_hidden_size,
                               gate_threshold, max_input_length, initial_hidden_size, decoder_lstm_rate)
        self.postnet = PostNet(n_conv_encoder, n_conv_postnet, postnet_conv_filters, postnet_conv_kernel_sizes,
                               postnet_dropout_rate, postnet_conv_activation, n_mels)

    def call(self, inputs, mel_gts):
        encoder_outputs = self.encoder(inputs)
        mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, mel_gts)
        # 后处理网络
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments

    def inference(self, inputs):
        encoder_outputs = self.encoder(inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(encoder_outputs)
        # 后处理网络
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments
