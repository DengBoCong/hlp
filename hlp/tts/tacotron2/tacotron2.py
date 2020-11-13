import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 自己的模型
class ConvBatchDrop(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, activation, dropout_rate):
        super(ConvBatchDrop, self).__init__()
        self.conv1d = tf.keras.layers.Conv1D(
            filters,
            kernel_size,
            padding="same",
            activation=activation
        )
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.norm = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        outputs = self.conv1d(inputs)
        outputs = self.dropout(outputs)
        outputs = self.norm(outputs)
        return outputs


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, config):
        super(Encoder, self).__init__()
        self.num_filters = config.encoder_conv_filters

        self.kernel_size = config.encoder_conv_kernel_sizes
        self.lstm_unit = config.encoder_lstm_units
        self.rate = config.encoder_conv_dropout_rate

        self.vocab_size = vocab_size
        self.embedding_dim = config.embedding_hidden_size
        self.encoder_conv_activation = config.encoder_conv_activation
        # 定义嵌入层
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, mask_zero=True)

        # 定义三层卷积层
        self.conv_batch_norm = []
        for i in range(config.n_conv_encoder):
            conv = ConvBatchDrop(
                filters=self.num_filters,
                kernel_size=self.kernel_size,
                activation=self.encoder_conv_activation,
                dropout_rate=self.rate,
            )
            self.conv_batch_norm.append(conv)

        # 定义两次LSTM
        self.forward_layer = tf.keras.layers.LSTM(units=self.lstm_unit, return_sequences=True)
        self.backward_layer = tf.keras.layers.LSTM(units=self.lstm_unit, return_sequences=True,
                                                   go_backwards=True)
        self.bidir = tf.keras.layers.Bidirectional(layer=self.forward_layer,
                                                   backward_layer=self.backward_layer)

    def call(self, x):
        x = self.embedding(x)
        for conv in self.conv_batch_norm:
            x = conv(x)
        output = self.bidir(x)
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
            name="location_conv",
        )
        self.location_layer1 = tf.keras.layers.Dense(
            units=attention_dim1, use_bias=False, activation="tanh", name="location_layer"
        )

    def call(self, attention_weights_cat):
        processed_attention = self.location_convolution(attention_weights_cat)
        processed_attention = self.location_layer1(processed_attention)
        return processed_attention


class Attention(tf.keras.layers.Layer):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.attention_rnn_dim = config.attention_dim
        self.attention_dim = config.attention_dim
        self.attention_location_n_filters = config.attention_filters
        self.attention_location_kernel_size = config.attention_kernel
        self.query_layer = tf.keras.layers.Dense(self.attention_rnn_dim, use_bias=False, activation="tanh")
        self.memory_layer = tf.keras.layers.Dense(self.attention_rnn_dim, use_bias=False, activation="tanh")
        self.V = tf.keras.layers.Dense(1, use_bias=False)
        self.location_layer = LocationLayer(self.attention_location_n_filters, self.attention_location_kernel_size,
                                            self.attention_rnn_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, memory, attention_weights_cat):
        # print("query:", query.shape)
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


# attention结束
class Prenet(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.prenet_units = config.prenet_units
        self.n_prenet_layers = config.n_prenet_layers
        self.prenet_dropout_rate = config.prenet_dropout_rate
        self.prenet_dense = [
            tf.keras.layers.Dense(
                units=self.prenet_units,
                activation='relu'
            )
            for i in range(self.n_prenet_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(
            rate=self.prenet_dropout_rate
        )

    def __call__(self, inputs):
        """Call logic."""
        outputs = inputs
        for layer in self.prenet_dense:
            outputs = layer(outputs)
            outputs = self.dropout(outputs)
        return outputs


class Postnet(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.conv_batch_norm = []
        for i in range(config.n_conv_encoder):
            if i == config.n_conv_postnet-1:
                conv = ConvBatchDrop(
                    filters=config.postnet_conv_filters,
                    kernel_size=config.postnet_conv_kernel_sizes,
                    activation=None,
                    dropout_rate=config.postnet_dropout_rate,
                )
            else:
                conv = ConvBatchDrop(
                    filters=config.postnet_conv_filters,
                    kernel_size=config.postnet_conv_kernel_sizes,
                    activation=config.postnet_conv_activation,
                    dropout_rate=config.postnet_dropout_rate,
                )
            self.conv_batch_norm.append(conv)

        self.fc = tf.keras.layers.Dense(units=config.n_mels, activation=None, name="frame_projection1")

    def call(self, inputs):
        x = tf.transpose(inputs, [0, 2, 1])
        for _, conv in enumerate(self.conv_batch_norm):
            x = conv(x)
        x = self.fc(x)
        x = tf.transpose(x, [0, 2, 1])
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.attention_dim = config.attention_dim
        self.decoder_lstm_dim = config.decoder_lstm_dim
        self.embedding_hidden_size = config.embedding_hidden_size
        self.gate_threshold = config.gate_threshold
        self.max_len = config.max_len
        self.n_mels = config.n_mels
        self.max_len = config.max_len
        self.prenet2 = Prenet(config)
        self.postnet = Postnet(config)
        # 两个单层LSTM
        self.decoder_lstms1 = tf.keras.layers.LSTMCell(self.decoder_lstm_dim, dropout=config.decoder_lstm_rate)
        self.decoder_lstms2 = tf.keras.layers.LSTMCell(self.decoder_lstm_dim, dropout=config.decoder_lstm_rate)
        # 线性变换投影成目标帧
        self.frame_projection = tf.keras.layers.Dense(
            units=self.n_mels, activation=None, name="frame_projection"
        )
        # 停止记号
        self.stop_projection = tf.keras.layers.Dense(
            units=1, activation='sigmoid', name="stop_projection"
        )
        # 用于注意力
        self.attention_layer = Attention(config)

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

        self.attention_hidden = tf.zeros(shape=[B, self.decoder_lstm_dim], dtype=tf.float32)

        self.attention_cell = tf.zeros(shape=[B, self.decoder_lstm_dim], dtype=tf.float32)

        self.decoder_hidden = tf.zeros(shape=[B, self.decoder_lstm_dim], dtype=tf.float32)

        self.decoder_cell = tf.zeros(shape=[B, self.decoder_lstm_dim], dtype=tf.float32)

        self.attention_weights = tf.zeros(shape=[B, MAX_TIME], dtype=tf.float32)
        self.attention_weights_cum = tf.zeros(shape=[B, MAX_TIME], dtype=tf.float32)
        self.attention_context = tf.zeros(shape=[B, self.embedding_hidden_size], dtype=tf.float32)

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
        alignments = tf.transpose(alignments, (1, 0, 2))
        # (T_out, B) -> (B, T_out)
        gate_outputs = tf.stack(gate_outputs)
        gate_outputs = tf.transpose(gate_outputs, (1, 0))
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = tf.stack(mel_outputs)
        mel_outputs = tf.transpose(mel_outputs, (1, 0, 2))
        mel_outputs = tf.reshape(mel_outputs, (mel_outputs.shape[0], -1, self.n_mels))
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = tf.transpose(mel_outputs, (0, 2, 1))
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
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
            # 拼接
            mel_outputs += [tf.squeeze(mel_output)]
            gate_outputs += [tf.squeeze(gate_output, axis=1)]
            alignments += [attention_weights]
            # 调整维度输出
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)
        return mel_outputs, gate_outputs, alignments

    # 预测 我觉得有问题 但是我还没有改好，它一直很让我头疼
    def inference(self, memory):
        """    参数
               memory: 编码器输出
        """
        # go frame
        decoder_input = self.get_go_frame(memory)
        self.initialize_decoder_states(memory)
        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < self.max_len:
            # 通过pre_net
            decoder_input = self.prenet2(decoder_input)
            # 解码
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
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
    def __init__(self, vocab_inp_size, config):
        super(Tacotron2, self).__init__()
        self.encoder = Encoder(vocab_inp_size, config)
        self.decoder = Decoder(config)
        self.postnet = Postnet(config)

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


# 恢复检查点
def load_checkpoint(tacotron2, path, config):
    # 加载检查点
    checkpoint_path = path
    ckpt = tf.train.Checkpoint(tacotron2=tacotron2)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=config.max_to_keep)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    return ckpt_manager
