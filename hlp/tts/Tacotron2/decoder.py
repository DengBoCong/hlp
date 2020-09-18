class Decoder(tf.keras.Model):
    def __init__(self,dec_units,batch_sz,target_unit,rate):
        # self.dec_units=1024
        super(Decoder, self).__init__()
        self.target_unit=target_unit
        self.batch_sz = batch_sz
        self.rate = rate
        self.dec_units = dec_units

        # pre-net
        self.df1=tf.keras.layers.Dense(256,activation='relu')
        self.df2=tf.keras.layers.Dense(256,activation='relu')#pre-net
        self.attention= BahdanauAttention(self.dec_units)
        self.lstm1 = tf.keras.layers.LSTM(self.dec_units,return_sequences=True,
                                   return_state=True,recurrent_initializer='glorot_uniform')
        self.lstm2 = tf.keras.layers.LSTM(self.dec_units,return_sequences=True,
                                   return_state=True,recurrent_initializer='glorot_uniform')
        #线性变换投影成目标帧
        self.frame_projection = tf.keras.layers.Dense(units=self.target_unit,
                                                      activation='none',name="frame_projection")
        #post-net
        self.conv1d1=tf.layers.conv1d(num_filters, kernel_size,activation='tanh')
        self.dropout1 = tf.keras.layers.Dropout(self.rate)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv1d2=tf.layers.conv1d(num_filters, kernel_size,activation='tanh')
        self.dropout2 = tf.keras.layers.Dropout(self.rate)
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.conv1d3=tf.layers.conv1d(num_filters, kernel_size,activation='tanh')
        self.dropout3 = tf.keras.layers.Dropout(self.rate)
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.conv1d4=tf.layers.conv1d(num_filters, kernel_size,activation='tanh')
        self.dropout4 = tf.keras.layers.Dropout(self.rate)
        self.bn4 = tf.keras.layers.BatchNormalization()

        self.conv1d5=tf.layers.conv1d(num_filters, kernel_size,activation='none')
        self.dropout5 = tf.keras.layers.Dropout(self.rate)
        self.bn5 = tf.keras.layers.BatchNormalization()

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        #经过pre-net
        x=self.df1(x)
        x=self.df2(x)
        #将pre-net与context_vector拼接
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # 将合并后的向量传送到两层LSTM
        output= self.lstm1(x)
        output= self.lstm2(output)
        # 将LSTM输出的向量与context_vector拼接
        x = tf.concat([tf.expand_dims(context_vector, 1), output], axis=-1)
        #输入到线性变换中去，预测目标谱帧
        x1=self.frame_projection(x)
        x=x1
        #目标谱帧经过post-net来预测残差
        x=self.conv1d1(x)
        x=self.deopout1(x)
        x=self.bn1(x)

        x=self.conv1d2(x)
        x=self.deopout2(x)
        x=self.bn2(x)

        x=self.conv1d3(x)
        x=self.deopout3(x)
        x=self.bn3(x)

        x=self.conv1d4(x)
        x=self.deopout4(x)
        x=self.bn4(x)

        x=self.conv1d5(x)
        x=self.deopout5(x)
        x=self.bn5(x)
        #将残差叠加到原来的目标谱帧上
        x = tf.concat([tf.expand_dims(x1, 1), x], axis=-1)
    return x