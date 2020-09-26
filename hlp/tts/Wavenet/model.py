# -*- coding:utf-8 -*-
# @Time: 2020/9/19 19:25
# @Author:XuMengting
# @Email:1871489154@qq.com
from keras.utils import to_categorical
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv1D, Activation, Multiply, Add
from tensorflow.keras.models import Model
from hlp.tts.wavenet.audio import batch_outputs, batch_inputs
import numpy as np


causal_kernel = 2
causal_channels = 32
dilation_rate = 2
quantization_channels = 256
dilation_channels = 64
causal_channels = 32
dilation_kernel = 2
#残差块
def residual_block(X, dilation_rate):
    F = Conv1D(dilation_channels, dilation_kernel, padding='causal', dilation_rate=dilation_rate)(X)
    G = Conv1D(dilation_channels, dilation_kernel, padding='causal', dilation_rate=dilation_rate)(X)
    F = Activation('tanh')(F)
    G = Activation('sigmoid')(G)
    Y = Multiply()([F, G])
    return Y

#构建训练集和测试集数据，分为（输入和输出），输出作为输入的标签
X_=batch_inputs
Y_=batch_outputs

#创建模型
def model(input_shape):
    X_input = Input(input_shape)
    X = Conv1D(causal_channels, causal_kernel, padding='causal', dilation_rate=1)(X_input)
    Y = residual_block(X, 1)
    S0 = Conv1D(dilation_channels, 1, padding="same")(Y)
    Y = Conv1D(causal_channels, 1, padding="same")(Y)
    X = Add()([X, Y])
    Y = residual_block(X, 2)
    S1 = Conv1D(dilation_channels, 1, padding="same")(Y)
    Y = Conv1D(causal_channels, 1, padding="same")(Y)
    X = Add()([X, Y])
    Y = residual_block(X, 4)
    S2 = Conv1D(dilation_channels, 1, padding="same")(Y)
    Y = Conv1D(causal_channels, 1, padding="same")(Y)
    X = Add()([X, Y])
    Y = residual_block(X, 8)
    S3 = Conv1D(dilation_channels, 1, padding="same")(Y)
    Y = Conv1D(causal_channels, 1, padding="same")(Y)
    X = Add()([X, Y])
    Y = residual_block(X, 16)
    S4 = Conv1D(dilation_channels, 1, padding="same")(Y)
    Y = Conv1D(causal_channels, 1, padding="same")(Y)
    X = Add()([X, Y])
    Y = residual_block(X, 32)
    S5 = Conv1D(dilation_channels, 1, padding="same")(Y)
    Y = Conv1D(causal_channels, 1, padding="same")(Y)
    X = Add()([X, Y])
    Y = residual_block(X, 64)
    S6 = Conv1D(dilation_channels, 1, padding="same")(Y)
    Y = Conv1D(causal_channels, 1, padding="same")(Y)
    X = Add()([X, Y])
    Y = residual_block(X, 128)
    S7 = Conv1D(dilation_channels, 1, padding="same")(Y)
    Y = Conv1D(causal_channels, 1, padding="same")(Y)
    X = Add()([X, Y])
    Y = residual_block(X, 256)
    S8 = Conv1D(dilation_channels, 1, padding="same")(Y)
    Y = Conv1D(causal_channels, 1, padding="same")(Y)
    X = Add()([X, Y])
    Y = residual_block(X, 512)
    S9 = Conv1D(dilation_channels, 1, padding="same")(Y)
    S = Add()([S0, S1, S2, S3, S4, S5, S6, S7, S8, S9])
    X = Activation('relu')(S)
    X = Conv1D(128, 1, padding="same")(X)
    X = Activation('relu')(X)
    X = Conv1D(256, 1, padding="same")(X)
    X = Activation('softmax')(X)
    return Model(inputs = X_input, outputs = X)
net = model((6000, 256))
print(net.input, net.output)
#打印模型
net.summary()
#配置模型
net.compile(optimizer = "adam", loss = "binary_crossentropy", metrics=["accuracy"])
#训练模型
X_ = to_categorical(X_)
Y_ = to_categorical(Y_)
print(X_.shape)
print(Y_.shape)
net.fit(x = X_, y = Y_, epochs =2, batch_size = 2)



#评估模型


#预测模型



#保存训练参数和权重


#保存训练模型结构
