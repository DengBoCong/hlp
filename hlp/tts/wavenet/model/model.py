# -*- coding:utf-8 -*-
# @Time: 2020/9/19 19:25
# @Author:XuMengting
# @Email:1871489154@qq.com
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv1D, Activation, Multiply, Add
from tensorflow.keras.models import Model
from hlp.tts.Wavenet.model.batch import random_batch_generator, batch_inputs, batch_outputs

causal_kernel = 2
causal_channels = 32
dilation_rate = 2
quantization_channels = 256
dilation_channels = 64
causal_channels = 32
dilation_kernel = 2


# 残差块
def residual_block(X, dilation_rate):
    F = Conv1D(dilation_channels, dilation_kernel, padding='causal', dilation_rate=dilation_rate)(X)
    G = Conv1D(dilation_channels, dilation_kernel, padding='causal', dilation_rate=dilation_rate)(X)
    F = Activation('tanh')(F)
    G = Activation('sigmoid')(G)
    Y = Multiply()([F, G])
    return Y


# 构建训练集，分为（输入和输出），输出作为输入的标签
X_train = batch_inputs
Y_train = batch_outputs
X_ = to_categorical(X_train, 256)
Y_ = to_categorical(Y_train, 256)
print(X_.shape)
print(Y_.shape)
print(X_train[0].shape)
# 构建测试集数据
X_test, Y_test = random_batch_generator(6000, 10, 160124, r'E:\git\hlp\hlp\tts\wavenet\data\test', True)
X_test = to_categorical(X_test, 256)
Y_test = to_categorical(Y_test, 256)


# 创建模型
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
    return Model(inputs=X_input, outputs=X)


# model实例化
net = model((6000, 256))
print(net.input, net.output)

# 配置模型
net.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# 断点续训
checkpoint_save_path = "./checkpoint/wavenet.ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    net.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')
# 训练模型，这里的batch_size是指一个epoch里面的输入分为几个时间步，总数据为10，batch_size为5，则分为2个时间步
history = net.fit(x=X_, y=Y_, epochs=2, batch_size=5, validation_split=0.2, callbacks=[cp_callback])
# 打印模型
net.summary()
# 保存训练参数和权重
file = open('./weights.txt', 'w')  # 参数提取
for v in net.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()
# loss & acc 可视化
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


# 使用`evaluate'在测试数据上评估模型
print("Evaluate on test data")
results = net.evaluate(X_test, Y_test, batch_size=5,)
print("test loss, test acc:", results)


# 预测模型
# 生成预测（概率-最后一层的输出）
print("Generate predictions for 3 samples")
predictions = net.predict(X_test[:3])
print("predictions shape:", predictions.shape)
print(predictions)
# 对预测数据还原---从（0，1）反归一化到原始范围

# 对真实数据还原---从（0，1）反归一化到原始范围

# 画出真实数据和预测数据的对比曲线
