# -*- coding:utf-8 -*-
# @Time: 2020/9/19 19:25
# @Author:XuMengting
# @Email:1871489154@qq.com
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv1D, Activation, Multiply, Add, Convolution1D, Flatten, Dense
from tensorflow.keras.models import Model
from hlp.tts.wavenet.model.batch import random_batch_generator, batch_inputs, batch_outputs

causal_kernel = 2
causal_channels = 32
dilation_rate = 2
quantization_channels = 256
dilation_channels = 64
causal_channels = 32
dilation_kernel = 2


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


# 输出维度，每个过滤器扩展的时间或空间，
def wavenetBlock(n_atrous_filters, atrous_filter_size, atrous_rate):
    def f(input_):
        residual = input_
        h = Convolution1D(n_atrous_filters, atrous_filter_size, padding='same', dilation_rate=atrous_rate)(input_)
        tanh_out = Activation('tanh')(h)
        s = Convolution1D(n_atrous_filters, atrous_filter_size, padding='same', dilation_rate=atrous_rate)(input_)
        sigmoid_out = Activation('sigmoid')(s)
        merged = Multiply()([tanh_out, sigmoid_out])
        skip_out = Convolution1D(1, 1, activation='relu', padding='same')(merged)
        out = Add()([skip_out, residual])
        return out, skip_out
    return f


# 创建模型
def model(input_shape):
    input_ = Input(input_shape)
    # dilation_rate为1的一层因果卷积
    X = Convolution1D(32, 2, padding='causal', dilation_rate=1)(input_)
    # dilation_rate为2的一层扩张卷积
    A, B = wavenetBlock(64, 2, 2)(X)
    skip_connections = [B]
    # 扩大倍数4,8,16,32,64,128,256,1,2,4,8,16,32,64,128,256,1,2,4,8的20层扩张卷积
    for i in range(20):
        A, B = wavenetBlock(64, 2, 2**((i+2)%9))(A)
        skip_connections.append(B)
    net = Add()(skip_connections)
    net = Activation('relu')(net)
    net = Convolution1D(128, 1, activation='relu')(net)
    net = Convolution1D(256, 1)(net)
    net = Activation('softmax')(net)
    return Model(inputs=input_, outputs=net)


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
