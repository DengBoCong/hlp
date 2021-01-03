import os
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# TODO: 指标名不在代码中指定
def show_and_save_history(history, save_dir, valid_freq=1):
    """
    用于显示历史指标趋势以及保存历史指标图表图
    :param history: 历史指标字典
    :param save_dir: 历史指标显示图片保存位置
    :param valid_freq: 验证频率
    :return: 无返回值
    """
    train_x_axis = [i + 1 for i in range(len(history['loss']))]
    valid_x_axis = [(i + 1) * valid_freq for i in range(len(history['val_loss']))]

    figure, axis = plt.subplots(1, 1)
    tick_spacing = 1
    if len(history['loss']) > 20:
        tick_spacing = len(history['loss']) // 20

    plt.plot(train_x_axis, history['loss'], label='loss', marker='.')
    plt.plot(train_x_axis, history['accuracy'], label='accuracy', marker='.')
    plt.plot(valid_x_axis, history['val_loss'], label='val_loss', marker='.', linestyle='--')
    plt.plot(valid_x_axis, history['val_accuracy'], label='val_accuracy', marker='.', linestyle='--')

    plt.xticks(valid_x_axis)
    plt.xlabel('epoch')
    plt.legend()
    axis.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    save_path = save_dir + time.strftime("%Y_%m_%d_%H_%M_%S_", time.localtime(time.time()))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)
    plt.show()
