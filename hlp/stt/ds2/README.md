
# 运行说明
+ 训练
    + 先通过config.py进行相关参数配置
    + 运行preprocess.py进行数据集的预处理，加载数据集的相关信息
    + 运行train.py
+ 评价
    + 运行evaluate.py
+ 预测(识别)
    + 运行recognize.py
    + 控制台手动输入录音时长

# 配置说明
+ 音频特征(mfcc、fbank)、text_process_mode文本切分方式(en_char、en_word、cn)
