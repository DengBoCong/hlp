# 目录
+ [运行说明](#运行说明)
+ [参数配置说明](#参数配置说明)
+ [模型效果](#模型效果)


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

# 参数配置说明(模型参数、音频特征(mfcc、fbank)、text_process_mode文本切分方式(en_char、en_word、cn)等可选)
+ number语料
{
    "train": {
        "train_epochs": 2,
        "data_path": "./data/number/train",
        "batch_size": 36,
        "num_examples": null
    },
    "test": {
        "data_path": "./data/number/dev",
        "batch_size": 36,
        "num_examples": null
    },
    "model": {
        "conv_layers": 1,
        "conv_filters": 256,
        "conv_kernel_size": 11,
        "conv_strides": 2,
        "bi_gru_layers": 1,
        "gru_units": 256,
        "fc_units": 512
    },
    "checkpoint": {
        "directory": "./checkpoints",
        "max_to_keep": 5,
        "save_interval": 2
    },
    "other": {
        "audio_feature_type": "mfcc"
    },
    "preprocess": {
        "dataset_information_path": "./dataset_information.json",
        "dataset_name": "number",
        "text_row_style": 1,
        "text_process_mode": "cn"
    }
}
+ LibriSpeech语料
{
    "train": {
        "train_epochs": 2,
        "data_path": "./data/LibriSpeech/train-clean-5",
        "batch_size": 36,
        "num_examples": null
    },
    "test": {
        "data_path": "./data/LibriSpeech/dev-clean-2",
        "batch_size": 36,
        "num_examples": null
    },
    "model": {
        "conv_layers": 1,
        "conv_filters": 256,
        "conv_kernel_size": 11,
        "conv_strides": 2,
        "bi_gru_layers": 1,
        "gru_units": 256,
        "fc_units": 512
    },
    "checkpoint": {
        "directory": "./checkpoints",
        "max_to_keep": 5,
        "save_interval": 2
    },
    "other": {
        "audio_feature_type": "mfcc"
    },
    "preprocess": {
        "dataset_information_path": "./dataset_information.json",
        "dataset_name": "librispeech",
        "text_row_style": 1,
        "text_process_mode": "en_char"
    }
}
+ thchs30语料
{
    "train": {
        "train_epochs": 2,
        "data_path": "./data/data_thchs30/train",
        "batch_size": 36,
        "num_examples": null
    },
    "test": {
        "data_path": "./data/data_thchs30/test",
        "batch_size": 36,
        "num_examples": null
    },
    "model": {
        "conv_layers": 1,
        "conv_filters": 256,
        "conv_kernel_size": 11,
        "conv_strides": 2,
        "bi_gru_layers": 1,
        "gru_units": 256,
        "fc_units": 512
    },
    "checkpoint": {
        "directory": "./checkpoints",
        "max_to_keep": 5,
        "save_interval": 2
    },
    "other": {
        "audio_feature_type": "mfcc"
    },
    "preprocess": {
        "dataset_information_path": "./dataset_information.json",
        "dataset_name": "thchs30",
        "text_row_style": 1,
        "text_process_mode": "cn"
    }
}

# 模型效果
待完善