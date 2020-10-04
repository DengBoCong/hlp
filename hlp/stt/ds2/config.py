import char_index_map

"""----------------------model-------------------------------"""
configs_model={}
#模型目前使用的是1D卷积和双向gru
configs_model["conv_layers"] = 1 #模型1D卷积层的层数1-3
configs_model["conv_filters"] = 256 #模型卷积层的过滤器数
configs_model["conv_kernel_size"] = 11 #模型卷积核的大小
configs_model["conv_strides"] = 2 #卷积核移动步数
configs_model["bi_gru_layers"] = 1 #模型双向gru层数1-3
configs_model["gru_units"] = 256 #gru的单元数
configs_model["dense_units"] = len(char_index_map.index_map)+2

"""----------------------train-------------------------------"""
configs_train={}
configs_train["is_first_train"] = False #是否初次训练，否则加载保存的checkpoint
configs_train["train_epochs"] = 100 #训练轮数
configs_train["data_path"] = "./data/train_data" #训练数据文件夹路径
configs_train["batch_size"] = 36 #训练数据的batch_size

"""----------------------test--------------------------------"""
configs_test={}
configs_test["data_path"] = "./data/test_data"
configs_test["batch_size"] = 24 #测试数据集的batch_size

"""----------------------checkpoint--------------------------"""
configs_checkpoint={}
configs_checkpoint["directory"] = "./checkpoint" #检查点保存路径
configs_checkpoint["max_to_keep"] = 5 #保存的最近的检查点的最多个数

"""----------------------other-------------------------------"""
configs_other={}
configs_other["n_mfcc"] = 20 #声学模型里提取的mfcc特征数
configs_other["record_path"] = "./record/record.wav" #说话人的录音文件路径
