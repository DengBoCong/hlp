

"""----------------------model-------------------------------"""
def configs_model():
    configs_model={}
    #模型目前使用的是1D卷积和双向gru
    configs_model["conv_layers"] = 1 #模型1D卷积层的层数1-3
    configs_model["conv_filters"] = 256 #模型卷积层的过滤器数
    configs_model["conv_kernel_size"] = 11 #模型卷积核的大小
    configs_model["conv_strides"] = 2 #卷积核移动步数
    configs_model["bi_gru_layers"] = 1 #模型双向gru层数1-3
    configs_model["gru_units"] = 256 #gru的单元数
    return configs_model

"""----------------------train-------------------------------"""
def configs_train():
    configs_train={}
    configs_train["train_epochs"] = 10 #训练轮数
    configs_train["data_path"] = "./data/train_data" #训练数据文件夹路径
    configs_train["batch_size"] = 1 #训练数据的batch_size
    configs_train["num_examples"] = None #None指所有数据，整形则对应相应数量
    return configs_train

"""----------------------test--------------------------------"""
def configs_test():
    configs_test={}
    configs_test["data_path"] = "./data/test_data"
    configs_test["batch_size"] = 24 #测试数据集的batch_size
    configs_test["num_examples"] = None #None指所有数据，整形则对应相应数量
    return configs_test

"""----------------------checkpoint--------------------------"""
def configs_checkpoint():
    configs_checkpoint={}
    configs_checkpoint["directory"] = "./checkpoint" #检查点保存路径
    configs_checkpoint["max_to_keep"] = 5 #保存的最近的检查点的最多个数
    configs_checkpoint["save_interval"] = 2
    return configs_checkpoint

"""----------------------record------------------------------"""
def configs_record():
    configs_record={}
    configs_record["record_path"] = "./record/record.wav" #说话人的录音文件路径
    configs_record["record_times"] = 4 #录音时长
    return configs_record

"""----------------------other-------------------------------"""
def configs_other():
    configs_other={}
    configs_other["n_mfcc"] = 20 #声学模型里提取的mfcc特征数
    configs_other["index_word_json_path"] = "index_word_json.json"
    return configs_other

"""---preprocess"""
def configs_preprocess():
    configs_preprocess={}
    configs_preprocess["text_raw_style"] = 1 #文本数据集每行的格式：1是'index string',2是'string',3...
    configs_preprocess["text_process_mode"] = "en_word" #文本处理的方法：en_word(英文单词)或者cn(中文或不带中午的单字符)、en_char(英文字符)
    return configs_preprocess

if __name__ == "__main__":
    pass