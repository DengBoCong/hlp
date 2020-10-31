from utils import get_config
from audio_process import get_input_tensor
from text_process import get_text_label
import numpy as np
import tensorflow as tf

# 数据生成器
def data_generator(data, train_or_test, batchs, batch_size):
    configs = get_config()

    if train_or_test == "train":
        audio_data_path_list, text_int_sequences, label_length_list = data
        
        while True:
            # 基于batchs随机构建一个处理顺序序列
            order = np.random.choice(batchs, batchs, replace=False)
            for idx in order:
                batch_input_tensor = get_input_tensor(
                    audio_data_path_list[idx*batch_size : (idx+1)*batch_size],
                    configs["other"]["n_mfcc"],
                    configs["preprocess"]["max_input_length"]
                    )
                batch_label_tensor = get_text_label(
                    text_int_sequences[idx*batch_size : (idx+1)*batch_size],
                    configs["preprocess"]["max_label_length"]
                )
                batch_label_length = tf.convert_to_tensor(label_length_list[idx*batch_size : (idx+1)*batch_size])
                
                yield batch_input_tensor, batch_label_tensor, batch_label_length
    
    else:
        audio_data_path_list, text_list = data
        
        while True:
            order = np.random.choice(batchs, batchs, replace=False)
            for idx in order:
                batch_input_tensor = get_input_tensor(
                    audio_data_path_list[idx*batch_size : (idx+1)*batch_size],
                    configs["other"]["n_mfcc"],
                    configs["preprocess"]["max_input_length"]
                    )
                batch_text_list = text_list[idx*batch_size : (idx+1)*batch_size]
                
                #测试集只需要文本串list
                yield batch_input_tensor, batch_text_list



if __name__ == '__main__':
    pass
