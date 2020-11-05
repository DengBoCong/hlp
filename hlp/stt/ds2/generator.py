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
        
        # generator只能进行一次生成，故需要while True来进行多个epoch的数据生成
        while True:
            # 每epoch将所有数据进行一次shuffle
            order = np.random.choice(len(audio_data_path_list), len(audio_data_path_list), replace=False)
            audio_data_path_list = [audio_data_path_list[i] for i in order]
            text_int_sequences = [text_int_sequences[i] for i in order]
            label_length_list = [label_length_list[i] for i in order]

            for idx in range(batchs):
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
    
    elif train_or_test == "test":
        audio_data_path_list, text_list = data
        
        for idx in range(batchs):
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
