from util import get_config, set_config
import json
import os


# 重新训练时将config.json初始化
if __name__ == "__main__":
    configs = get_config()

    # 将if_is_first_train改为true
    set_config(configs, "train", "if_is_first_train" , True)

    # 清除checkpoint
    checkpoint_path = configs["checkpoint"]["directory"]
    checkpoint_files = os.listdir(checkpoint_path)
    for checkpoint_file in checkpoint_files:
        os.remove(checkpoint_path + "/" + checkpoint_file)
    
    # 清除dataset_information
    os.remove(configs["preprocess"]["dataset_information_path"])

    # 清除录音文件
    os.remove(configs["record"]["record_path"])