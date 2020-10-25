import os
from utils import get_config
from load_dataset import load_dataset_number



#加载数据
def load_data(data_path, train_or_test, num_examples):
    configs = get_config()
    dataset_name = configs["preprocess"]["dataset_name"]
    if dataset_name == "number":
        if train_or_test == "train":
            input_tensor, target_tensor, target_length = load_dataset_number(data_path, train_or_test, num_examples)
            return input_tensor, target_tensor, target_length
        else:
            input_tensor, labels_list = load_dataset_number(data_path, train_or_test, num_examples)
            return input_tensor, labels_list


if __name__ == '__main__':
    pass