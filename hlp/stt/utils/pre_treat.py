def preprocess_thchs30_speech_raw_data(data_path: str):
    """
    用于处理thchs30数据集的方法，将数据整理为<音频地址, 句子>的
    形式，这样方便后续进行分批读取
    """