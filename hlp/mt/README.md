# 使用说明

- 运行入口： 
    - nmt_train: 使用 ./data/corpus/en-zh_train.txt 进行训练
    - nmt_evaluate : 使用 ./data/corpus/en-zh_eval.txt 对模型进行测试,需对指标类型进行选择
        - bleu指标
    - nmt_translate : 对输入句子进行翻译，输入0退出
    - language_model : 对语言模型进行训练
    
- 分词方法(在配置文件中进行选择)：
    - 英文分词:
        - BPE:使用子词分词器（subwords tokenizer）对英文进行BPE(Byte Pair Encoding)分词
        - WORD:使用单词划分的方法进行分词
    - 中文分词：
        - CHAR:使用字划分方法进行分词
        - WORD:使用jieba模块对词进行划分来分词

- 配置文件参数提示：
    - 验证集部分：
        - 若想从训练数据中划分验证集，则validate_from_txt设置为False，并设置训练集比例train_size
        - 若想从 ./data/en-zh_val.txt 中读取验证集，则validate_from_txt设置为True，并设置验证集句子数num_validate_sentences


# 数据集

github上传为测试数据集