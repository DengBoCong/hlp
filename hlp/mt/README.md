# 机器翻译

- 运行入口： 
    - nmt_train: 使用 ./data/corpus/en-zh_train.txt 进行训练
    - nmt_evaluate : 使用 ./data/corpus/en-zh_eval.txt 对模型进行测试,需对指标类型进行选择
        - bleu指标
    - nmt_translate : 对输入句子进行翻译，输入0退出

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

# 版本日志
- 2020.11.16：version 1.5.1
    - 训练完成根据history绘制图像并保存至./data/result
    - 可使用./data/en-zh_val.txt中的数据集来进行训练时的验证
    
- 2020.11.09：version 1.5.0
    - 检查点保存最大数量及保存频率可在配置文件设置
    - 训练过程增加validation数据验证（可在配置文件设置验证频率及验证集大小划分比例）
    - 支持stop-early（通过监控val_accuracy的值）
    - 训练过程保存并返回history数据，包括:loss、accuracy、val_loss、val_accuracy
    
- 2020.11.03：version 1.4.0
    - 可设置库中任意两语言对互译（可选：en、zh）
    - 模型结构优化（改为三个入口、模块增删）
    - 代码优化（变量、方法命名）
    
- 2020.10.23：version 1.3.2
    - 增加从磁盘逐个获得batch进行训练功能
    - 修复优化代码（文本加载、打印分词类型、检查点保存）
    
- 2020.10.17：version 1.3.1
    - 将编码后的数据集保存文本，以优化内存使用
    - 训练打印标准化
    - 更改模型配置
    
- 2020.10.13：version 1.3.0
    - 加入Beam search
    - 训练打印标准化
    - 增加命令行帮助信息
    
- 2020.10.06：version 1.2.3
    - 评价与交互翻译时只加载字典而不加载处理数据集
    
- 2020.10.02：version 1.2.2
    - 评价与交互翻译时只加载字典而不加载处理数据集
    - 命令行参数优化

- 2020.09.28：version 1.2.1
    - 代码优化、消除所有全局变量
    
- 2020.09.26：version 1.2.0
    - 英文分词加入了BPE选项

- 2020.09.22：version 1.1.0
    - 对代码结构进行了更进一步的重构
    - 添加了必要的注释，减少了重复代码
    
- 2020.09.18：version 1.0.0
    - 对代码的结构进行了初步的重构
    - 修复了上一版本BLEU在各n-gram都为0时计算出错的问题
    - 下一步考虑对代码结构进一步调整，并添加Beam search


    
# 运行效果

使用BLEU指标进行评估

![](./image/test_eval_bleu.png)


交互式翻译

![](./image/test_translate.png)

# 数据集

github上传为测试数据集

若需完整数据集请自行下载：

链接：https://pan.baidu.com/s/1cXPHJXFtMLJSJv-_Qn9cQA
提取码：mt66 
复制这段内容后打开百度网盘手机App，操作更方便哦