# 机器翻译

- 运行入口： transformer.py
- 执行方法：
    - 指令格式：python transformer.py -t/--type [执行模式]
    - 可选模式：
        - train:    使用 ./data 下的指定文件 (默认en-ch.txt) 进行训练
        - bleu :    使用 ./data 下的指定文件 (默认 en-ch_evaluate.txt) 进行BLEU指标计算
        - translate : 对输入句子进行翻译，输入exit退出
   

# 版本日志
- 2020.09.18：更新模型
    - 对代码的结构进行了初步的重构
    - 修复了上一版本BLEU在各n-gram都为0时计算出错的问题
    - 下一步考虑对代码结构进一步调整，并添加Beam search