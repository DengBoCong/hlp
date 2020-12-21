# 使用说明
项目自带较小的训练和验证数据集，可以无需配置运行所有功能。
- 语料预处理
   - （可选步骤）在mt/config/config.json中配置语料路径和切分方法
   - 运行mt/preprocess.py

- 训练模型
   - （可选步骤）在mt/config/config.json中配置语料路径、切分方法、模型参数和训练超参数等
   - 运行mt/train.py

- 评价模型
   - （可选步骤）在mt/config/config.json中配置验证语料路径
   - 运行mt/evaluate.py

- 交互式翻译
   - 运行mt/translate.py