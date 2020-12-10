# 目录
+ [运行说明](#运行说明)
+ [SMN检索部署说明](#SMN检索部署说明)


# 运行说明
+ 运行入口：
   + seq2seq_chatter.py为seq2seq的执行入口文件：指令需要附带运行参数
   + transformer_chatter.py为transformer的执行入口文件：指令需要附带运行参数
   + smn_chatter.py为smn的执行入口文件：指令需要附带运行参数

详细指令参数见各chatter文件
**注意：程序同时支持json配置文件和命令行参数执行，接送配置文件的使用，只需要使**
**用--config_file参数进行指定配置文件即可，如果指定了配置文件，则以配置文件中的配置优先**

+ 正常执行顺序为pre_treat->train->evaluate->chat

# SMN检索部署说明
**工具版本如下：**
+ [Solr：8.6.3](https://lucene.apache.org/)
+ [pysolr：3.9.0](https://pypi.org/project/pysolr/)
+ python：3.7
+ CentOS：7.6
+ Docker：19.03.9

**整体流程说明**

以SMN模型为例，利用启发式方法从索引中获取候选response，将前一轮的utterances（也就是对话的历史）和 u 进行计算，根据他们的tf-idf得分，从 utterances 中提取前 5 个关键字，然后将扩展后的message用于索引，并使用索引的内联检索算法来检索候选response。模型结构和训练至关重要，但是检索候选回复也是使得整个对话流程实现闭环的关键。

**说明**
Dockerfile从[docker-solr](https://github.com/docker-solr/docker-solr)中拉取，随后执行如下：

```
docker run -itd --name solr -p 8983:8983 solr:8.6.3
# 然后创建core核心选择器，我这里因为以SMN模型讲解，所以取名SMN
# exec -it ：交互式执行容器
# -c  内核的名称（必须）
docker exec -it --user=solr solr bin/solr create_core -c smn
```

+ 配置IK
首先是IK，IK Analyzer(中文分词器)是一个开源的，基于java语言开发的轻量级的中文分词工具包。最初，它是以开源项目 Lucene为应用主体的，结合词典分词和文法分析算法的中文分词组件。新版本的IKAnalyzer3.0则发展为 面向Java的公用分词组件，独立于Lucene项目，同时提供了对Lucene的默认优化实现。[获取IK](https://github.com/magese/ik-analyzer-solr)