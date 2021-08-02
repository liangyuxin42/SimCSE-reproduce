# SimCSE-reproduce

在jupyter-notebook中复现SimCSE，目前的代码是基于bert-base-uncased的unsupervised方法。

模型基于bert-base-uncased, 复现SimCSE-unsupervised，训练数据集是从wiki sample的1m条句子，验证/测试数据集是senteval data（和原论文/代码一致）；代码基于SimCSE论文给出的代码，为了方便理解在结构上做了一些调整。

# 使用方法

1. 下载所需数据集：

训练数据集： 

1m wiki sentence https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt

验证数据集&代码： 

SentEval：https://github.com/princeton-nlp/SimCSE/tree/main/SentEval

SentEval/data: https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/senteval.tar

2. pre-prepare dataset:

运行dataset-prepare，得到处理好的dataset wiki_for_sts_32

3. 训练 unsupervised SimCSE

运行SimCSETrainer，训练无监督SimCSE

模型训练流程：

sentence -> bert -> (cls token output embedding) -> MLP -> cosine similarity -> loss

模型验证流程：

sentence -> bert -> (cls token output embedding) -> SentEval -> correlation

注意验证的时候是直接取cls token，不经过MLP层，这样的效果会更好。MLP层相当于是对比学习中的projector。

训练结果为Avg 74.96，基本达到了原代码的水准（不同的初始化可能对结果有一定影响）。

```
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| 68.16 | 79.49 | 71.91 | 81.08 | 76.90 |    75.82     |      71.34      | 74.96 |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
```

# 参考

SimCSE论文：https://arxiv.org/abs/2104.08821

SimCSE代码：https://github.com/princeton-nlp/SimCSE
