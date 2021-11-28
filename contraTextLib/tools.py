## 提供基础工具，如模型layer（embedding、encoder），loss计算

## embedding 就用 bertembedding啊，反正也没改什么
## encoder 就用 bertencoder啊，反正也没改什么
import sys
import copy
import math
import torch
import torch.nn as nn
from prettytable import PrettyTable

PATH_TO_SENTEVAL = 'D:/jupyterFile/text-contrastive/SimCSE-main/SimCSE/SentEval'
PATH_TO_DATA = 'D:/jupyterFile/text-contrastive/SimCSE-main/SimCSE/SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class InfoNCE(nn.Module):
    def __init__(self,temp=0.05):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
        self.loss_fct = nn.CrossEntropyLoss()
        
    def forward(self,online_output,target_output):
        target = torch.LongTensor([i for i in range(online_output.shape[0])]).cuda()
        sim_matrix = self.cos(online_output.unsqueeze(1), target_output.unsqueeze(0)) / self.temp
        loss = self.loss_fct(sim_matrix,target)
        return loss


# evaluate model in all STS tasks
def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def evalModel(model,tokenizer, pooler = "cls_before_pooler"): 
    tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    
    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
    params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                'tenacity': 3, 'epoch_size': 2}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def prepare(params, samples):
        return

    def batcher(params, batch, max_length=None):
            # Handle rare token encoding issues in the dataset
            if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
                batch = [[word.decode('utf-8') for word in s] for s in batch]

            sentences = [' '.join(s) for s in batch]
            
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                max_length=max_length,
                truncation=True
            )
            # Move to the correct device
            for k in batch:
                batch[k] = batch[k].to(device)
            
            # Get raw embeddings
            with torch.no_grad():
                pooler_output = model(**batch, output_hidden_states=True, return_dict=True,sent_emb = True)
                if pooler == "cls_before_pooler":
                    pooler_output = pooler_output.last_hidden_state[:, 0]
                elif pooler == "cls_after_pooler":
                    pooler_output = pooler_output.pooler_output

            return pooler_output.cpu()
    results = {}

    for task in tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result
    task_names = []
    scores = []
    for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
        task_names.append(task)
        if task in results:
            if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
            else:
                scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
        else:
            scores.append("0.00")
    task_names.append("Avg.")
    scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
    print_table(task_names, scores)

    return sum([float(score) for score in scores])/len(scores)

class EMA(torch.nn.Module):
    """
    [https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage]
    """

    def __init__(self, model, decay,total_step=15000):
        super().__init__()
        self.decay = decay
        self.total_step = total_step
        self.step = 0
        self.model = copy.deepcopy(model).eval()

    def update(self, model):
        self.step = self.step+1
        decay_new = 1-(1-self.decay)*(math.cos(math.pi*self.step/self.total_step)+1)/2
        # 慢慢把self.model往model移动
        with torch.no_grad():
            e_std = self.model.state_dict().values()
            m_std = model.state_dict().values()
            for e, m in zip(e_std, m_std):
                e.copy_(decay_new * e + (1. - decay_new) * m)

class BYOLMSE(nn.Module):
    def __init__(self,temp=0.05):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
        #self.loss_fct = nn.CrossEntropyLoss()
    def forward(self,online_output,target_output):
        # online_output batch-size * hidden-size 
        # target_output batch-size * hidden-size 
        
        # 实际上就是2-2*cosine_similarity
        out = torch.diag(self.cos(online_output.unsqueeze(1), target_output.unsqueeze(0)))

        return (2-2*out).mean() / self.temp

