from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader
from transformers.trainer import Trainer,TrainingArguments
from datasets import Dataset, load_dataset, load_from_disk
from transformers import (
    BertTokenizer,
)
from tools import *

batch_size=64
wiki_for_sts = "D:/jupyterFile/text-contrastive/data-handling/wiki_for_sts_32_norepeat"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = load_from_disk(wiki_for_sts)
train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last = True)

# override the evaluate method
class SimCSETrainer(Trainer):
    def __init__(self,**paraments):
        super().__init__(**paraments)
        
        self.best_sts = 0.0
        self.best_pool_sts = 0.0

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        eval_senteval_transfer: bool = False,
    ) -> Dict[str, float]:

        # Set params for SentEval (fastmode)
        
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                            'tenacity': 3, 'epoch_size': 2}

        self.model.eval()
        sum_acc = evalModel(self.model,tokenizer, pooler = 'cls_before_pooler')

        if sum_acc > self.best_sts:
            self.best_sts = sum_acc
            self.save_model(self.args.output_dir+"\\best-model")

        self.model.train()
        print('acc before pooler:',sum_acc,'\n max acc ',self.best_sts)
        metrics = {'acc before pooler':sum_acc}
        self.log(metrics)

        return metrics