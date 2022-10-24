#!/home/zhouheng/anaconda3/bin/python
# -*- encoding: utf-8 -*-
'''
@文件    :run_mlm_base.py
@时间    :2021/12/02 14:21:25
@作者    :周恒
@版本    :1.0
@说明    :
'''
import argparse
import json
import pickle
import random
import sys
from typing import Any, Dict

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers.models.roberta import RobertaForMaskedLM, RobertaTokenizer
from typing_extensions import Literal

from prompt_tuning import (BaselineCollator, BaselineFinetuningModel,
                           BaselineSampler, baseline_init_tokenizer,
                           baseline_preprocess_data, batch_cal_loss_func,
                           batch_forward_func, batch_metrics_func,
                           get_optimizer, metrics_cal_func,
                           valid_data_preprocess)
from trainer import Trainer


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument("--main_device", type=int, default=0)
parser.add_argument("--device_ids", type=str, default="0,1")
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--num_workers", type=int, default=12)
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--mlm_name_or_path", type=str, default="roberta-base")
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--split_n", type=int, required=True)
parser.add_argument("--gradient_accumulate",type=int,default=1)
if __name__ == '__main__':
    set_random_seed(114514)
    args=parser.parse_args()
    """一些常规的设置"""
    dev = torch.device(3)
    dev=torch.device(args.main_device)
    device_ids=list(map(lambda x:int(x),args.device_ids.split(",")))
    batch_size=args.batch_size
    num_workers=args.num_workers
    learning_rate=args.learning_rate
    epochs=args.epochs
    mlm_type=args.mlm_name_or_path
    data_path=args.data_path
    split_n=args.split_n
    gradient_accumulate=args.gradient_accumulate

    for arg in args._get_kwargs():
        print(arg)

    metrics = []
    raw_data: Dict[str, Any] = {}
    with open(data_path, "r") as f:
        raw_data = json.load(f)
    kfold = KFold(n_splits=split_n, shuffle=True)
    for train_indexs, valid_indexs in kfold.split(raw_data):
        """Timebank十折交叉验证  Eventstory五折交叉验证"""
        train_raw_dataset = [raw_data[i] for i in train_indexs]
        valid_raw_dataset = [raw_data[i] for i in valid_indexs]

        tokenizer = baseline_init_tokenizer(mlm_type, "mlm_base/mlm")
        mlm = RobertaForMaskedLM.from_pretrained(mlm_type)
        train_dataset = []
        valid_dataset = []
        for data in train_raw_dataset:
            train_dataset.extend(baseline_preprocess_data(data, tokenizer))
        for data in valid_raw_dataset:
            data = valid_data_preprocess(data)
            valid_dataset.extend(baseline_preprocess_data(data, tokenizer))
        # all_dataset=[]
        # for data in raw_data:
        #     all_dataset.extend(baseline_preprocess_data(data,tokenizer))
        # random.shuffle(all_dataset)
        # train_dataset,valid_dataset=split_dataset(all_dataset)

        model = BaselineFinetuningModel(mlm_type)
        optimizer = get_optimizer(model, learning_rate)
        # model=torch.nn.DataParallel(model,device_ids=device_ids)

        collator = BaselineCollator(tokenizer)

        # train_dataset_sampler=RandomSampler(train_dataset)
        train_dataset_sampler = BaselineSampler(train_dataset, True)
        # valid_dataset_sampler=BaselineSampler(valid_dataset,False)
        valid_dataset_sampler = SequentialSampler(valid_dataset)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            output_dir="prompt_tuning/saved",
            training_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=None,
            metrics_key="f1",
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            batch_forward_func=batch_forward_func,
            batch_cal_loss_func=batch_cal_loss_func,
            batch_metrics_func=batch_metrics_func,
            metrics_cal_func=metrics_cal_func,
            collate_fn=collator,
            device=dev,
            train_dataset_sampler=train_dataset_sampler,
            valid_dataset_sampler=valid_dataset_sampler,
            valid_step=1,
            start_epoch=0,
            gradient_accumulate=gradient_accumulate
        )

        trainer.train()
        metrics.append(trainer.epoch_metrics[trainer.get_best_epoch()])
    with open("prompt_tuning/saved/final_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    print()
    print(sum(list(map(lambda x: x["f1"], metrics)))/len(metrics))
