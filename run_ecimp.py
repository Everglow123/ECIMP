#!/home/zhouheng/anaconda3/bin/python
# -*- encoding: utf-8 -*-
'''
@文件    :run_final.py
@时间    :2021/12/07 20:26:34
@作者    :周恒
@版本    :1.0
@说明    :
'''
import sys
import pickle
import random
import json
from typing import Any, Dict
from typing_extensions import Literal
import numpy as np
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from trainer import Trainer
import torch
import argparse
from ecimp import (
    ecimp_init_tokenizer,
    ecimp_preprocess_data,
    ECIMPCollator,
    ECIMPModel,
    batch_cal_loss_func,
    batch_metrics_func,
    batch_forward_func,
    metrics_cal_func,
    get_optimizer,
    valid_data_preprocess,
    ECIMPSampler,
    ECIMPLrScheduler
)
from transformers.models.roberta import RobertaForMaskedLM, RobertaTokenizer
from sklearn.model_selection import KFold


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
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=12)
parser.add_argument("--learning_rate", type=float, default=0.00005)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--mlm_name_or_path", type=str, default="roberta-base")
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--split_n", type=int, required=True)

parser.add_argument("--use_event_prompt", type=int, required=True)
parser.add_argument("--use_signal_prompt", type=int, required=True)
parser.add_argument("--use_sep_gate", type=int, required=True)
parser.add_argument("--use_mask1_gate", type=int, required=True)
parser.add_argument("--use_linear", type=int, default=0)
parser.add_argument("--reuse", type=int, required=True)
parser.add_argument("--gradient_accumulate", type=int, default=3)

if __name__ == '__main__':
    set_random_seed(114514)
    args = parser.parse_args()
    """一些常规的设置"""
    dev = torch.device(args.main_device)
    device_ids = list(map(lambda x: int(x), args.device_ids.split(",")))
    batch_size = args.batch_size
    num_workers = args.num_workers
    learning_rate = args.learning_rate
    epochs = args.epochs
    mlm_type = args.mlm_name_or_path
    data_path = args.data_path
    split_n = args.split_n
    gradient_accumulate = args.gradient_accumulate

    """一个样本构造两个输入，经过模型获得两个输出，如果为and 那么当两个输出都认为有因果关系时，认为这个样本有因果关系
        如果为or，那么只要有一个输出认为有因果关系，就视作该样本有因果关系
    """
    metrics_mode: Literal["and", "or"] = "or"

    """ecimp模型的对比实验设置"""
    use_linear: bool = bool(args.use_linear) #是否使用线性层
    use_event_prompt: bool = bool(args.use_event_prompt)#是否使用event_prompt
    use_signal_prompt: bool = bool(args.use_signal_prompt)#是否使用signal_prompt
    use_sep_gate: bool = bool(args.use_sep_gate)#是否使用sep_gate
    use_mask1_gate: bool = bool(args.use_mask1_gate)#是否使用mask1_gate
    reuse: bool = bool(args.reuse)  # 是否复用 na cause caused_by 这3个向量
    """"""
    for arg in args._get_kwargs():
        print(arg)

    """functor初始化"""
    ecimp_preprocess_data.reuse = reuse
    ecimp_preprocess_data.use_signal_prompt = use_signal_prompt
    ecimp_preprocess_data.use_event_prompt = use_event_prompt
    batch_metrics_func.mode = metrics_mode
    batch_cal_loss_func.use_event_prompt = use_event_prompt
    batch_cal_loss_func.use_signal_prompt = use_signal_prompt

    metrics = []
    raw_data: Dict[str, Any] = {}
    with open(data_path, "r") as f:
        raw_data = json.load(f)
    kfold = KFold(n_splits=split_n, shuffle=True)
    for train_indexs, valid_indexs in kfold.split(raw_data):
        """Timebank十折交叉验证  Eventstory五折交叉验证"""
        train_raw_dataset = [raw_data[i] for i in train_indexs]
        valid_raw_dataset = [raw_data[i] for i in valid_indexs]

        tokenizer = ecimp_init_tokenizer(mlm_type, "ecimp/mlm")
        mlm = RobertaForMaskedLM.from_pretrained(mlm_type)
        train_dataset = []
        valid_dataset = []
        for data in train_raw_dataset:
            train_dataset.extend(ecimp_preprocess_data(data, tokenizer))
        for data in valid_raw_dataset:
            data = valid_data_preprocess(data)
            valid_dataset.extend(ecimp_preprocess_data(data, tokenizer))
        # all_dataset=[]
        # for data in raw_data:
        #     all_dataset.extend(baseline_preprocess_data(data,tokenizer))
        # random.shuffle(all_dataset)
        # train_dataset,valid_dataset=split_dataset(all_dataset)

        model = ECIMPModel(
            mlm_type,
            use_event_prompt=use_event_prompt,
            use_signal_prompt=use_signal_prompt,
            use_sep_gate=use_sep_gate,
            use_mask1_gate=use_mask1_gate,
            use_linear=use_linear
        )
        optimizer = get_optimizer(model, learning_rate)

        collator = ECIMPCollator(
            tokenizer,
            use_event_prompt=use_event_prompt,
            use_signal_prompt=use_signal_prompt
        )

        # train_dataset_sampler=RandomSampler(train_dataset)
        train_dataset_sampler = ECIMPSampler(train_dataset, True)
        # valid_dataset_sampler=BaselineSampler(valid_dataset,False)
        valid_dataset_sampler = SequentialSampler(valid_dataset)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            output_dir="ecimp/saved",
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
    with open("ecimp/saved/final_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    print(metrics)

    print()
    print(sum(list(map(lambda x:x["f1"],metrics)))/len(metrics))