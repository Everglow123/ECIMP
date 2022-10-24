#! C:\Users\92429\Anaconda3\python.exe
# -*- encoding: utf-8 -*-
'''
@File    :   active_learning_trainer.py
@Time    :   2021/09/18 13:48:54
@Author  :   zhouheng
@Version :   1.0
'''

import os
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, StepLR
from torch.utils.data import dataset, Sampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, overload
from abc import abstractmethod

# from torch.utils.data.sampler import RandomSampler, Sampler
import pickle
import json
import numpy as np
import logging
import traceback
from tqdm import tqdm
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)s:%(funcName)s] - %(message)s",
)


# class QueryStrategy:

#     def __init__(self) -> None:
#         pass

#     @abstractmethod
#     def sample(self, trainer: 'Trainer', top_n: int, **kwargs) -> None:
#         """直接从trainer的query数据中采样到训练数据集中"""
#         pass


# class RandomSampleStrategy(QueryStrategy):
#     def __init__(self) -> None:
#         super().__init__()
#         self.unlabeled_index: np.ndarray = None

#     @overload
#     def sample(self, trainer: 'Trainer', top_n: int, **kwargs) -> None:
#         """初始化未标记样本索引"""
#         if self.unlabeled_index == None:
#             self.unlabeled_index = np.arange(
#                 0, len(trainer.query_dataset), step=1, dtype=np.int)
#         np.random.shuffle(self.unlabeled_index)
#         count = min(top_n, len(self.unlabeled_index))
#         indexs: np.ndarray = self.unlabeled_index[:count]
#         self.unlabeled_index = self.unlabeled_index[count:]
#         datas = [trainer.query_dataset[indexs[i]] for i in range(count)]
#         trainer.training_dataset += datas

# class MySample(Sampler):


class Trainer(object):
    def __init__(self,
                 model: Module,
                 optimizer: Optimizer,
                 output_dir: str,
                 training_dataset: Dataset,
                 valid_dataset: Dataset,
                 test_dataset: Dataset,
                 metrics_key: str,
                 epochs: int,
                 batch_size: int,
                 num_workers: int,
                 batch_forward_func: Callable[[Tuple[torch.Tensor, ...], 'Trainer'], Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]], Union[torch.Tensor, Tuple[torch.Tensor, ...]]]],
                 batch_cal_loss_func: Callable[[Union[torch.Tensor, Tuple[torch.Tensor, ...]], Union[torch.Tensor, Tuple[torch.Tensor, ...]], 'Trainer'], torch.Tensor],
                 batch_metrics_func: Callable[[Union[torch.Tensor, Tuple[torch.Tensor, ...]], Union[torch.Tensor, Tuple[torch.Tensor, ...]], Dict[str, Union[int, torch.Tensor]], 'Trainer'], Tuple[Dict[str, Union[int, torch.Tensor]], Dict[str, Union[int, torch.Tensor]]]],
                 metrics_cal_func: Callable[[Dict[str, Union[int, torch.Tensor]]], Dict[str, int]],
                 device: torch.device = torch.device("cpu"),
                 resume_path: str = None,
                 start_epoch: int = 0,
                 train_dataset_sampler: Sampler = None,
                 valid_dataset_sampler: Sampler = None,
                 collate_fn=None,
                 valid_step=1,
                 lr_scheduler: _LRScheduler = None,
                 gradient_accumulate=1,
                 save_model: bool = True,
                 save_model_steps: int = 5
                 ) -> None:
        """

        """
        self.variables: Dict[str, Any] = {}

        # dict<epoch,metrics>
        self.epoch_metrics: List[Dict[str, int]] = []
        self.device = device
        self.model = model
        if not isinstance(self.model, torch.nn.parallel.DataParallel):
            self.model = self.model.to(self.device)
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.optimizer = optimizer
        self.training_dataset = training_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.batch_forward_func = batch_forward_func
        self.batch_cal_loss_func = batch_cal_loss_func
        self.batch_metrics_func = batch_metrics_func
        self.metrics_cal_func = metrics_cal_func
        self.metrics_key = metrics_key
        self.valid_step = valid_step
        self.lr_scheduler = lr_scheduler
        self.gradient_accumulate = gradient_accumulate
        self.save_model = save_model
        # if self.lr_scheduler==None:
        #     self.lr_scheduler=CosineAnnealingLR(self.optimizer,eta_min=1e-6,verbose=True,T_max=40)
        if self.training_dataset != None and len(self.training_dataset) > 0:
            self.training_dataloader = DataLoader(
                self.training_dataset, batch_size=self.batch_size, shuffle=True if train_dataset_sampler == None else False,
                num_workers=self.num_workers, drop_last=False, sampler=train_dataset_sampler,
                collate_fn=self.collate_fn)
        if self.valid_dataset != None and len(self.valid_dataset) > 0:
            self.valid_dataloader = DataLoader(
                self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                sampler=valid_dataset_sampler, collate_fn=self.collate_fn)
        if self.test_dataset != None and len(self.test_dataset) > 0:
            self.test_dataloader = DataLoader(
                self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn
            )

        self.logger = logging.getLogger(self.__class__.__name__)
        self.start_epoch = start_epoch
        self.save_model_steps = save_model_steps
        # if resume_path != None:
        #     with open(resume_path, "rb") as f:
        #         self.model.load_state_dict(torch.load(f))

    def valid_epoch(self, epoch: int):
        self.model.eval()
        data_iter = iter(self.valid_dataloader)
        batch_index = 0
        total_loss = 0.0
        metrics = {}
        with tqdm(total=len(self.valid_dataloader), ncols=80) as tqbar:
            with torch.no_grad():
                while True:
                    data = None
                    try:
                        data = next(data_iter)
                    except Exception:
                        break
                    labels, preds = self.batch_forward_func(data, self)
                    metrics, batch_metrics = self.batch_metrics_func(
                        labels, preds, metrics, self)
                    loss = self.batch_cal_loss_func(labels, preds, self)
                    total_loss += loss.item()
                    batch_index += 1
                    tqbar.update(1)
            self.logger.info("epoch {0} : valid mean loss {1}".format(
                epoch, total_loss/len(self.valid_dataloader)))
            metrics_result = self.metrics_cal_func(metrics)
            for k, v in metrics_result.items():
                self.logger.info(
                    "epoch {0} : valid {1}\t{2}".format(epoch, k, v))
        return metrics_result

    def train_epoch(self, epoch: int) -> Dict[str, int]:
        self.model = self.model.to(self.device)
        self.model.train()
        with tqdm(total=len(self.training_dataloader), ncols=80) as tqbar:
            data_iter = iter(self.training_dataloader)
            batch_index = 0
            metrics = {}
            total_loss = 0.0
            while True:
                data = None
                try:
                    data = next(data_iter)
                except StopIteration:
                    break
                except Exception as ex:
                    self.logger.warn(ex)

                    break

                labels, preds = self.batch_forward_func(data, self)
                loss = self.batch_cal_loss_func(labels, preds, self)

                loss.backward()
                if (batch_index+1) % self.gradient_accumulate == 0 or batch_index == len(self.training_dataloader)-1:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                metrics, batch_matrics = self.batch_metrics_func(
                    labels, preds, metrics, self)
                total_loss += loss.item()
                self.logger.info("epoch {0} : batch {1}/{2} mean loss : {3}".format(
                    epoch, (batch_index+1), len(self.training_dataloader), loss.item()))
                for k, v in batch_matrics.items():
                    self.logger.info("epoch {0} : batch {1}/{2} training {3} : {4}".format(
                        epoch, (batch_index+1), len(self.training_dataloader), k, v))
                batch_index += 1
                tqbar.update(1)
            metrics_result = self.metrics_cal_func(metrics)
            if self.lr_scheduler != None:
                self.lr_scheduler.step()
            self.logger.info("epoch {0} : trainging mean loss {1}".format(
                epoch, total_loss/len(self.training_dataloader)))
            for k, v in metrics_result.items():
                self.logger.info(
                    "epoch {0} : training {1} : {2}".format(epoch, k, v))
            # torch.cuda.empty_cache()
            return metrics_result

    def test_epoch(self, epoch: int):
        self.model = self.model.to(self.device)
        self.model.eval()
        data_iter = iter(self.test_dataloader)
        batch_index = 0
        test_result = None
        with torch.no_grad():
            data = None
            while True:
                try:
                    data = next(data_iter)
                except StopIteration:
                    break
                labels, preds = self.batch_forward_func(data, self)
                if type(preds) == tuple:
                    if test_result == None:
                        temp = []
                        for i, ele in enumerate(preds):
                            if type(ele) == list:
                                temp.append(ele)
                            elif type(ele) == torch.Tensor:
                                temp.append(ele.cpu())
                            elif type(ele) == np.ndarray:
                                temp.append(ele)
                            else:
                                raise RuntimeError("preds元素类型错误")
                        test_result = tuple(temp)
                    else:
                        temp = []
                        for i, ele in enumerate(preds):
                            if type(ele) == list:
                                temp.append(test_result[i]+ele)
                            elif type(ele) == torch.Tensor:
                                temp.append(
                                    torch.cat([test_result[i], ele.cpu()], dim=0))
                            elif type(ele) == np.ndarray:
                                temp.append(
                                    np.append(test_result[i], ele, axis=0)
                                )

                        test_result = tuple(temp)
                elif type(preds) == torch.Tensor:
                    if test_result == None:
                        test_result = preds.cpu()
                    else:
                        test_result = torch.cat(
                            [test_result, preds.cpu()], dim=0)
                elif type(preds) == np.ndarray:
                    if(test_result) == None:
                        test_result = preds
                    else:
                        test_result: np.ndarray = np.append(
                            test_result, preds, axis=0)
                elif type(preds) == list:
                    if test_result == None:
                        test_result = preds
                    else:
                        test_result = test_result+preds
                else:
                    raise RuntimeError("preds元素类型错误")

        return test_result

    def get_best_epoch(self) -> Optional[int]:
        if len(self.epoch_metrics) == 0:
            return None
        else:
            res = 0
            for i in range(len(self.epoch_metrics)):
                if self.epoch_metrics[i][self.metrics_key] > \
                        self.epoch_metrics[res][self.metrics_key]:
                    res = i
            return res

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            try:
                self.logger.info("开始训练 epoch : {}".format(epoch))
                self.train_epoch(epoch)
                if (epoch+1) % self.valid_step == 0 and self.valid_dataset is not None:
                    self.epoch_metrics.append(self.valid_epoch(epoch))
                    with open(os.path.join(self.output_dir, "metrics{0}.json".format(epoch)), "w") as f:
                        json.dump(
                            self.epoch_metrics[-1], f, indent=4, ensure_ascii=False, default=str)
                if self.save_model:
                    if epoch % self.save_model_steps == 0:
                        if isinstance(self.model, torch.nn.DataParallel):
                            torch.save(self.model.module.state_dict(),
                                       os.path.join(self.output_dir, "epoch{0}.pt".format(epoch)))
                        else:
                            torch.save(self.model.state_dict(),
                                       os.path.join(self.output_dir, "epoch{0}.pt".format(epoch)))

                if self.test_dataset != None:
                    test_result = self.test_epoch(epoch)
                    torch.save(
                        test_result, os.path.join(self.output_dir, "test_result{0}.bin".format(epoch)))
            except Exception as ex:
                traceback.print_stack()
                self.logger.warn(ex)
