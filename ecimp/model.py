#!/home/zhouheng/anaconda3/bin/python
# -*- encoding: utf-8 -*-
'''
@文件    :model.py
@时间    :2021/12/22 16:48:02
@作者    :周恒
@版本    :1.0
@说明    :
'''


from typing import Any, Callable, Dict, Optional, Tuple, Union
from typing_extensions import Literal
import torch
from torch.functional import Tensor
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn.modules.sparse import Embedding
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoConfig
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.roberta import RobertaForMaskedLM, RobertaModel, RobertaConfig
from transformers.models.bert import BertForMaskedLM, BertModel, BertConfig
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from sklearn.metrics import recall_score, f1_score, precision_score
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, StepLR
from torch.optim import Optimizer

class ECIMPModel(torch.nn.Module):
    def __init__(self,
                 mlm_name_or_path: str,
                 use_event_prompt: bool,
                 use_signal_prompt: bool,
                 use_sep_gate: bool,
                 use_mask1_gate: bool,
                 use_linear:bool
        ):

        super().__init__()
        mlm_for_maskedlm: Union[BertForMaskedLM, RobertaForMaskedLM] = AutoModelForMaskedLM.from_pretrained(
            mlm_name_or_path)
        self.use_event_prompt = use_event_prompt
        self.use_signal_prompt = use_signal_prompt
        self.use_sep_gate = use_sep_gate
        self.use_mask1_gate = use_mask1_gate
        self.use_linear=use_linear

        self.mlm_config: Union[BertConfig, RobertaConfig] = AutoConfig.from_pretrained(
            mlm_name_or_path)
        self.hidden_dim = self.mlm_config.hidden_size
        self.dropout = torch.nn.Dropout()

        if hasattr(mlm_for_maskedlm, "bert"):
            assert type(mlm_for_maskedlm) == BertForMaskedLM
            self.mlm_type = "bert"
            self.mlm: BertModel = mlm_for_maskedlm.bert
            self.lm_head = mlm_for_maskedlm.cls.predictions.transform
            self.lm_decoder = mlm_for_maskedlm.cls.predictions.decoder

        elif hasattr(mlm_for_maskedlm, "roberta"):
            assert type(mlm_for_maskedlm) == RobertaForMaskedLM
            self.mlm_type = "roberta"
            self.mlm: RobertaModel = mlm_for_maskedlm.roberta
            self.lm_head = torch.nn.Sequential(
                mlm_for_maskedlm.lm_head.dense,
                torch.nn.GELU(),
                mlm_for_maskedlm.lm_head.layer_norm
            )
            self.lm_decoder = mlm_for_maskedlm.lm_head.decoder
        else:
            raise NotImplemented("目前仅支持bert,roberta")
        """1+3+14"""
        self.new_embedding = torch.nn.Embedding(18, self.hidden_dim)

        self.sep_gate_fc = torch.nn.Linear(self.hidden_dim*2, 1,bias=False)
        self.mask1_gate_fc = torch.nn.Linear(self.hidden_dim*2, 1,bias=False)
        self.mask_linear=torch.nn.Linear(self.hidden_dim,3,bias=False)
        # self.mask1_classification=torch.nn.Linear(self.hidden_dim,3)

    def forward_one_sentence(
        self,
        batch_mask: torch.Tensor,
        batch_input_ids: torch.Tensor,
        batch_mask1_index: torch.Tensor,
        batch_input_ids_for_new: torch.Tensor,
        batch_sep_event_index: torch.Tensor,
        batch_sep_signal_index: torch.Tensor
    ):

        batch_size, sequence_length = batch_input_ids.shape
        input_ids_for_new_repeated = batch_input_ids_for_new.reshape(
            [batch_size, sequence_length, 1]).repeat(1, 1, self.hidden_dim)
        # batch_size*sequence*768
        raw_embeddings: torch.Tensor = self.mlm.embeddings.word_embeddings(
            batch_input_ids)
        """把raw_embeddings中属于新加的17个token的部分置零"""
        zeros = torch.zeros(
            raw_embeddings.shape, dtype=raw_embeddings.dtype, device=raw_embeddings.device)
        raw_embeddings = torch.where(
            ~input_ids_for_new_repeated.bool(), raw_embeddings, zeros)
        # batch_size*sequence*768
        new_embeddings: torch.Tensor = self.new_embedding(
            batch_input_ids_for_new)
        """把new_embeddings中属于新加的17个token以外的部分置零"""
        new_embeddings = torch.where(
            input_ids_for_new_repeated.bool(), new_embeddings, zeros)
        input_embedding = new_embeddings+raw_embeddings

        mlm_output = self.mlm(inputs_embeds=input_embedding,
                              attention_mask=batch_mask)
        sequence_output = mlm_output[0]  # batch_size*sequence_length*768
        # batch_size*sequence_length*768
        lm_head_output = self.lm_head(sequence_output)
        # batch_size*sequence_length*vocab_size
        lm_decoder_output = self.lm_decoder(lm_head_output)
        mask1_feature: torch.Tensor = lm_head_output[torch.arange(batch_size),
                                                     batch_mask1_index.reshape([-1]), :]  # batch_size*768
        

        if self.use_mask1_gate:
            other_feature: Optional[torch.Tensor] = None
            if self.use_sep_gate:
                assert self.use_event_prompt and self.use_signal_prompt
                sep_event: torch.Tensor = lm_head_output[torch.arange(batch_size),
                                                         batch_sep_event_index.reshape([-1]), :]  # batch_size*768
                sep_signal: torch.Tensor = lm_head_output[torch.arange(batch_size),
                                                          batch_sep_signal_index.reshape([-1]), :]  # batch_size*768
                gate_weight: torch.Tensor = self.sep_gate_fc(
                    torch.cat([sep_event, sep_signal], dim=1))  # batch_size*1
                gate_weight = torch.sigmoid(gate_weight)
                gate_weight_repeated = gate_weight.repeat(
                    [1, self.hidden_dim])  # batch_size*768
                sep_gate_output = (gate_weight_repeated*sep_event) + \
                    ((1-gate_weight_repeated)*sep_signal)  # batch_size*768
                other_feature = sep_gate_output
            elif self.use_event_prompt:
                other_feature = lm_head_output[torch.arange(batch_size),
                                               batch_sep_event_index.reshape([-1]), :]  # batch_size*768
            elif self.use_signal_prompt:
                other_feature = lm_head_output[torch.arange(batch_size),
                                               batch_sep_signal_index.reshape([-1]), :]  # batch_size*768
            gate_weight = self.mask1_gate_fc(
                torch.cat([mask1_feature, other_feature], dim=1))
            gate_weight = torch.sigmoid(gate_weight)
            gate_weight_repeated = gate_weight.repeat([1, self.hidden_dim])
            mask1_feature = (gate_weight_repeated*mask1_feature) + \
                ((1-gate_weight_repeated)*other_feature)
            # mask1_feature=torch.tanh(mask1_feature)
        
        mask1_feature=self.dropout(mask1_feature)
        if self.use_linear:
            mask1_output=self.mask_linear(mask1_feature)
        else:
            mask1_weights: torch.Tensor = self.new_embedding(torch.tensor(
                [1, 2, 3], dtype=torch.long, device=mask1_feature.device)).T  # 768*3
            mask1_output = torch.mm(mask1_feature, mask1_weights)  # batch_size*3
        return mask1_output, lm_decoder_output

    def forward(self,
                batch_mask_1: torch.Tensor,
                batch_input_ids_1: torch.Tensor,
                batch_mask1_index_1: torch.Tensor,
                batch_input_ids_for_new_1: torch.Tensor,
                batch_sep_event_index_1: torch.Tensor,
                batch_sep_signal_index_1: torch.Tensor,
                batch_mask_2: torch.Tensor,
                batch_input_ids_2: torch.Tensor,
                batch_mask1_index_2: torch.Tensor,
                batch_input_ids_for_new_2: torch.Tensor,
                batch_sep_event_index_2: torch.Tensor,
                batch_sep_signal_index_2: torch.Tensor
                ):
        mask1_output_1, lm_decoder_output_1 = self.forward_one_sentence(
            batch_mask=batch_mask_1,
            batch_input_ids=batch_input_ids_1,
            batch_mask1_index=batch_mask1_index_1,
            batch_input_ids_for_new=batch_input_ids_for_new_1,
            batch_sep_event_index=batch_sep_event_index_1,
            batch_sep_signal_index=batch_sep_signal_index_1
        )
        mask1_output_2, lm_decoder_output_2 = self.forward_one_sentence(
            batch_mask=batch_mask_2,
            batch_input_ids=batch_input_ids_2,
            batch_mask1_index=batch_mask1_index_2,
            batch_input_ids_for_new=batch_input_ids_for_new_2,
            batch_sep_event_index=batch_sep_event_index_2,
            batch_sep_signal_index=batch_sep_signal_index_2
        )
        return mask1_output_1, lm_decoder_output_1, mask1_output_2, lm_decoder_output_2


def batch_forward_func(batch_data: Tuple[torch.Tensor, ...], trainer):
    batch_vocab_masks,        \
        batch_label_1,            \
        batch_mask_1,             \
        batch_input_ids_1,        \
        batch_cause_1,            \
        batch_mask1_index_1,      \
        batch_mask_for_mask2_1,   \
        batch_mask_for_mask3_1,   \
        batch_mask_for_mask4_1,   \
        batch_input_ids_for_new_1,\
        batch_sep_event_index_1,  \
        batch_sep_signal_index_1, \
        batch_label_2,            \
        batch_mask_2,             \
        batch_input_ids_2,        \
        batch_cause_2,            \
        batch_mask1_index_2,      \
        batch_mask_for_mask2_2,   \
        batch_mask_for_mask3_2,   \
        batch_mask_for_mask4_2,   \
        batch_input_ids_for_new_2,\
        batch_sep_event_index_2,  \
        batch_sep_signal_index_2 ,\
        batch_signals               = batch_data

    batch_vocab_masks,        \
        batch_label_1,            \
        batch_mask_1,             \
        batch_input_ids_1,        \
        batch_cause_1,            \
        batch_mask1_index_1,      \
        batch_mask_for_mask2_1,   \
        batch_mask_for_mask3_1,   \
        batch_mask_for_mask4_1,   \
        batch_input_ids_for_new_1,\
        batch_sep_event_index_1,  \
        batch_sep_signal_index_1, \
        batch_label_2,            \
        batch_mask_2,             \
        batch_input_ids_2,        \
        batch_cause_2,            \
        batch_mask1_index_2,      \
        batch_mask_for_mask2_2,   \
        batch_mask_for_mask3_2,   \
        batch_mask_for_mask4_2,   \
        batch_input_ids_for_new_2,\
        batch_sep_event_index_2,  \
        batch_sep_signal_index_2 ,\
        batch_signals = \
            batch_vocab_masks.cuda(trainer.device,non_blocking=True),        \
            batch_label_1.cuda(trainer.device,non_blocking=True),            \
            batch_mask_1.cuda(trainer.device,non_blocking=True),             \
            batch_input_ids_1.cuda(trainer.device,non_blocking=True),        \
            batch_cause_1.cuda(trainer.device,non_blocking=True),            \
            batch_mask1_index_1.cuda(trainer.device,non_blocking=True),      \
            batch_mask_for_mask2_1.cuda(trainer.device,non_blocking=True),   \
            batch_mask_for_mask3_1.cuda(trainer.device,non_blocking=True),   \
            batch_mask_for_mask4_1.cuda(trainer.device,non_blocking=True),   \
            batch_input_ids_for_new_1.cuda(trainer.device,non_blocking=True),\
            batch_sep_event_index_1.cuda(trainer.device,non_blocking=True),  \
            batch_sep_signal_index_1.cuda(trainer.device,non_blocking=True), \
            batch_label_2.cuda(trainer.device,non_blocking=True),            \
            batch_mask_2.cuda(trainer.device,non_blocking=True),             \
            batch_input_ids_2.cuda(trainer.device,non_blocking=True),        \
            batch_cause_2.cuda(trainer.device,non_blocking=True),            \
            batch_mask1_index_2.cuda(trainer.device,non_blocking=True),      \
            batch_mask_for_mask2_2.cuda(trainer.device,non_blocking=True),   \
            batch_mask_for_mask3_2.cuda(trainer.device,non_blocking=True),   \
            batch_mask_for_mask4_2.cuda(trainer.device,non_blocking=True),   \
            batch_input_ids_for_new_2.cuda(trainer.device,non_blocking=True),\
            batch_sep_event_index_2.cuda(trainer.device,non_blocking=True),  \
            batch_sep_signal_index_2.cuda(trainer.device,non_blocking=True) ,\
            batch_signals.cuda(trainer.device,non_blocking=True)
        
    mask1_output_1, lm_decoder_output_1, mask1_output_2, lm_decoder_output_2 = trainer.model(
        batch_mask_1,
        batch_input_ids_1,
        batch_mask1_index_1,
        batch_input_ids_for_new_1,
        batch_sep_event_index_1,
        batch_sep_signal_index_1,
        batch_mask_2,
        batch_input_ids_2,
        batch_mask1_index_2,
        batch_input_ids_for_new_2,
        batch_sep_event_index_2,
        batch_sep_signal_index_2
    )
    return \
        (
            batch_vocab_masks,
            batch_label_1,
            batch_cause_1,
            batch_mask_for_mask2_1,
            batch_mask_for_mask3_1,
            batch_mask_for_mask4_1,
            batch_label_2,
            batch_cause_2,
            batch_mask_for_mask2_2,
            batch_mask_for_mask3_2,
            batch_mask_for_mask4_2,
            batch_signals
            
        ),\
        (
            mask1_output_1,
            lm_decoder_output_1,
            mask1_output_2,
            lm_decoder_output_2
        )


class BatchCalLossFunc:
    def __init__(self, use_event_prompt: bool, use_signal_prompt: bool) -> None:
        self.use_event_prompt = use_event_prompt
        self.use_signal_prompt = use_signal_prompt

    def __call__(self, labels: Tuple[torch.Tensor, ...], preds: Tuple[torch.Tensor, ...], trainer) -> Any:
        batch_vocab_masks,\
            batch_label_1,\
            batch_cause_1,\
            batch_mask_for_mask2_1,\
            batch_mask_for_mask3_1,\
            batch_mask_for_mask4_1,\
            batch_label_2,\
            batch_cause_2,\
            batch_mask_for_mask2_2,\
            batch_mask_for_mask3_2,\
            batch_mask_for_mask4_2,\
            batch_signals = labels

        mask1_output_1,\
            lm_decoder_output_1,\
            mask1_output_2,\
            lm_decoder_output_2 = preds

        batch_size, sequence_length, vocab_size = lm_decoder_output_1.shape

        mask1_loss,\
        mask2_loss,\
        mask3_loss,\
        mask4_loss =\
            torch.tensor(0, dtype=torch.float32, device=lm_decoder_output_1.device),\
            torch.tensor(0, dtype=torch.float32, device=lm_decoder_output_1.device),\
            torch.tensor(0, dtype=torch.float32, device=lm_decoder_output_1.device),\
            torch.tensor(0, dtype=torch.float32, device=lm_decoder_output_1.device)

        """mask 1 loss"""
        mask1_loss_1 = F.cross_entropy(
            mask1_output_1, batch_cause_1, reduction="mean")
        mask1_loss_2 = F.cross_entropy(
            mask1_output_2, batch_cause_2, reduction="mean")
        mask1_loss = (mask1_loss_1+mask1_loss_2)/batch_size  # 标量

        """mask 234 loss"""
        if self.use_event_prompt or self.use_signal_prompt:
            # batch_vocab_masks   batch_size*vocab_size
            batch_vocab_masks_repeated = batch_vocab_masks.reshape(
                [batch_size, 1, vocab_size])
            # batch_size*sequence_length*vocab_size
            batch_vocab_masks_repeated = batch_vocab_masks_repeated.repeat(
                [1, sequence_length, 1])

            """不在原句里的token的output全部设为-inf"""
            lm_decoder_output_1 = lm_decoder_output_1+batch_vocab_masks_repeated
            lm_decoder_output_2 = lm_decoder_output_2+batch_vocab_masks_repeated

            # batch_size*sequence_length
            lm_decoder_loss_1 = F.cross_entropy(lm_decoder_output_1.reshape(
                [-1, vocab_size]), batch_label_1.reshape([-1]), reduction="none").reshape([batch_size, sequence_length])
            lm_decoder_loss_2 = F.cross_entropy(lm_decoder_output_2.reshape(
                [-1, vocab_size]), batch_label_2.reshape([-1]), reduction="none").reshape([batch_size, sequence_length])
            lm_decoder_loss_1 = torch.where(torch.isnan(lm_decoder_loss_1) | torch.isinf(lm_decoder_loss_1) | torch.isneginf(
                    lm_decoder_loss_1), torch.tensor(0, dtype=torch.float, device=lm_decoder_loss_1.device), lm_decoder_loss_1)
            lm_decoder_loss_2 = torch.where(torch.isnan(lm_decoder_loss_2) | torch.isinf(lm_decoder_loss_2) | torch.isneginf(
                    lm_decoder_loss_2), torch.tensor(0, dtype=torch.float, device=lm_decoder_loss_2.device), lm_decoder_loss_2)
            
            if self.use_event_prompt:
                
                """mask2 loss 因为event长度不一定为1,所以要加权平均"""
                # batch_size*sequence_length

                mask2_loss_1 = lm_decoder_loss_1*batch_mask_for_mask2_1
                mask2_loss_2 = lm_decoder_loss_2*batch_mask_for_mask2_2

                # batch_size
                mask2_loss_1 = mask2_loss_1.sum(
                    dim=-1)/batch_mask_for_mask2_1.sum(dim=-1)
                mask2_loss_2 = mask2_loss_2.sum(
                    dim=-1)/batch_mask_for_mask2_2.sum(dim=-1)

                mask2_loss = (mask2_loss_1+mask2_loss_2).sum()/batch_size  # 标量

                """mask3 loss 同上"""
                mask3_loss_1 = lm_decoder_loss_1*batch_mask_for_mask3_1
                mask3_loss_2 = lm_decoder_loss_2*batch_mask_for_mask3_2
                # batch_size
                mask3_loss_1 = mask3_loss_1.sum(
                    dim=-1)/batch_mask_for_mask3_1.sum(dim=-1)
                mask3_loss_2 = mask3_loss_2.sum(
                    dim=-1)/batch_mask_for_mask3_2.sum(dim=-1)

                mask3_loss = (mask3_loss_1+mask3_loss_2).sum()/batch_size  # 标量
            if self.use_signal_prompt:
                """mask4 loss 同上"""
                mask4_loss_1 = lm_decoder_loss_1*batch_mask_for_mask4_1
                mask4_loss_2 = lm_decoder_loss_2*batch_mask_for_mask4_2
                # batch_size
                batch_mask_for_mask4_sum_1 = batch_mask_for_mask4_1.sum(dim=1)
                batch_mask_for_mask4_sum_2 = batch_mask_for_mask4_2.sum(dim=1)

                """处理mask"""
                mask4_loss_1 = mask4_loss_1.sum(dim=-1)/torch.where(batch_mask_for_mask4_sum_1 == 0, torch.tensor(
                    1, device=mask4_loss_1.device, dtype=torch.float), batch_mask_for_mask4_sum_1)
                mask4_loss_2 = mask4_loss_2.sum(dim=-1)/torch.where(batch_mask_for_mask4_sum_2 == 0, torch.tensor(
                    1, device=mask4_loss_2.device, dtype=torch.float), batch_mask_for_mask4_sum_2)

                mask4_loss_1 = mask4_loss_1 * \
                    (batch_mask_for_mask4_sum_1.bool().float())
                mask4_loss_2 = mask4_loss_2 * \
                    (batch_mask_for_mask4_sum_2.bool().float())
                mask4_loss = (mask4_loss_1+mask4_loss_2).sum()/batch_size  # 标量
        trainer.logger.info(
            f"\nmask1 loss : {mask1_loss.item()}\tmask2 loss : {mask2_loss.item()}\tmask3 loss : {mask3_loss.item()}\tmask4 loss : {mask4_loss.item()}")
        return mask1_loss*5+mask2_loss+mask3_loss+mask4_loss


batch_cal_loss_func = BatchCalLossFunc(True, True)


def get_optimizer(model: ECIMPModel, lr: float):
    raw_params=set()
    for layer in [model.mlm,model.lm_head,model.lm_decoder]:
        for param in layer.parameters():
            raw_params.add(param)
    raw_params=list(raw_params)
    optimizer = torch.optim.AdamW([
        {"params": raw_params, "lr": lr/5},
        {"params": model.new_embedding.parameters()},
        {"params": model.sep_gate_fc.parameters()},
        {"params":model.mask1_gate_fc.parameters()},
        {"params":model.mask_linear.parameters()}
    ], lr=lr)
    return optimizer


class BatchMetricsFunc:
    def __init__(self, mode: Literal["and", "or"] = "or") -> None:
        self.mode = mode

    def __call__(self, labels: Tuple[torch.Tensor, ...], preds: Tuple[torch.Tensor, ...], metrics: Dict[str, Union[bool, int, float]], trainer):

        batch_cause_1 = labels[2]
        batch_signals=labels[-1]
        batch_signals=batch_signals.cpu()
        mask1_output_1,\
            lm_decoder_output_1,\
            mask1_output_2,\
            lm_decoder_output_2 = preds

        causes = batch_cause_1.reshape([-1]).bool().long().cpu()

        cause_preds = None
        cause_preds1 = torch.argmax(mask1_output_1, dim=1).reshape([-1]).cpu()
        cause_preds2 = torch.argmax(mask1_output_2, dim=1).reshape([-1]).cpu()
        if self.mode == "or":
            """正反两方向只要有一个预测有关系，就视作有关系"""
            cause_preds = torch.logical_or(cause_preds1, cause_preds2).long()

        elif self.mode == "and":
            cause_preds = torch.logical_and(cause_preds1, cause_preds2).long()

        batch_nosignals=torch.logical_and(torch.logical_not(batch_signals),causes.bool()).long()
        nosignal_labels=(batch_nosignals*causes).long()

        signal_labels=batch_signals*causes

        signal_preds=batch_signals*cause_preds
        nosignal_preds=batch_nosignals*cause_preds

        trainer.logger.info("labels: \n{}".format(causes))
        trainer.logger.info("preds: \n{}".format(cause_preds))

        precision = precision_score(causes.numpy(), cause_preds.numpy(),zero_division=0)
        recall = recall_score(causes.numpy(), cause_preds.numpy(),zero_division=0)
        f1 = f1_score(causes.numpy(), cause_preds.numpy(),zero_division=0)
        signal_recall=recall_score(signal_labels.numpy(),signal_preds.numpy(),zero_division=0)
        nosignal_recall=recall_score(nosignal_labels.numpy(),nosignal_preds.numpy(),zero_division=0)
        batch_metrics={"precision":precision,"recall":recall,"f1":f1,"signal_recall":signal_recall,"nosignal_recall":nosignal_recall}
        if "labels" in metrics:
            metrics["labels"] = torch.cat([metrics["labels"], causes], dim=0)
        else:
            metrics["labels"] = causes
        if "preds" in metrics:
            metrics["preds"] = torch.cat(
                [metrics["preds"], cause_preds], dim=0)
        else:
            metrics["preds"] = cause_preds

        if "signal_labels" in metrics:
            metrics["signal_labels"]=torch.cat([metrics["signal_labels"],signal_labels],dim=0)
        else:
            metrics["signal_labels"]=signal_labels

        if "signal_preds" in metrics:
            metrics["signal_preds"]=torch.cat([metrics["signal_preds"],signal_preds],dim=0)
        else:
            metrics["signal_preds"]=signal_preds
        if "nosignal_labels" in metrics:
            metrics["nosignal_labels"]=torch.cat([metrics["nosignal_labels"],nosignal_labels],dim=0)
        else:
            metrics["nosignal_labels"]=nosignal_labels

        if "nosignal_preds" in metrics:
            metrics["nosignal_preds"]=torch.cat([metrics["nosignal_preds"],nosignal_preds],dim=0)
        else:
            metrics["nosignal_preds"]=nosignal_preds

        return metrics, batch_metrics


batch_metrics_func = BatchMetricsFunc()


def metrics_cal_func(metrics: Dict[str, torch.Tensor]):
    causes=metrics["labels"]
    cause_preds=metrics["preds"]
    signal_labels=metrics["signal_labels"]
    signal_preds=metrics["signal_preds"]
    nosignal_labels=metrics["nosignal_labels"]
    nosignal_preds=metrics["nosignal_preds"]
    precision=precision_score(causes.numpy(),cause_preds.numpy())
    recall=recall_score(causes.numpy(),cause_preds.numpy())
    f1=f1_score(causes.numpy(),cause_preds.numpy())
    signal_recall=recall_score(signal_labels.numpy(),signal_preds.numpy())
    nosignal_recall=recall_score(nosignal_labels.numpy(),nosignal_preds.numpy(),zero_division=0)
    res={"precision":precision,"recall":recall,"f1":f1,"signal_recall":signal_recall,"nosignal_recall":nosignal_recall}
    return res

class ECIMPLrScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, last_epoch: int = ...) -> None:
        self.optimizer=optimizer
        self.step_count=0
    def step(self, epoch: Optional[int] = ...) -> None:

        self.step_count+=1
        if self.step_count==1:
            lr_set=set()
            for p in self.optimizer.param_groups:
                lr_set.add(str(p["lr"]))
            lrs=list(map(lambda x:float(x),lr_set))
            max_lr=max(lrs)
            for p in self.optimizer.param_groups:
                p["lr"]=max_lr
                   
            
        # return super().step(epoch=epoch)