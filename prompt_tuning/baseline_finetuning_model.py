#!/home/zhouheng/anaconda3/bin/python
# -*- encoding: utf-8 -*-
'''
@文件    :baseline_finetuning_model.py
@说明    :
@时间    :2021/11/27 16:02:31
@作者    :周恒
@版本    :1.0
'''



from typing import Callable, Dict, Optional, Tuple,Union
import torch
from torch.functional import Tensor
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn.modules.sparse import Embedding
from torch.types import Number
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM,AutoConfig
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.roberta import RobertaForMaskedLM,RobertaModel,RobertaConfig
from transformers.models.bert import BertForMaskedLM,BertModel,BertConfig
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from sklearn.metrics import recall_score,f1_score,precision_score
# from data_process import T1,T2,T3,T4,T5,T6,CAUSEOF,NOTCAUSEOF


class BaselineFinetuningModel(torch.nn.Module):
    def __init__(self,mlm_name_or_path:str):
        super().__init__()
        mlm_for_maskedlm:Union[BertForMaskedLM,RobertaForMaskedLM]=AutoModelForMaskedLM.from_pretrained(mlm_name_or_path)
        self.mlm_config:Union[BertConfig,RobertaConfig]=AutoConfig.from_pretrained(mlm_name_or_path)
        self.hidden_dim=self.mlm_config.hidden_size
        if hasattr(mlm_for_maskedlm,"bert"):
            assert type(mlm_for_maskedlm)==BertForMaskedLM
            self.mlm_type="bert"
            self.mlm:BertModel=mlm_for_maskedlm.bert
            self.lm_head=mlm_for_maskedlm.cls.predictions.transform
            # self.lm_decoder=mlm_for_maskedlm.cls.predictions.decoder

        elif hasattr(mlm_for_maskedlm,"roberta"):
            assert type(mlm_for_maskedlm)==RobertaForMaskedLM
            self.mlm_type="roberta"
            self.mlm:RobertaModel=mlm_for_maskedlm.roberta
            self.lm_head=torch.nn.Sequential(
                mlm_for_maskedlm.lm_head.dense,
                torch.nn.GELU(),
                mlm_for_maskedlm.lm_head.layer_norm
            ) 
            # self.lm_decoder=mlm_for_maskedlm.lm_head.decoder
        else:
            raise NotImplemented("目前仅支持bert,roberta")
        
        
        self.dropout=torch.nn.Dropout()

        """1+6,第一个为占位"""
        self.new_embedding=torch.nn.Embedding(7,self.hidden_dim) 

        

        # self.fc=torch.nn.Linear(self.hidden_dim,8)
        self.classification=torch.nn.Linear(self.hidden_dim,3)
    # def train(self, mode: bool = True):
    #     super().train(mode=mode)
    #     for param in self.mlm.parameters():
    #         param.requires_grad=False
    #     self.lm_head.requires_grad_(False)
    #     self.lm_decoder.requires_grad_(False)
    #     # self.mlm_word_embedding.requires_grad_(False)

    #     self.new_embedding.requires_grad_(True)
    #     self.classification.requires_grad_(True)


    #     # self.fc.requires_grad_(True)
    #     # self.new_embedding.requires_grad_()

    #     return self
    def forward_one_sentence(
            self,
            input_ids:torch.Tensor,#batch_size,sequence_length
            masks:torch.Tensor,#batch_size,sequence_length
            input_ids_for_new:torch.Tensor,#用于指示输入序列中属于6个新embedding的部分 值为[0,6]
            mask_positions:torch.Tensor
        ):
        batch_size,sequence_length=input_ids.shape


        """扩展input_ids_for_new,使之为batch_size*sequence*768"""

        input_ids_for_new_repeated=input_ids_for_new.reshape([batch_size,sequence_length,1]).repeat(1,1,self.hidden_dim)


        #batch_size*sequence*768
        
        raw_embeddings:torch.Tensor=self.mlm.embeddings.word_embeddings(input_ids)
        """把raw_embeddings中属于新加的6个token的部分置零"""
        zeros=torch.zeros(raw_embeddings.shape,dtype=raw_embeddings.dtype,device=raw_embeddings.device)
        raw_embeddings=torch.where(~input_ids_for_new_repeated.bool(),raw_embeddings,zeros)

        #batch_size*sequence*768
        new_embeddings:torch.Tensor=self.new_embedding(input_ids_for_new)
        """把new_embeddings中属于新加的6个token以外的部分置零"""
        new_embeddings=torch.where(input_ids_for_new_repeated.bool(),new_embeddings,zeros)

        input_embedding=new_embeddings+raw_embeddings
       
         

        mlm_output=self.mlm(inputs_embeds=input_embedding,attention_mask=masks)

        sequence_output = mlm_output[0]#batch_size*sequence_length*768 

        lm_head_output=self.lm_head(sequence_output)#batch_size*sequence_length*768 

        masked_features=lm_head_output[torch.arange(batch_size),\
            mask_positions.reshape([-1]),:]#batch_size*768
        if self.training:
            masked_features=self.dropout(masked_features)
        
        prediction=self.classification(masked_features)#batch_size*2

        # lm_decoder_output=self.lm_decoder(lm_head_output)#batch_size*sequence_length*vocab_size 

        # new_decoder_output=self.fc(lm_head_output)#batch_size*sequence_length*8 
        
        # prediction=torch.cat([lm_decoder_output,new_decoder_output],dim=2)#batch_size*sequence_length*(vocab_size+8) 

        return prediction

    def forward(
            self,
            input_ids1:torch.Tensor,#batch_size,sequence_length
            masks1:torch.Tensor,#batch_size,sequence_length
            input_ids_for_new1:torch.Tensor,#用于指示输入序列中属于6个新embedding的部分 值为[0,6]
            mask_positions1:torch.Tensor,
            input_ids2:torch.Tensor,#batch_size,sequence_length
            masks2:torch.Tensor,#batch_size,sequence_length
            input_ids_for_new2:torch.Tensor,#用于指示输入序列中属于6个新embedding的部分 值为[0,6]
            mask_positions2:torch.Tensor
        ): 
        pred1=self.forward_one_sentence(input_ids1,masks1,input_ids_for_new1,mask_positions1)
        pred2=self.forward_one_sentence(input_ids2,masks2,input_ids_for_new2,mask_positions2)
        return pred1,pred2
    

def get_optimizer(model:BaselineFinetuningModel,lr:float):
    # optimizer=torch.optim.AdamW([
        # {"params":model.new_embedding.parameters()},
        # # {"params":model.fc.parameters()}
        # {"params":model.classification.parameters()}
        # ],lr=lr)
    optimizer=torch.optim.AdamW(
        [
            {"params":model.mlm.parameters(),"lr":lr/10},
            {"params":model.lm_head.parameters(),"lr":lr/10},
            {"params":model.new_embedding.parameters()},
            {"params":model.classification.parameters()}
        ],
        lr=lr
    )
    return optimizer

def batch_forward_func(batch_data:Tuple[torch.Tensor,...],trainer):
    
    input_ids1,masks1,input_ids_for_new1,mask_pos1,labels1,\
            input_ids2,masks2,input_ids_for_new2,mask_pos2,labels2,batch_signals=batch_data
    input_ids1,\
    masks1,\
    input_ids_for_new1,\
    mask_pos1,\
    labels1,\
    input_ids2,\
    masks2,\
    input_ids_for_new2,\
    mask_pos2,\
    labels2,\
    batch_signals=\
        input_ids1.to(trainer.device),\
        masks1.to(trainer.device),\
        input_ids_for_new1.to(trainer.device),\
        mask_pos1.to(trainer.device),\
        labels1.to(trainer.device),\
        input_ids2.to(trainer.device),\
        masks2.to(trainer.device),\
        input_ids_for_new2.to(trainer.device),\
        mask_pos2.to(trainer.device),\
        labels2.to(trainer.device),\
        batch_signals.to(trainer.device)
    
    prediction=trainer.model(
        input_ids1,
        masks1,
        input_ids_for_new1,
        mask_pos1,
        input_ids2,
        masks2,
        input_ids_for_new2,
        mask_pos2,
    )

    return (labels1,labels2,batch_signals),prediction
    
def batch_cal_loss_func(labels:Tuple[torch.Tensor,...],preds:Tuple[torch.Tensor,...],trainer):
    labels1,labels2,batch_signals=labels
    pred1,pred2=preds
    
    return F.cross_entropy(pred1,labels1,reduction="mean")+F.cross_entropy(pred2,labels2,reduction="mean")


    # labels,mask_pos=labels
    # loss_fct = CrossEntropyLoss()
    # masked_lm_loss = loss_fct(preds.view(-1, preds.shape[2]), labels.view(-1))
    # print(torch.gather(labels,1,mask_pos.reshape([-1,1])))
    # preds_max=torch.argmax(preds,dim=2)
    # print(torch.gather(torch.argmax(preds,dim=2),1,mask_pos.reshape([-1,1])))
    # return masked_lm_loss

def batch_metrics_func(labels:Tuple[torch.Tensor,...],preds:Tuple[torch.Tensor,...],metrics:Dict[str,Number],trainer):
    labels1,labels2,batch_signals=labels
    pred1,pred2=preds
    batch_signals=batch_signals.cpu()
    
    cause_preds1=torch.argmax(pred1,dim=1).reshape([-1]).bool().long().cpu()
    cause_preds2=torch.argmax(pred2,dim=1).reshape([-1]).bool().long().cpu()
    cause_preds=torch.logical_or(cause_preds1,cause_preds2).long()
    causes=labels1.bool().long().cpu()
    signal_labels=(batch_signals*causes).long()

    batch_nosignals=torch.logical_and(torch.logical_not(batch_signals),causes.bool()).long()
    nosignal_labels=(batch_nosignals*causes).long()

    signal_preds=batch_signals*cause_preds
    nosignal_preds=batch_nosignals*cause_preds

    trainer.logger.info("labels: \n{}".format(causes))
    trainer.logger.info("preds: \n{}".format(cause_preds))
    
    precision=precision_score(causes.numpy(),cause_preds.numpy(),zero_division=0)
    recall=recall_score(causes.numpy(),cause_preds.numpy(),zero_division=0)
    f1=f1_score(causes.numpy(),cause_preds.numpy(),zero_division=0)

    signal_recall=recall_score(signal_labels.numpy(),signal_preds.numpy(),zero_division=0)
    nosignal_recall=recall_score(nosignal_labels.numpy(),nosignal_preds.numpy(),zero_division=0)
    batch_metrics={"precision":precision,"recall":recall,"f1":f1,"signal_recall":signal_recall,"nosignal_recall":nosignal_recall}
    if "labels" in metrics:
        metrics["labels"]=torch.cat([metrics["labels"],causes],dim=0)
    else:
        metrics["labels"]=causes
    if "preds" in metrics:
        metrics["preds"]=torch.cat([metrics["preds"],cause_preds],dim=0)
    else:
        metrics["preds"]=cause_preds
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

    return metrics,batch_metrics
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