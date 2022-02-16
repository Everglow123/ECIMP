#!/home/zhouheng/anaconda3/bin/python
# -*- encoding: utf-8 -*-
'''
@文件    :legacy_model.py
@时间    :2021/11/27 17:18:26
@作者    :周恒
@版本    :1.0
@说明    :mlm->linear
'''
from typing import Dict, Tuple, Union
import torch
import torch.nn.functional as F
from transformers.models.bert import BertModel,BertConfig
from transformers.models.roberta import RobertaModel,RobertaConfig
from transformers import AutoModel,AutoConfig
from sklearn.metrics import recall_score,f1_score,precision_score

class MlmBaseModel(torch.nn.Module):
    def __init__(self,
        mlm_name_or_path,
    ):
        super().__init__()
        
        self.mlm:Union[BertModel,RobertaModel]=AutoModel.from_pretrained(mlm_name_or_path)
        self.mlm_config:Union[BertConfig,RobertaConfig]=AutoConfig.from_pretrained(mlm_name_or_path)
        self.hidden_dim:int=self.mlm_config.hidden_size
        
        self.dropout=torch.nn.Dropout()
        # self.attention=torch.nn.MultiheadAttention(batch_first=True,embed_dim=self.hidden_dim + 2 * position_embedding_dim,num_heads=8)
        # self.source_position_embedding=torch.nn.Embedding(201,position_embedding_dim)
        # self.target_position_embedding=torch.nn.Embedding(201,position_embedding_dim)
        # self.transformer_encoder=torch.nn.TransformerEncoder(
        #     torch.nn.TransformerEncoderLayer(self.hidden_dim+2*position_embedding_dim,8,batch_first=True),6,
        #     norm=torch.nn.LayerNorm(self.hidden_dim+2*position_embedding_dim)
        # )
        # self.fc=torch.nn.Linear(self.hidden_dim * 2 + 4 * position_embedding_dim,self.hidden_dim + 2 * position_embedding_dim)
        self.classification=torch.nn.Linear(self.hidden_dim * 2 ,3,bias=False)
        # self.classification
    def forward_one_sentence(
                self,
                batch_token_ids:torch.Tensor, #batch_size * sequence_size 
                batch_attention_mask:torch.Tensor, #batch_size * sequence_size 
                batch_source_mask:torch.Tensor, #batch_size * sequence_size 
                batch_target_mask:torch.Tensor,#batch_size * sequence_size 
            ):
        """
        """
        batch_size,sequence_size=batch_token_ids.shape

        """batch_size * sequence_size * hidden_dim"""
        mlm_output=self.mlm(batch_token_ids,attention_mask=batch_attention_mask)
        sequence_embedding=mlm_output[0]
        batch_source_mask_repeated= batch_source_mask.reshape([batch_size,sequence_size,1])\
            .repeat(1,1,self.hidden_dim)
        source_feature=sequence_embedding*batch_source_mask_repeated
        source_feature=source_feature.sum(dim=1).reshape(batch_size,self.hidden_dim)
        batch_source_for_div=batch_source_mask.sum(dim=-1).reshape([batch_size,1])\
            .repeat(1,self.hidden_dim)
        source_feature=source_feature/batch_source_for_div
        
        batch_target_mask_repeated= batch_target_mask.reshape([batch_size,sequence_size,1])\
            .repeat(1,1,self.hidden_dim)
        target_feature=sequence_embedding*batch_target_mask_repeated
        target_feature=target_feature.sum(dim=1).reshape(batch_size,self.hidden_dim)
        batch_target_for_div=batch_target_mask.sum(dim=-1).reshape([batch_size,1])\
            .repeat(1,self.hidden_dim)
        target_feature=target_feature/batch_target_for_div
        
        feature=torch.cat([source_feature,target_feature],dim=1)
        if self.training:
            feature=self.dropout(feature)
        return self.classification(feature)
        # """batch_size * sequence_size * position_embedding_dim"""
        # source_position_embedding=self.source_position_embedding(batch_source_relative_position+100)
        # target_position_embedding=self.target_position_embedding(batch_target_relative_position+100)
        
        # """batch_size * sequence_size * (hidden_dim + 2 * position_embedding_dim)"""
        # features=torch.cat([sequence_embedding,source_position_embedding,target_position_embedding],dim=-1)

        # """batch_size * sequence_size * (hidden_dim + 2 * position_embedding_dim)"""
        # encoder_output:torch.Tensor=self.transformer_encoder(features)
        # encoder_output=F.dropout(encoder_output,0.5)
        # """build source feature"""
        # batch_source_mask_repeated= batch_source_mask.reshape([batch_size,sequence_size,1])\
        #     .repeat(1,1,(self.position_embedding_dim*2+self.hidden_dim))

        # """batch_size * sequence_size * (hidden_dim + 2 * position_embedding_dim)""" 
        # source_feature= encoder_output*batch_source_mask_repeated

        # """batch_size * (hidden_dim + 2 * position_embedding_dim)""" 
        # source_feature=source_feature.sum(dim=1)
        # batch_source_for_div=batch_source_mask.sum(dim=-1).reshape([batch_size,1])\
        #     .repeat(1,self.position_embedding_dim*2+self.hidden_dim)
        # source_feature=source_feature/batch_source_for_div
        
        # """build target feature"""
        # batch_target_mask_repeated= batch_target_mask.reshape([batch_size,sequence_size,1])\
        #     .repeat(1,1,(self.position_embedding_dim*2+self.hidden_dim))

        # """batch_size * sequence_size * (hidden_dim + 2 * position_embedding_dim)""" 
        # target_feature= encoder_output*batch_target_mask_repeated

        # """batch_size * (hidden_dim + 2 * position_embedding_dim)""" 
        # target_feature=target_feature.sum(dim=1)
        # batch_target_for_div=batch_target_mask.sum(dim=-1).reshape([batch_size,1])\
        #     .repeat(1,self.position_embedding_dim*2+self.hidden_dim)
        # target_feature=target_feature/batch_target_for_div
        
        # """batch_size * (hidden_dim * 2 + 4 * position_embedding_dim)""" 
        # event_feature=torch.cat([source_feature,target_feature],dim=-1)

        # """batch_size * (hidden_dim + 2 * position_embedding_dim)""" 
        # event_query=self.fc(event_feature).reshape([batch_size,1,-1])
        # event_query=torch.tanh(event_query)
        # """batch_size * (hidden_dim + 2 * position_embedding_dim)""" 
        # attention_output=self.attention(query=event_query,key=encoder_output,value=encoder_output)[0].reshape(batch_size,-1)

        # """batch_size * (hidden_dim * 3 + 6 * position_embedding_dim)""" 
        # final_feature=torch.cat([event_feature,attention_output],dim=-1)

        # pred=self.classification(final_feature)
        # return pred

    def forward(
            self,
            batch_token_ids_1:torch.Tensor,
            batch_attention_mask_1:torch.Tensor,
            batch_source_mask_1:torch.Tensor,
            batch_target_mask_1:torch.Tensor,
            batch_token_ids_2:torch.Tensor,
            batch_attention_mask_2:torch.Tensor,
            batch_source_mask_2:torch.Tensor,
            batch_target_mask_2:torch.Tensor,
        ):
        output1=self.forward_one_sentence(
            batch_token_ids_1,\
            batch_attention_mask_1,\
            batch_source_mask_1,\
            batch_target_mask_1
        )
        output2=self.forward_one_sentence(
            batch_token_ids_2,\
            batch_attention_mask_2,\
            batch_source_mask_2,\
            batch_target_mask_2
        )
        return output1,output2

def get_optimizer(model:MlmBaseModel,lr:float):
    optimizer=torch.optim.AdamW([
        {"params":model.mlm.parameters(),"lr":lr/10},
        {"params":model.classification.parameters()}
        ],lr=lr)
    # optimizer=torch.optim.AdamW(
    #     [
    #         {"params":model.mlm.parameters(),"lr":lr/10},
    #         {"params":model.lm_head.parameters(),"lr":lr/10},
    #         {"params":model.new_embedding.parameters()},
    #         {"params":model.classification.parameters()}
    #     ],
    #     lr=lr
    # )
    return optimizer

def batch_forward_func(batch_data:Tuple[torch.Tensor,...],trainer):
    
    batch_token_ids_1,\
    batch_attention_mask_1,\
    batch_source_mask_1,\
    batch_target_mask_1,\
    batch_labels_1,\
    batch_token_ids_2,\
    batch_attention_mask_2,\
    batch_source_mask_2,\
    batch_target_mask_2,\
    batch_labels_2=batch_data


    batch_token_ids_1,\
    batch_attention_mask_1,\
    batch_source_mask_1,\
    batch_target_mask_1,\
    batch_labels_1,\
    batch_token_ids_2,\
    batch_attention_mask_2,\
    batch_source_mask_2,\
    batch_target_mask_2,\
    batch_labels_2=\
        batch_token_ids_1.to(trainer.device),\
        batch_attention_mask_1.to(trainer.device),\
        batch_source_mask_1.to(trainer.device),\
        batch_target_mask_1.to(trainer.device),\
        batch_labels_1.to(trainer.device),\
        batch_token_ids_2.to(trainer.device),\
        batch_attention_mask_2.to(trainer.device),\
        batch_source_mask_2.to(trainer.device),\
        batch_target_mask_2.to(trainer.device),\
        batch_labels_2.to(trainer.device)
        

    prediction=trainer.model(
        batch_token_ids_1,\
        batch_attention_mask_1,\
        batch_source_mask_1,\
        batch_target_mask_1,\
        batch_token_ids_2,\
        batch_attention_mask_2,\
        batch_source_mask_2,\
        batch_target_mask_2,\
    )

    return (batch_labels_1,batch_labels_2), prediction
    
def batch_cal_loss_func(labels:Tuple[torch.Tensor,...],preds:Tuple[torch.Tensor,...],trainer):
    batch_labels_1,batch_labels_2=labels
    pred1,pred2=preds
    batch_size,class_n=pred1.shape

    loss1=F.cross_entropy(pred1,batch_labels_1,reduction="mean")
    loss2=F.cross_entropy(pred2,batch_labels_2,reduction="mean")

    return loss1+loss2

def batch_metrics_func(labels:Tuple[torch.Tensor,...],preds:Tuple[torch.Tensor,...],metrics:Dict[str,float],trainer):
    batch_labels_1,batch_labels_2=labels
    pred1,pred2=preds
    cause_preds1=torch.argmax(pred1,dim=1).reshape([-1]).bool().long().cpu()
    cause_preds2=torch.argmax(pred2,dim=1).reshape([-1]).bool().long().cpu()

    cause_preds=torch.logical_or(cause_preds1,cause_preds2).long()

    causes=batch_labels_1.reshape([-1]).bool().long().cpu()

    trainer.logger.info("labels: {}".format(causes))
    trainer.logger.info("preds: {}".format(cause_preds))

    precision=precision_score(causes.numpy(),cause_preds.numpy(),zero_division=0)
    recall=recall_score(causes.numpy(),cause_preds.numpy(),zero_division=0)
    f1=f1_score(causes.numpy(),cause_preds.numpy(),zero_division=0)

    batch_metrics={"precision":precision,"recall":recall,"f1":f1}
    if "labels" in metrics:
        metrics["labels"]=torch.cat([metrics["labels"],causes],dim=0)
    else:
        metrics["labels"]=causes
    if "preds" in metrics:
        metrics["preds"]=torch.cat([metrics["preds"],cause_preds],dim=0)
    else:
        metrics["preds"]=cause_preds
    return metrics,batch_metrics
def metrics_cal_func(metrics: Dict[str, torch.Tensor]):
    causes=metrics["labels"]
    cause_preds=metrics["preds"]
    precision=precision_score(causes.numpy(),cause_preds.numpy())
    recall=recall_score(causes.numpy(),cause_preds.numpy())
    f1=f1_score(causes.numpy(),cause_preds.numpy())
    res={"precision":precision,"recall":recall,"f1":f1}
    return res


        