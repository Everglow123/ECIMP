#!/home/zhouheng/anaconda3/bin/python
# -*- encoding: utf-8 -*-
'''
@文件    :data_process.py
@时间    :2021/11/27 17:18:36
@作者    :周恒
@版本    :1.0
@说明    :🤣🤣🤣
'''

from typing import Any, Dict, Iterator, List, Optional, Sequence, Sized, Tuple, Union
from dataclasses import asdict, dataclass
import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler,Sampler
from transformers import AutoModel,AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.tokenization_utils_base import TruncationStrategy

"""6个指示符"""

"""</t1></t2>指示source event 的位置"""
T1="</t1>"
T2="</t2>"

"""</t3></t4>指示target event 的位置"""
T3="</t3>"
T4="</t4>"





"""bert input sequence max length """
MAX_LENTH=512

def mlm_base_init_tokenizer(name_or_path:str,save_dir:str)->Union[BertTokenizer,RobertaTokenizer]:
    """初始化分词器，加入6个特殊字符"""
    
    # mlm:AutoModel=AutoModel.from_pretrained(name_or_path)
    mlm_tokenizer:Union[BertTokenizer,RobertaTokenizer]=AutoTokenizer.from_pretrained(name_or_path) # NO LINT
    """分词器增加6个字符"""

    mlm_tokenizer.add_special_tokens({'additional_special_tokens':[T1,T2,T3,T4]})

    
    mlm_tokenizer.save_pretrained(save_dir)

    return  mlm_tokenizer

@dataclass
class MlmBaseInputfeature:
    cause1:int
    cause2:int
    format_str1:str
    format_str2:str
def make_format_str(
        tokens:List[str],
        source_start_index:int,
        source_end_index:int,
        target_start_index:int,
        target_end_index:int,
        substr_token_start_index:int,
        substr_token_end_index:int
    )->Optional[str]:

    new_tokens=tokens.copy()
    new_tokens[source_start_index]=T1+" "+new_tokens[source_start_index]
    new_tokens[source_end_index]=new_tokens[source_end_index]+" "+T2
    new_tokens[target_start_index]=T3+" "+new_tokens[target_start_index]
    new_tokens[target_end_index]=new_tokens[target_end_index]+" "+T4
    
    new_tokens=new_tokens[substr_token_start_index:substr_token_end_index+1]
    return " ".join(new_tokens)
def valid_data_preprocess(data:Dict[str,Any]):
    
    for rel in data["relations"]:
        rel["signal_start_index"]=-1
        rel["signal_end_index"]=-1
    
    return data
def mlm_base_preprocess_data(data:Dict[str,Any],tokenizer:Union[BertTokenizer,RobertaTokenizer])->List[MlmBaseInputfeature]:
    tokens=data["tokens"]
    token_index2sentence_index=data["token_index2sentence_index"]
    sentences=data["sentences"]
    relations:Dict[str,int]=data["relations"]
    res=[]
    for rel in relations:
        event1_start_index=rel["event1_start_index"]
        event1_end_index=rel["event1_end_index"]
        event2_start_index=rel["event2_start_index"]
        event2_end_index=rel["event2_end_index"]
        # signal_start_index=rel["signal_start_index"]
        # signal_end_index=rel["signal_end_index"]
        cause=rel["cause"]
        substr_token_start_index=-114514
        substr_token_end_index=-114514
        # if signal_start_index>=0:
        #     assert signal_end_index>=0
        #     substr_token_start_index=sentences[token_index2sentence_index[min(event1_start_index,event2_start_index,signal_start_index)]]["start"]
        #     substr_token_end_index=sentences[token_index2sentence_index[max(event1_end_index,event2_end_index,signal_end_index)]]["end"]
        # else:
        substr_token_start_index=sentences[token_index2sentence_index[min(event1_start_index,event2_start_index)]]["start"]
        substr_token_end_index=sentences[token_index2sentence_index[max(event1_end_index,event2_end_index)]]["end"]
        if substr_token_end_index-substr_token_start_index > 300:
            continue
        format_str1=make_format_str(
            tokens=tokens,
            source_start_index=event1_start_index,
            source_end_index=event1_end_index,
            target_start_index=event2_start_index,
            target_end_index=event2_end_index,
            substr_token_start_index=substr_token_start_index,
            substr_token_end_index=substr_token_end_index
            
        )
        if format_str1==None:
            continue
        event1_start_index,event1_end_index,event2_start_index,event2_end_index=event2_start_index,event2_end_index,event1_start_index,event1_end_index
        format_str2=make_format_str(
            tokens=tokens,
            source_start_index=event1_end_index,
            source_end_index=event1_end_index,
            target_start_index=event2_start_index,
            target_end_index=event2_end_index,
            substr_token_start_index=substr_token_start_index,
            substr_token_end_index=substr_token_end_index
            
        )
        res.append(MlmBaseInputfeature(cause1=cause,cause2=-cause,format_str1=format_str1,format_str2=format_str2))
    return res

    
def process_one_sentence(
        token_ids:List[int],\
        attention_mask:List[int],\
        tokenizer:Union[BertTokenizer,RobertaTokenizer]
    )->Tuple[torch.Tensor,...]:
    """
    🤣🤣🤣\n
    Args:
        inputfeature (LegacyInputfeature): 
        token_ids (List[int]):经过batch encode 包括 max length padding 后的 token ids
    Returns:
        token_ids,
        attention_mask,
        source_mask,
        target_mask
    """    
    
    source_start_index:int=token_ids.index(tokenizer.convert_tokens_to_ids(T1))+1
    source_end_index:int=token_ids.index(tokenizer.convert_tokens_to_ids(T2))-1
    
    target_start_index:int=token_ids.index(tokenizer.convert_tokens_to_ids(T3))+1
    target_end_index:int=token_ids.index(tokenizer.convert_tokens_to_ids(T4))-1



    source_mask:np.ndarray=np.zeros(len(token_ids),dtype=np.int)
    target_mask:np.ndarray=np.zeros(len(token_ids),dtype=np.int)
    source_mask[source_start_index:source_end_index+1]=1
    target_mask[target_start_index:target_end_index+1]=1
    
    signal_mask:np.ndarray=np.zeros(len(token_ids),dtype=np.int)

 
    
    """同步删除t1,t2,t3,t4"""
    for token in [T1,T2,T3,T4]:
        index=token_ids.index(tokenizer.convert_tokens_to_ids(token))
        token_ids.pop(index)
        attention_mask.pop(index)
        source_mask=np.delete(source_mask,index)
        target_mask=np.delete(target_mask,index)
        signal_mask=np.delete(signal_mask,index)

    return  torch.tensor(token_ids,dtype=torch.long),\
            torch.tensor(attention_mask,dtype=torch.long),\
            torch.from_numpy(source_mask),\
            torch.from_numpy(target_mask),\


    
class MlmBaseCollator:
    def __init__(self,tokenizer:Union[BertTokenizer,RobertaTokenizer]) -> None:
        self.tokenizer=tokenizer
        # self.raw_vacob_size=self.tokenizer.vocab_size
    
    def __call__(self,data:List[MlmBaseInputfeature]) -> Tuple[torch.Tensor,...]:
        batch_size=len(data)
        batch_labels1=[]
        batch_labels2=[]
        texts=[]
        for inputfeature in data:
            texts.append(inputfeature.format_str1)
            texts.append(inputfeature.format_str2)
            if inputfeature.cause1==0:
                batch_labels1.append(0)
                batch_labels2.append(0)
            elif inputfeature.cause1==1:
                batch_labels1.append(1)
                batch_labels2.append(2)
            else:
                batch_labels1.append(2)
                batch_labels2.append(1)
        
        batch_labels_1=torch.tensor(batch_labels1,dtype=torch.long)
        batch_labels_2=torch.tensor(batch_labels2,dtype=torch.long)


        batch_encoded:List[List[int]]=self.tokenizer(texts,padding=PaddingStrategy.LONGEST,\
            truncation=True,max_length=512,return_attention_mask=True)
        batch_token_ids,\
        batch_attention_mask=\
            batch_encoded["input_ids"],\
            batch_encoded["attention_mask"]

        res=[]
        for i in range(batch_size):
            res.append(
                (
                    process_one_sentence(
                        token_ids=batch_token_ids[2*i],attention_mask=batch_attention_mask[2*i],tokenizer=self.tokenizer),
                    process_one_sentence(
                        token_ids=batch_token_ids[2*i+1],attention_mask=batch_attention_mask[2*i+1],tokenizer=self.tokenizer)
                )
            )

        batch_token_ids_1:torch.Tensor=torch.cat([d[0][0].reshape(1,-1) for d in res],dim=0)
        batch_attention_mask_1:torch.Tensor=torch.cat([d[0][1].reshape(1,-1) for d in res],dim=0)
        batch_source_mask_1:torch.Tensor=torch.cat([d[0][2].reshape(1,-1) for d in res],dim=0)
        batch_target_mask_1:torch.Tensor=torch.cat([d[0][3].reshape(1,-1) for d in res],dim=0)

        batch_token_ids_2:torch.Tensor=torch.cat([d[1][0].reshape(1,-1) for d in res],dim=0)
        batch_attention_mask_2:torch.Tensor=torch.cat([d[1][1].reshape(1,-1) for d in res],dim=0)
        batch_source_mask_2:torch.Tensor=torch.cat([d[1][2].reshape(1,-1) for d in res],dim=0)
        batch_target_mask_2:torch.Tensor=torch.cat([d[1][3].reshape(1,-1) for d in res],dim=0)
        

        return  batch_token_ids_1,\
                batch_attention_mask_1,\
                batch_source_mask_1,\
                batch_target_mask_1,\
                batch_labels_1,\
                batch_token_ids_2,\
                batch_attention_mask_2,\
                batch_source_mask_2,\
                batch_target_mask_2,\
                batch_labels_2

class MlmBaseSampler(Sampler):
    def __init__(self, data_source:List[MlmBaseInputfeature],replacement:bool=True) -> None:
        super().__init__(data_source=data_source)
        positive_index:List[int]=[]
        negative_index:List[int]=[]
        
        for index,inputfeature in enumerate(data_source):
            if inputfeature.cause1:
                positive_index.append(index)
            else:
                negative_index.append(index)
        positive_weight=(len(negative_index)/len(positive_index))/7
        weights=torch.ones([len(data_source)],dtype=torch.float32)
        positive_indices=torch.tensor(positive_index,dtype=torch.long)
        weights[positive_indices]=positive_weight

        self.impl=WeightedRandomSampler(weights=weights,num_samples=len(data_source),replacement=replacement)
        
    def __iter__(self):
        return self.impl.__iter__()

    def __len__(self) -> int:
        return len(self.impl)

if __name__=='__main__':
    mlm_type="roberta-base"
    tokenizer=mlm_base_init_tokenizer(mlm_type,"legacy/mlm")
    raw_data:List[Dict[str,Any]]=[]
    import json
    with open("data/timebank.json","r") as f:
        raw_data=json.load(f)
    inputfeatures:List[MlmBaseInputfeature]=[]
    for data in raw_data:
        inputfeatures.extend(mlm_base_preprocess_data(data,tokenizer))
    collate_fn=MlmBaseCollator(tokenizer)
    index=11451
    for i,inputfeature in enumerate(inputfeatures):
        if T5 in inputfeature.format_str1:
            index=i 
            break
    batch_token_ids,\
    batch_attention_mask,\
    batch_source_mask,\
    batch_target_mask,\
    batch_signal_mask,\
    batch_source_relative_position,\
    batch_target_relative_position,\
    batch_labels=collate_fn(inputfeatures[index:index+3])
    batch_tokens=[tokenizer.convert_ids_to_tokens(token_ids) for token_ids in batch_token_ids.tolist()]
    res={
        "batch_token_ids":batch_token_ids.tolist(),
        "batch_tokens":batch_tokens,
        "batch_attention_mask":batch_attention_mask.tolist(),
        "batch_source_mask":batch_source_mask.tolist(),
        "batch_target_mask":batch_target_mask.tolist(),
        "batch_signal_mask":batch_signal_mask.tolist(),
        "batch_source_relative_position":batch_source_relative_position.tolist(),
        "batch_target_relative_position":batch_target_relative_position.tolist()
    }
    with open("legacy_batch_demo.json","w") as f:
        json.dump(res,f,indent=4,ensure_ascii=False)
       
