#!/home/zhouheng/anaconda3/bin/python
# -*- encoding: utf-8 -*-
'''
@文件    :data_process.py
@时间    :2021/12/22 17:26:16
@作者    :周恒
@版本    :1.0
@说明    :大力出奇迹了属于是 
'''



from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Sized, Tuple, Union
from dataclasses import asdict, dataclass
import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler,Sampler
from transformers import AutoModel,AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.tokenization_utils_base import TruncationStrategy


"""N/A,cause,causedby"""
TNA="</tna>"
TCAUSE="</tcause>"
TCAUSEDBY="</tcaused_by>"

"""6个指示符"""

"""</t1></t2>指示event1 的位置"""
T1="</t1>"
T2="</t2>"

"""</t3></t4>指示event2 的位置"""
T3="</t3>"
T4="</t4>"

"""</t5></t6>指示signal的位置"""
T5="</t5>"
T6="</t6>"

"""四个mask的位置"""
T7="</t7>"
T8="</t8>"
T9="</t9>"
T10="</t10>"
T11="</t11>"
T12="</t12>"
T13="</t13>"
T14="</t14>"



"""bert input sequence max length """
MAX_LENTH=512

def ecimp_init_tokenizer(name_or_path:str,save_dir:str)->Tuple[Union[BertModel,RobertaModel],Union[BertTokenizer,RobertaTokenizer]]:
    """初始化分词器,加入17个特殊字符"""
    
    # mlm:AutoModel=AutoModel.from_pretrained(name_or_path)
    mlm_tokenizer=AutoTokenizer.from_pretrained(name_or_path)
    """分词器增加特殊字符"""
    special_tokens_dict={"additional_special_tokens":[TNA,TCAUSE,TCAUSEDBY,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14]}
    mlm_tokenizer.add_special_tokens(special_tokens_dict)

    # """预训练模型扩充token embedding,新增加的token embedding是随机初始化的"""
    # mlm.resize_token_embeddings(len(mlm_tokenizer))
    
    mlm_tokenizer.save_pretrained(save_dir)

    return  mlm_tokenizer

def make_prompted(
        tokens:List[str],
        event1_start_index:int,
        event1_end_index:int,
        event2_start_index:int,
        event2_end_index:int,
        tokenizer:Union[BertTokenizer,RobertaTokenizer],
        substr_token_start_index:int,
        substr_token_end_index:int,
        signal_start_token_index:int,
        signal_end_token_index:int,
        cause:int,
        use_event_prompt:bool,
        use_signal_prompt:bool,
        reuse:bool
                )->Optional[str]:
    """
    调用前确保prompt后完整的句子分词后长度不会超过512
    """
    new_tokens=tokens.copy()
    raw_source=tokens[event1_start_index:event1_end_index+1]
    raw_target=tokens[event2_start_index:event2_end_index+1]
    
    new_tokens[event1_start_index]=T1+" "+new_tokens[event1_start_index]
    new_tokens[event1_end_index]=new_tokens[event1_end_index]+" "+T2
    new_tokens[event2_start_index]=T3+" "+new_tokens[event2_start_index]
    new_tokens[event2_end_index]=new_tokens[event2_end_index]+" "+T4
    cause_str=TNA
    if cause==1:
        cause_str=TCAUSE
    elif cause==-1:
        cause_str=TCAUSEDBY
    if signal_start_token_index>=0:
        assert signal_end_token_index>=0
        new_tokens[signal_start_token_index]=T5+" "+new_tokens[signal_start_token_index]
        new_tokens[signal_end_token_index]=new_tokens[signal_end_token_index]+" "+T6
        
    prompted_tokens=\
        new_tokens[substr_token_start_index:substr_token_end_index+1]+\
        ["The event"]+\
        new_tokens[event1_start_index:event1_end_index+1]+\
        [T7+cause_str+T8,"the event"]+\
        new_tokens[event2_start_index:event2_end_index+1]+["."]
    if use_event_prompt:
        prompted_tokens=prompted_tokens+\
        [tokenizer.sep_token,"according",tokenizer.cls_token,", event"]+\
        new_tokens[event1_start_index:event1_end_index+1]+\
        [ TCAUSE if reuse else "cause","event",T9]+raw_target+\
        [T10,"or event"]+new_tokens[event1_start_index:event1_end_index+1]+\
        [TCAUSEDBY if reuse else "caused by","event",T11]+raw_target+\
        [T12,'.']

    if use_signal_prompt:
        if signal_start_token_index>=0:
            raw_signal=tokens[signal_start_token_index:signal_end_token_index+1]
            prompted_tokens.extend([tokenizer.sep_token,"event"]+\
                new_tokens[event1_start_index:event1_end_index+1]+\
                [TCAUSE if reuse else "cause","or",TCAUSEDBY if reuse else "caused by"]+new_tokens[event2_start_index:event2_end_index+1]+\
                ["by",T13]+raw_signal+\
                [T14,'.']
            )
        else:
            prompted_tokens.extend([tokenizer.sep_token,"event"]+\
                new_tokens[event1_start_index:event1_end_index+1]+\
                [TCAUSE if reuse else "cause","or",TCAUSEDBY if reuse else "caused by"]+new_tokens[event2_start_index:event2_end_index+1]+\
                ["by",T13,"nothing",T14,'.']
            )
    
    return " ".join(prompted_tokens)


@dataclass
class ECIMPInputfeature:
    prompted_sentence1:str 
    prompted_sentence2:str
    cause1:int
    cause2:int
    signal:bool #为了统计结果

def valid_data_preprocess(data:Dict[str,Any]):
    """验证集不能出现signal_start_index signal_end_index
    """
    for rel in data["relations"]:
        if rel["signal_start_index"]>=0:
            rel["signal"]=True
        rel["signal_start_index"]=-1
        rel["signal_end_index"]=-1
        
    return data

class PreprocessDataFunc:
    def __init__(
            self,
            use_event_prompt:bool,
            use_signal_prompt:bool,
            reuse:bool) -> None:
        self.use_event_prompt=use_event_prompt
        self.use_signal_prompt=use_signal_prompt
        self.reuse=reuse
    def __call__(self,data:Dict[str,Any],tokenizer:Union[BertTokenizer,RobertaTokenizer]) -> List[ECIMPInputfeature]:
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
            signal_start_index=rel["signal_start_index"]
            signal_end_index=rel["signal_end_index"]
            cause:int=rel["cause"]
            signal=False 
            if "signal" in rel:
                signal=rel["signal"]
            if signal_start_index>=0:
                signal=True
            substr_token_start_index=-1000
            substr_token_end_index=-1000
            
            event1_start_index,event1_end_index,event2_start_index,event2_end_index=\
                event2_start_index,event2_end_index,event1_start_index,event1_end_index
            cause=-cause
            if signal_start_index>=0:
                assert signal_end_index>=0
                substr_token_start_index=sentences[token_index2sentence_index[min(event1_start_index,event2_start_index,signal_start_index)]]["start"]
                substr_token_end_index=sentences[token_index2sentence_index[max(event1_end_index,event2_end_index,signal_end_index)]]["end"]
            else:
                substr_token_start_index=sentences[token_index2sentence_index[min(event1_start_index,event2_start_index)]]["start"]
                substr_token_end_index=sentences[token_index2sentence_index[max(event1_end_index,event2_end_index)]]["end"]
            if substr_token_end_index-substr_token_start_index>300:
                continue
            prompt1=make_prompted(
                tokens=tokens,
                event1_start_index=event1_end_index,
                event1_end_index=event1_end_index,
                event2_start_index=event2_start_index,
                event2_end_index=event2_end_index,
                tokenizer=tokenizer,
                substr_token_start_index=substr_token_start_index,
                substr_token_end_index=substr_token_end_index,
                signal_start_token_index=signal_start_index,
                signal_end_token_index=signal_end_index,
                cause=cause,
                use_event_prompt=self.use_event_prompt,
                use_signal_prompt=self.use_signal_prompt,
                reuse=self.reuse
            )
            event1_start_index,event1_end_index,event2_start_index,event2_end_index=\
                event2_start_index,event2_end_index,event1_start_index,event1_end_index
            cause=-cause
            # if prompt1==None:
                # continue
            prompt2=make_prompted(
                tokens=tokens,
                event1_start_index=event1_end_index,
                event1_end_index=event1_end_index,
                event2_start_index=event2_start_index,
                event2_end_index=event2_end_index,
                tokenizer=tokenizer,
                substr_token_start_index=substr_token_start_index,
                substr_token_end_index=substr_token_end_index,
                signal_start_token_index=signal_start_index,
                signal_end_token_index=signal_end_index,
                cause=cause,
                use_event_prompt=self.use_event_prompt,
                use_signal_prompt=self.use_signal_prompt,
                reuse=self.reuse
            )
            if prompt2==None:
                continue
            res.append(ECIMPInputfeature(prompt1,prompt2,-cause,cause,signal))
        return res

ecimp_preprocess_data=PreprocessDataFunc(True,True,True)

def process_one_prompt(
        token_ids:List[int],
        tokenizer:Union[BertTokenizer,RobertaTokenizer],
        use_event_prompt:bool,
        use_signal_prompt:bool
        ):
    size=len(token_ids)
    input_ids=torch.tensor(token_ids,dtype=torch.long)


    


    """要让这个input_ids_for_new里的(tna到t14)为(1到17),其他都是0"""
    input_ids_for_new=input_ids.clone()-tokenizer.vocab_size+1
    input_ids_for_new=torch.where(input_ids_for_new<0,torch.tensor(0,dtype=torch.long),input_ids_for_new)
    
    """当且仅当use_event_prompt=True 且 use_signal_prompt=True时 两个sep index才会有意义"""

    sep_event_index=torch.tensor([-1],dtype=torch.long)
    sep_signal_index=torch.tensor([-1],dtype=torch.long)
    if use_event_prompt and use_signal_prompt:
        sep_event_index[0]=token_ids.index(tokenizer.sep_token_id)
        sep_signal_index[0]=token_ids.index(tokenizer.sep_token_id,sep_event_index[0].item()+1)

    input_ids=torch.where(input_ids>=tokenizer.vocab_size,torch.tensor(0,dtype=torch.long),input_ids)
    """第1个mask 也就是base prompt部分"""
    
    t7index=token_ids.index(tokenizer.convert_tokens_to_ids(T7))
    t8index=token_ids.index(tokenizer.convert_tokens_to_ids(T8))
    """注意到roberta的 tokenizer 可能会分成 [...,</t7>,Ġ,</tna>,Ġ,</t8>,...]"""
    mask1_index=(t7index+t8index)//2
    cause=0
    if token_ids[mask1_index]==tokenizer.convert_tokens_to_ids(TCAUSE):
        cause=1
    elif token_ids[mask1_index]==tokenizer.convert_tokens_to_ids(TCAUSEDBY):
        cause=2
    input_ids[mask1_index]=tokenizer.mask_token_id
    """!!!input_ids_for_new 在 mask1处必须为0"""
    input_ids_for_new[mask1_index]=0


    """第2个和第3个mask 也就是event prompt部分"""
    mask_for_mask2=None
    mask_for_mask3=None
    if use_event_prompt:
        mask_for_mask2=torch.zeros(size,dtype=torch.float)
        mask_for_mask3=torch.zeros(size,dtype=torch.float)
        
        t9index=token_ids.index(tokenizer.convert_tokens_to_ids(T9))
        t10index=token_ids.index(tokenizer.convert_tokens_to_ids(T10))
        t11index=token_ids.index(tokenizer.convert_tokens_to_ids(T11))
        t12index=token_ids.index(tokenizer.convert_tokens_to_ids(T12))

        mask_for_mask2[t9index+1:t10index]=1.0
        input_ids[t9index+1:t10index]=tokenizer.mask_token_id
        mask_for_mask3[t11index+1:t12index]=1.0
        input_ids[t11index+1:t12index]=tokenizer.mask_token_id
    else:
        """没用就占位"""
        mask_for_mask2=torch.zeros([1,1],dtype=torch.float32)
        mask_for_mask3=torch.zeros([1,1],dtype=torch.float32)
    

    """第4个mask 对应signal prompt部分"""
    mask_for_mask4=None
    if use_signal_prompt:
        mask_for_mask4=torch.zeros(size,dtype=torch.float)
        t13index=token_ids.index(tokenizer.convert_tokens_to_ids(T13))
        t14index=token_ids.index(tokenizer.convert_tokens_to_ids(T14))
        mask_for_mask4[t13index+1:t14index]=1.0
        input_ids[t13index+1:t14index]=tokenizer.mask_token_id
    else:
        mask_for_mask4=torch.zeros([1,1],dtype=torch.float32)
    return \
        input_ids,                                    \
        torch.tensor([cause],dtype=torch.long),       \
        torch.tensor([mask1_index],dtype=torch.long), \
        mask_for_mask2,                               \
        mask_for_mask3,                               \
        mask_for_mask4,                               \
        input_ids_for_new,                            \
        sep_event_index,                              \
        sep_signal_index

        
class ECIMPCollator:
    def __init__(
        self,
        tokenizer:Union[BertTokenizer,RobertaTokenizer],
        use_event_prompt:bool,
        use_signal_prompt:bool
        ) -> None:
        self.use_event_prompt=use_event_prompt
        self.use_signal_prompt=use_signal_prompt
        self.tokenizer=tokenizer
        self.raw_vacob_size=self.tokenizer.vocab_size
        self.special_id_set:Set[int]=set([
            self.tokenizer.sep_token_id,\
            self.tokenizer.cls_token_id,
            self.tokenizer.mask_token_id
            ])
        for token in [TNA,TCAUSE,TCAUSEDBY,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14]:
            self.special_id_set.add(self.tokenizer.convert_tokens_to_ids(token))
    def _build_vocab_mask_for_sentence(self,token_ids:List[int]):
        mask=torch.zeros(self.tokenizer.vocab_size,dtype=torch.float32)
        for id_ in token_ids:
            if not (id_ in self.special_id_set):
                mask[id_]=1.0
            mask[self.tokenizer.convert_tokens_to_ids("nothing")]=1.0
        res=torch.where(mask>0,torch.tensor(0,dtype=torch.float),torch.tensor(float("-inf")))
        return res
    def __call__(self,data:List[ECIMPInputfeature]) ->Tuple[torch.Tensor,...]:
        batch_size=len(data)
        texts=[]
        for inputfeature in data:
            texts.append(inputfeature.prompted_sentence1)
            texts.append(inputfeature.prompted_sentence2)
        output=self.tokenizer(
            texts,
            padding=PaddingStrategy.LONGEST,
            truncation=True,
            max_length=512,
            return_attention_mask=True
            )    
        res=[]
        
        labels:torch.Tensor=output["input_ids"]
        
        masks=output["attention_mask"]
        
        for i in range(batch_size):
            try:
                vocab_mask=self._build_vocab_mask_for_sentence(labels[2*i])
                data1=process_one_prompt(labels[2*i],self.tokenizer,self.use_event_prompt,self.use_signal_prompt)
                data2=process_one_prompt(labels[2*i+1],self.tokenizer,self.use_event_prompt,self.use_signal_prompt)
                label1=torch.tensor(labels[2*i],dtype=torch.long)
                label1=torch.where(label1>=self.raw_vacob_size,torch.tensor(0),label1)
                label2=torch.tensor(labels[2*i+1],dtype=torch.long)
                label2=torch.where(label2>=self.raw_vacob_size,torch.tensor(0),label2)
                
                res.append((
                    data1,
                    data2,
                    label1,
                    label2,
                    torch.tensor(masks[2*i],dtype=torch.long),
                    torch.tensor(masks[2*i+1],dtype=torch.long),
                    vocab_mask))
            except Exception as ex:
                print(ex)
        

        batch_vocab_masks=torch.cat([d[6].reshape(1,-1) for d in res],dim=0)

        batch_label_1=torch.cat([d[2].reshape(1,-1) for d in res],dim=0)
        batch_mask_1=torch.cat([d[4].reshape(1,-1) for d in res],dim=0)
        batch_input_ids_1=torch.cat([d[0][0].reshape([1,-1]) for d in res],dim=0)
        batch_cause_1=torch.cat([d[0][1] for d in res],dim=0)
        batch_mask1_index_1=torch.cat([d[0][2] for d in res],dim=0)
        batch_mask_for_mask2_1=torch.cat([d[0][3].reshape([1,-1]) for d in res],dim=0)
        batch_mask_for_mask3_1=torch.cat([d[0][4].reshape([1,-1]) for d in res],dim=0)
        batch_mask_for_mask4_1=torch.cat([d[0][5].reshape([1,-1]) for d in res],dim=0)
        batch_input_ids_for_new_1=torch.cat([d[0][6].reshape([1,-1]) for d in res],dim=0)
        batch_sep_event_index_1=torch.cat([d[0][7].reshape([1,-1]) for d in res],dim=0)
        batch_sep_signal_index_1=torch.cat([d[0][8].reshape([1,-1]) for d in res],dim=0)

        batch_label_2=torch.cat([d[3].reshape(1,-1) for d in res],dim=0)
        batch_mask_2=torch.cat([d[5].reshape(1,-1) for d in res],dim=0)
        batch_input_ids_2=torch.cat([d[1][0].reshape([1,-1]) for d in res],dim=0)
        batch_cause_2=torch.cat([d[1][1] for d in res],dim=0)
        batch_mask1_index_2=torch.cat([d[1][2] for d in res],dim=0)
        batch_mask_for_mask2_2=torch.cat([d[1][3].reshape([1,-1]) for d in res],dim=0)
        batch_mask_for_mask3_2=torch.cat([d[1][4].reshape([1,-1]) for d in res],dim=0)
        batch_mask_for_mask4_2=torch.cat([d[1][5].reshape([1,-1]) for d in res],dim=0)
        batch_input_ids_for_new_2=torch.cat([d[1][6].reshape([1,-1]) for d in res],dim=0)
        batch_sep_event_index_2=torch.cat([d[1][7].reshape([1,-1]) for d in res],dim=0)
        batch_sep_signal_index_2=torch.cat([d[1][8].reshape([1,-1]) for d in res],dim=0)

        batch_signals=torch.tensor(list(map(lambda x:x.signal,data)),dtype=torch.long)
        return \
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
            batch_sep_signal_index_2, \
            batch_signals

class ECIMPSampler(Sampler):
    def __init__(self, data_source:List[ECIMPInputfeature],replacement:bool=True) -> None:
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
