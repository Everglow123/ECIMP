'''
@文件    :data_process.py
@时间    :2021/11/21 15:31:23
@作者    :周恒
@版本    :1.0
@说明    :对于一个包含事件对的文本,将其构造为如下形式。
        w1,w2,…,</t1>e1</t2>,…,</t3>e2</t4>,…,wn.</t1>e1</t2>is the </t5> [MASK] </t6> of the </t3>e2</t4>.
        [MASK]处的目标词应为 [Cause_of]或者[Not_Cause_of]
'''



from typing import Any, Dict, Iterator, List, Optional, Sequence, Sized, Tuple, Union
from dataclasses import asdict, dataclass
from numpy.core.fromnumeric import shape
import torch
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

"""</t5></t6>指示Mask的位置"""
T5="</t5>"
T6="</t6>"

# """[Cause_of][Not_Cause_of]表示source事件是target事件的因果和非因果关系。"""
# CAUSEOF="[Cause_of]"
# NOTCAUSEOF="[Not_Cause_of]"

"""bert input sequence max length """
MAX_LENTH=512

def baseline_init_tokenizer(name_or_path:str,save_dir:str)->Tuple[Union[BertModel,RobertaModel],Union[BertTokenizer,RobertaTokenizer]]:
    """初始化分词器,加入8个特殊字符"""
    
    # mlm:AutoModel=AutoModel.from_pretrained(name_or_path)
    mlm_tokenizer=AutoTokenizer.from_pretrained(name_or_path)
    """分词器增加特殊字符"""
    special_tokens_dict={"additional_special_tokens":[T1,T2,T3,T4,T5,T6]}
    mlm_tokenizer.add_special_tokens(special_tokens_dict)

    # """预训练模型扩充token embedding,新增加的token embedding是随机初始化的"""
    # mlm.resize_token_embeddings(len(mlm_tokenizer))
    
    
    mlm_tokenizer.save_pretrained(save_dir)

    return  mlm_tokenizer


def make_prompted(
                tokens:List[str],
                source_start_index:int,
                source_end_index:int,
                target_start_index:int,
                target_end_index:int,
                mask_token:str,
                substr_token_start_index:int,
                substr_token_end_index:int,
                )->Optional[str]:
    """调用前确保prompt后完整的句子分词后长度不会超过512
    """
    
    
    new_tokens=tokens.copy()
    new_tokens[source_start_index]=T1+" "+new_tokens[source_start_index]
    new_tokens[source_end_index]=new_tokens[source_end_index]+" "+T2
    new_tokens[target_start_index]=T3+" "+new_tokens[target_start_index]
    new_tokens[target_end_index]=new_tokens[target_end_index]+" "+T4
    
    prompted_tokens=\
        new_tokens[substr_token_start_index:substr_token_end_index+1]+\
        ["The event"]+\
        new_tokens[source_start_index:source_end_index+1]+\
        ["has the",T5,mask_token,T6,"of the event"]+\
        new_tokens[target_start_index:target_end_index+1]+["."]
    return " ".join(prompted_tokens)

@dataclass
class BaselineInputfeature:
    cause1:int
    cause2:int
    prompted_sentence1:str
    prompted_sentence2:str
    signal:bool 

def valid_data_preprocess(data:Dict[str,Any]):
    
    for rel in data["relations"]:
        if rel["signal_start_index"]>=0:
            rel["signal"]=True
        rel["signal_start_index"]=-1
        rel["signal_end_index"]=-1
    
    return data
def baseline_preprocess_data(data:Dict[str,Any],tokenizer:Union[BertTokenizer,RobertaTokenizer])->List[BaselineInputfeature]:
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
        cause=rel["cause"]
        signal=False 
        if "signal" in rel:
            signal=rel["signal"]
        if signal_start_index>=0:
            signal=True
        substr_token_start_index=-1000
        substr_token_end_index=-1000
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
            source_start_index=event1_start_index,
            source_end_index=event1_end_index,
            target_start_index=event2_start_index,
            target_end_index=event2_end_index,
            mask_token=tokenizer.mask_token,
            substr_token_start_index=substr_token_start_index,
            substr_token_end_index=substr_token_end_index
        )
        # if prompt1==None:
        #     continue
        event1_start_index,event1_end_index,event2_start_index,event2_end_index=event2_start_index,event2_end_index,event1_start_index,event1_end_index
        prompt2=make_prompted(
            tokens=tokens,
            source_start_index=event1_start_index,
            source_end_index=event1_end_index,
            target_start_index=event2_start_index,
            target_end_index=event2_end_index,
            mask_token=tokenizer.mask_token,
            substr_token_start_index=substr_token_start_index,
            substr_token_end_index=substr_token_end_index
        )

        
        res.append(BaselineInputfeature(cause,-cause,prompt1,prompt2,signal))
    return res
    # rel_set=set()
    # for rel in relations:
    #     for rel in relations:
    #         source=rel["source"]
    #         target=rel["target"]
    #         rel_set.add(f"{source}-{target}")
    # res:Dict[str,BaselineInputfeature]={}
    # event_keys=list(events.keys())
    # events_count=len(events)

    # for i in range(events_count-1):
    #     for j in range(i+1,events_count):
    #         ei:str=event_keys[i]
    #         ej:str=event_keys[j]
    #         source_start_index=events[ei]["trigger_start_index"]
    #         source_end_index=events[ei]["trigger_end_index"]
    #         target_start_index=events[ej]["trigger_start_index"]
    #         target_end_index=events[ej]["trigger_end_index"]
    #         substr_token_start_index=sentences[token_index2sentence_index[min(source_start_index,target_start_index)]]["start"]
    #         substr_token_end_index=sentences[token_index2sentence_index[max(source_end_index,target_end_index)]]["end"]
    #         while abs(substr_token_end_index-substr_token_start_index) < 350:
    #             if token_index2sentence_index[substr_token_start_index] > 0:
    #                 substr_token_start_index = sentences[token_index2sentence_index[substr_token_start_index]-1]["start"]
         
    #             else:
    #                 break
    #         prompted1=make_prompted(
    #             tokens=tokens,
    #             source_start_index=source_start_index,
    #             source_end_index=source_end_index,
    #             target_start_index=target_start_index,
    #             target_end_index=target_end_index,
    #             mask_token=tokenizer.mask_token,
    #             substr_token_start_index=substr_token_start_index,
    #             substr_token_end_index=substr_token_end_index
    #             )
    #         if prompted1!=None:
    #             flag=False
    #             if "{0}-{1}".format(ei,ej) in rel_set or "{1}-{0}".format(ei,ej) in rel_set:
    #                 flag=True
    #             res["{0}-{1}".format(ei,ej)]=BaselineInputfeature(flag,prompted1)
    # return list(res.values())

class BaselineCollator:
    def __init__(self,tokenizer:Union[BertTokenizer,RobertaTokenizer]) -> None:
        self.tokenizer=tokenizer
        self.raw_vacob_size=self.tokenizer.vocab_size

    def __call__(self,data:List[BaselineInputfeature]) ->Tuple[torch.Tensor,...]:
        batch_size=len(data)
        text1,text2=[],[]
        batch_labels_1,batch_labels_2=[],[]
        for i in range(batch_size):
            text1.append(data[i].prompted_sentence1)
            text2.append(data[i].prompted_sentence2)
            if data[i].cause1==0:
                batch_labels_1.append(0)
                batch_labels_2.append(0)
            elif data[i].cause1==1:
                batch_labels_1.append(1)
                batch_labels_2.append(2)
            else:
                batch_labels_1.append(2)
                batch_labels_2.append(1)
        output1=self.tokenizer(
            text1,
            padding=PaddingStrategy.LONGEST,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_attention_mask=True,
            )    
        input_ids1:torch.Tensor=output1["input_ids"]
        mask_pos1=(input_ids1==self.tokenizer.mask_token_id).long().argmax(dim=1)
        masks1=output1["attention_mask"]
        input_ids_for_new1=input_ids1-self.raw_vacob_size+1
        input_ids_for_new1=torch.where(input_ids_for_new1<0,torch.tensor(0),input_ids_for_new1)
        input_ids1=torch.where(input_ids1>=(self.raw_vacob_size),torch.tensor(6666),input_ids1)
        labels1=torch.tensor(data=batch_labels_1,dtype=torch.long)
        
        output2=self.tokenizer(
            text2,
            padding=PaddingStrategy.LONGEST,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_attention_mask=True,
            )    
        input_ids2:torch.Tensor=output2["input_ids"]
        mask_pos2=(input_ids2==self.tokenizer.mask_token_id).long().argmax(dim=1)
        masks2=output2["attention_mask"]
        input_ids_for_new2=input_ids1-self.raw_vacob_size+1
        input_ids_for_new2=torch.where(input_ids_for_new2<0,torch.tensor(0),input_ids_for_new2)
        input_ids2=torch.where(input_ids2>=(self.raw_vacob_size),torch.tensor(6666),input_ids2)
        labels2=torch.tensor(data=batch_labels_2,dtype=torch.long)
        batch_signals=torch.tensor(list(map(lambda x:x.signal,data)),dtype=torch.long)
        return \
            input_ids1,masks1,input_ids_for_new1,mask_pos1,labels1,\
            input_ids2,masks2,input_ids_for_new2,mask_pos2,labels2,batch_signals

class BaselineSampler(Sampler):
    def __init__(self, data_source:List[BaselineInputfeature],replacement:bool=True) -> None:
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
        
    def __iter__(self)  :
        return self.impl.__iter__()

    def __len__(self) -> int:
        return len(self.impl)

if __name__=="__main__":
    import os
    THIS_FOLDER = os.path.split(os.path.realpath(__file__))[0]
    # roberta,tokenizer=init_MLM_and_tokenizer("roberta-base",os.path.join(THIS_FOLDER,"mlm"))
    # roberta=RobertaModel.from_pretrained(os.path.join(THIS_FOLDER,"mlm"))
    tokenizer=RobertaTokenizer.from_pretrained("roberta-base")
    import json
    f=open(os.path.join(THIS_FOLDER,"..","data","eventstory.json"),"r")
    js=json.load(f)
    f.close()
    # data=js[0]
    # tokens=data["tokens"]
    # rel=data["relations"][0]
    # source=data["events"][rel["source"]]
    # target=data["events"][rel["target"]]
    # res=make_prompted(tokens,source["trigger_start_index"],source["trigger_end_index"],
    #         target["trigger_start_index"],target["trigger_end_index"],tokenizer.mask_token
    #     )
    import random
    res=baseline_preprocess_data(random.choice(js),tokenizer)
    collate=BaselineCollator(tokenizer)
    collate(res)
    
    res=list(map(asdict,res))

    f=open("wtf.json","w")
    json.dump(res,f,ensure_ascii=False,indent=4)