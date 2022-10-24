from curses import raw
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid
from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
import logging
from ecimp import ECIMPModel
from transformers import AutoTokenizer,BertTokenizer, InputFeatures, RobertaTokenizer
from transformers.file_utils import PaddingStrategy
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)s:%(funcName)s] - %(message)s",datefmt= "%Y-%m-%d %H:%M:%S")
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
        reuse:bool,
        language="en"
                )->Optional[str]:
    """
    调用前确保prompt后完整的句子分词后长度不会超过512
    """
    if language=='en':
        new_tokens=tokens.copy()
        raw_source=tokens[event1_start_index:event1_end_index+1]
        raw_target=tokens[event2_start_index:event2_end_index+1]
        #如果是英文
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
    elif language=='zh':
        new_tokens=tokens.copy()
        raw_source=tokens[event1_start_index:event1_end_index+1]
        raw_target=tokens[event2_start_index:event2_end_index+1]
        new_tokens[event1_start_index]=T1+new_tokens[event1_start_index]
        new_tokens[event1_end_index]=new_tokens[event1_end_index]+T2
        new_tokens[event2_start_index]=T3+new_tokens[event2_start_index]
        new_tokens[event2_end_index]=new_tokens[event2_end_index]+T4
        cause_str=TNA
        if cause==1:
            cause_str=TCAUSE
        elif cause==-1:
            cause_str=TCAUSEDBY
        if signal_start_token_index>=0:
            assert signal_end_token_index>=0
            new_tokens[signal_start_token_index]=T5+new_tokens[signal_start_token_index]
            new_tokens[signal_end_token_index]=new_tokens[signal_end_token_index]+""+T6
            
        prompted_tokens=\
            new_tokens[substr_token_start_index:substr_token_end_index+1]+\
            ["事件"]+\
            new_tokens[event1_start_index:event1_end_index+1]+\
            [T7+cause_str+T8,"事件"]+\
            new_tokens[event2_start_index:event2_end_index+1]+["."]
        if use_event_prompt:
            prompted_tokens=prompted_tokens+\
            [tokenizer.sep_token,"根据",tokenizer.cls_token,"， 事件"]+\
            new_tokens[event1_start_index:event1_end_index+1]+\
            [ TCAUSE if reuse else "导致了","事件",T9]+raw_target+\
            [T10,"或者事件"]+new_tokens[event1_start_index:event1_end_index+1]+\
            [TCAUSEDBY if reuse else "由","事件",T11]+raw_target+\
            [T12,'。' if  reuse else "导致。"]

        if use_signal_prompt:
            if signal_start_token_index>=0:
                raw_signal=tokens[signal_start_token_index:signal_end_token_index+1]
                prompted_tokens.extend([tokenizer.sep_token,"事件"]+\
                    new_tokens[event1_start_index:event1_end_index+1]+\
                    [TCAUSE if reuse else "导致","或者",TCAUSEDBY if reuse else "由"]+new_tokens[event2_start_index:event2_end_index+1]+\
                    ["通过",T13]+raw_signal+\
                    [T14,'。']
                )
            else:
                prompted_tokens.extend([tokenizer.sep_token,"事件"]+\
                    new_tokens[event1_start_index:event1_end_index+1]+\
                    [TCAUSE if reuse else "导致","或者",TCAUSEDBY if reuse else "由"]+new_tokens[event2_start_index:event2_end_index+1]+\
                    ["导致" if reuse else "","通过",T13,"无",T14,'。']
                )
        
        return "".join(prompted_tokens)
    else:
        raise RuntimeError("语言设置错误,应为en或zh")

@dataclass
class ECIMPInputfeature:
    prompted_sentence1:str 
    prompted_sentence2:str
    cause1:int
    cause2:int
    sentence_id:uuid.UUID
    event1_start_index:int=0
    event1_end_index:int=0
    event2_start_index:int=0
    event2_end_index:int=0
    signal:bool=False

class PreprocessDataFunc:
    def __init__(
            self,
            language:str,
            use_event_prompt:bool,
            use_signal_prompt:bool,
            reuse:bool) -> None:
        self.language=language
        self.use_event_prompt=use_event_prompt
        self.use_signal_prompt=use_signal_prompt
        self.reuse=reuse
    def __call__(self,id_:str,data:Dict[str,Any],tokenizer:Union[BertTokenizer,RobertaTokenizer]) -> List[ECIMPInputfeature]:
        tokens=data["tokens"]
        token_index2sentence_index=data["token_index2sentence_index"]
        sentences=data["sentences"]
        relations:Dict[str,int]=data["relations"]
        res=[]
        if relations is None:
            return res
        for rel in relations:
            event1_start_index=rel["event1_start_index"]
            event1_end_index=rel["event1_end_index"]-1
            event2_start_index=rel["event2_start_index"]
            event2_end_index=rel["event2_end_index"]-1
            if (event1_start_index<=event2_start_index and event1_end_index>=event2_end_index)or\
                (event1_start_index>=event2_start_index and event1_end_index<=event2_end_index):
                continue
            if event1_start_index>event1_end_index or event2_start_index>event2_end_index :
                continue
            # signal_start_index=rel["signal_start_index"]
            # signal_end_index=rel["signal_end_index"]
            signal_start_index=-1
            signal_end_index=-1

            cause:int=0
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
                event1_start_index=event1_start_index,
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
                reuse=self.reuse,
                language=self.language
            )
            event1_start_index,event1_end_index,event2_start_index,event2_end_index=\
                event2_start_index,event2_end_index,event1_start_index,event1_end_index
            cause=-cause
            # if prompt1==None:
                # continue
            prompt2=make_prompted(
                tokens=tokens,
                event1_start_index=event1_start_index,
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
                reuse=self.reuse,
                language=self.language
            )
            if prompt2==None:
                continue
            res.append(ECIMPInputfeature(prompt1,prompt2,-cause,cause,id_,event1_start_index,event1_end_index,event2_start_index,event2_end_index))
        return res
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
    if use_event_prompt :
        sep_event_index[0]=token_ids.index(tokenizer.sep_token_id)
    if use_event_prompt and use_signal_prompt:
        sep_signal_index[0]=token_ids.index(tokenizer.sep_token_id,sep_event_index[0].item()+1)
    elif use_event_prompt:
        sep_signal_index[0]=token_ids.index(tokenizer.sep_token_id)

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
        
        labels:List=output["input_ids"]
        
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
@dataclass
class Event:
    start_index:int
    end_index:int
    

@dataclass
class SentenceInstance:
    tokens:List[str]
    events:List[Event]
    id:str=str(uuid.uuid4())
    
class EREPipeline:
    def __init__(self,language:str,mlm_dir,dump_path:str,tokenizer_dir:str,batch_size:int=4) -> None:
        """构造函数

        Args:
            language (str): 语言,en 或者 zh
            mlm_dir (_type_): 预训练语言模型的目录
            dump_path (str): ERE训练好的模型的文件路径
            tokenizer_dir (str): 预训练语言模型分词器的目录（中文应该设置成中文bert的目录）
            logger (logging.Logger): 日志
        """
        self.model:ECIMPModel=ECIMPModel(mlm_dir,True,False,False,True,True)
        with open(dump_path,"rb") as f:
            state=torch.load(f,map_location='cpu')
            self.model.load_state_dict(state)
        self.model.eval()
        self.tokenizer:BertTokenizer=BertTokenizer.from_pretrained(tokenizer_dir)
        self.data_preprocess_func:PreprocessDataFunc=PreprocessDataFunc(language,True,False,True)
        self.collator:ECIMPCollator=ECIMPCollator(self.tokenizer,True,False)
        self.logger=logging.getLogger(self.__class__.__name__)
        self.batch_size=batch_size
        # self.logger.
    def predict(self,instances:List[SentenceInstance],device:torch.device)->Dict[str,List[Dict[str,int]]]:
        """推理

        Args:
            instances (List[SentenceInstance]):
            device (torch.device): _description_

        Returns:
            Dict[str,List[Dict[str,int]]]: _description_
        """
        try:
            res:Dict[str,List[Dict[str,int]]]={}
            raw_data:Dict[str,Dict]={}
            for instance in instances:
                id_=instance.id
                res[id_]={}
                if not(id_ in raw_data):
                    raw_data[id_]={}
                    raw_data[id_]["tokens"]=instance.tokens
                    tokens_count=len(instance.tokens)
                    raw_data[id_]["token_index2sentence_index"]=[0 for x in range(tokens_count)]
                    raw_data[id_]["sentences"]=[{"start":0,"end":tokens_count-1}]
                    raw_data[id_]["events"]=instance.events
           
            raw_data1={}
            for k,v in raw_data.items():#过滤掉事件数量少于2的句子
                if len(v["events"])>1:
                    raw_data1[k]=v
            raw_data=raw_data1
            for k,v in raw_data.items():
                size=len(v['events'])
                events=v['events']
                v["relations"]=[]
                """事件之间两两构造事件对"""
                for i in range(size-1):
                    for j in range(i+1,size):
                        rel={}
                        e1:Event=events[i]
                        e2:Event=events[j]
                        rel["event1_start_index"]=e1.start_index
                        rel["event1_end_index"]=e1.end_index
                        rel['event2_start_index']=e2.start_index
                        rel["event2_end_index"]=e2.end_index
                        rel["signal_start_index"]=-1
                        rel["signal_end_index"]=-1
                        v["relations"].append(rel)
            
            inputfeatures:List[ECIMPInputfeature]=[]
            for k,v in raw_data.items():
                inputfeatures.extend(self.data_preprocess_func(k,v,self.tokenizer))
            if len(inputfeatures)>0:
                self.model=self.model.to(device=device)
            dataloader=DataLoader(dataset=inputfeatures,batch_size=self.batch_size,shuffle=False,num_workers=12)
            it=iter(dataloader)
            with tqdm(total=len(dataloader),ncols=80) as tqbar:
                with torch.no_grad():
                    while True:
                        batch_data:List[ECIMPInputfeature]=[]
                        
                        try:
                            batch_data=next(it)
                            
                        except StopIteration:
                            break
                        batch_tensors=self.collator(batch_data)
                        batch_tensors=tuple(list(map(lambda t:t.to(device),list(batch_tensors))))
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
                        batch_signals=batch_tensors

                        

                        preds=self.model(
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
                        mask1_output_1,\
                        lm_decoder_output_1,\
                        mask1_output_2,\
                        lm_decoder_output_2 = preds

                        

                        cause_preds = None
                        cause_preds1 = torch.argmax(mask1_output_1, dim=1).reshape([-1]).cpu()
                        cause_preds2 = torch.argmax(mask1_output_2, dim=1).reshape([-1]).cpu()
                        cause_preds=torch.logical_or(cause_preds1,cause_preds2).long()
                        
                        for idx,inputfeature in enumerate(batch_data):
                            pred=cause_preds[idx].item()
                            if pred!=0:
                                if not (id_ in res):
                                    res[id_]=[]
                                res[id_].append({
                                    "event1_start_index":inputfeature.event1_start_index,
                                    "event1_end_index":inputfeature.event1_end_index,
                                    "event1":"".join(raw_data[id_]["tokens"][inputfeature.event1_start_index:inputfeature.event1_end_index+1]),
                                    'event2_start_index':inputfeature.event2_start_index,
                                    "event2_end_index":inputfeature.event2_end_index,
                                    "event2":"".join(raw_data[id_]["tokens"][inputfeature.event2_start_index:inputfeature.event2_end_index+1])
                                })
                        tqbar.update(1)
            return res
        except Exception as ex:
            self.logger.warn(ex)
            return []
    