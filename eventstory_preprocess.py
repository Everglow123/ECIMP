#!/home/zhouheng/anaconda3/bin/python
# -*- encoding: utf-8 -*-
'''
@文件    :eventstory_preprocess.py
@时间    :2021/11/17 21:12:38
@作者    :周恒
@版本    :1.0
@说明    :eventstory 只考虑infra-sentence的事件对
'''


from genericpath import exists
import json
import os
from typing import Dict, List, Optional, Set, Tuple, Union
from bs4 import BeautifulSoup
import bs4
import dataclasses
from bs4.element import Tag
 
@dataclasses.dataclass
class Event:
    token_indexs:List[int]
    token_tids:List[int]
    group_id:int=-100
    mid:int=-100
    def __init__(self,s:str) -> None:
        self.token_indexs=[]
        self.token_tids=[]
        if "_" in s:
            for numstr in s.split("_"):
                self.token_tids.append(int(numstr))
        else:
            self.token_tids.append(int(s))
        for mid in self.token_tids:
            self.token_indexs.append(mid-1)
    def key(self):
        return "_".join(list(map(lambda x:str(x),self.token_tids)))
@dataclasses.dataclass
class Rel:
    source_group_id:int=-100
    target_group_id:int=-100
    signal_token_start_index:int=-100
    signal_token_end_index:int=-100
@dataclasses.dataclass
class Group:
    
    events:List[Event]
    rels:List[Rel]
    id:int=-100
def read_chain(path:str):
    """
    获取共指事件集合
    Args:
        path (str): [description]

    Returns:
        List[Set[str]]:例如 [{"34","57"},{"64","130","100","38_39"},...]
    """
    if not os.path.exists(path):
        return [],[],{}
    event_token_groups:List[Set[str]]=[]
    group_rels:List[Tuple[int,int]]=[]
    with open(path,"r") as f:
        added_set_str2group_index:Dict[str,int]={}
        event_token_groups=[]
        for line in f:
            line1=line.strip()
            if len(line1)>2:
                splited=line1.split('\t')
                assert len(splited)==3
                tokens1=splited[0].split(" ")
                tokens1.sort()
                if not (" ".join(tokens1) in added_set_str2group_index):
                    event_token_groups.append(set(tokens1))
                    added_set_str2group_index[(" ".join(tokens1))]=len(event_token_groups)-1
                index1=added_set_str2group_index[" ".join(tokens1)]
                if splited[0]==splited[1]:
                    continue
                tokens2=splited[1].split(" ")
                tokens2.sort()
                if not (" ".join(tokens2) in added_set_str2group_index):
                    event_token_groups.append(set(tokens2))
                    added_set_str2group_index[(" ".join(tokens2))]=len(event_token_groups)-1
                index2=added_set_str2group_index[" ".join(tokens2)]
                if splited[2] == "PRECONDITION":
                    group_rels.append((index1,index2))
                elif splited[2]== "FALLING_ACTION":
                    group_rels.append((index2,index1))
                if index1==index2:
                    continue
    res_size=len(event_token_groups)
    for i in range(res_size-1):
        for j in range(i+1,res_size):
            """一个事件不能出现在两个集合里"""
            assert len(event_token_groups[i]&event_token_groups[j])==0

    groups:List[Group]=[]
    rels:List[Rel]=[]
    for i in range(len(event_token_groups)):
        group=Group([],[],i)
        for s in event_token_groups[i]:
            event=Event(s)
            event.group_id=i
            group.events.append(event)
        groups.append(group)
    for source_group_index,target_group_index in group_rels:
        rel=Rel(source_group_index,target_group_index)
        rels.append(rel)
    key2events:Dict[str,Event]={}
    for group in groups:
        for event in group.events:
            key2events[event.key()]=event
    return groups,rels,key2events



def process_file(path: str,extended:str) -> Optional[Dict[str, Union[str, Dict, List[str]]]]:
    """ 
    
    """
    print(path)
    f = open(path, "r")
    text = f.read()
    f.close()
    bs = BeautifulSoup(text, 'lxml')
    
    res = {}

    
    event_token_groups,group_rels,key2events=read_chain(extended)
    
    event_id2group_index={}
    mid2group_index={}

    mid2event: Dict[int, Event] = {}
    id2token: Dict[str, bs4.element.Tag] = {}
    tokens=[]
    tid2tokens=[]
    id2event={}
    sentences=[]
    token_index2sentence_index:List[int]=[]
    for token_e in bs.findAll("token"):
        t_id=token_e["t_id"]
        sentence=token_e["sentence"]
        id2token[t_id] = token_e
        tokens.append(token_e.text)
        token_index2sentence_index.append(int(sentence))
    for index,sentence in enumerate(token_index2sentence_index):
        if len(sentences)<=sentence:
            sentences.append({"start":index,"end":-1})
        else:
            sentences[-1]["end"]=index
    res['tokens'] = tokens
    res["token_index2sentence_index"]=token_index2sentence_index
    res["sentences"]=sentences
    # events = {}

    # for index, event_e in enumerate(list(filter(lambda x: type(x) != bs4.element.NavigableString and len(x.findAll("token_anchor")) > 0 , bs.findAll("markables")[0].children))):
    #     id2event[event_e["m_id"]] = event_e
    for index, event_e in enumerate(list(filter(lambda x: type(x) != bs4.element.NavigableString and len(x.findAll("token_anchor")) > 0 and (x.name.startswith("action") or x.name.startswith("neg_action")), bs.findAll("markables")[0].children))):
        # print(type(event_e))
        event_token_ids=[]
        for token in event_e.findAll("token_anchor"):
            event_token_ids.append(token["t_id"])
        key="_".join(event_token_ids)
        m_id=int(event_e["m_id"])
        if key in key2events:
            key2events[key].mid=m_id
            mid2event[m_id]=key2events[key]
        else:
            event=Event(key)
            event.mid=m_id
            mid2event[m_id]=event
            
        # events["E{}".format(index)] = {
        #     # "class": event_e["class"],
        #     "trigger_start_index": int(event_e.findAll("token_anchor")[0]["t_id"])-1,
        #     "trigger_end_index": int(event_e.findAll("token_anchor")[-1]["t_id"])-1
        # }
        
    # res["events"] = events
    relations = {}
    for rel in bs.find_all("plot_link"):
        source_event_id = int(rel.source["m_id"])
        target_event_id = int(rel.target["m_id"])
        if not ("signal" in rel.attrs):
            continue
        signal_event_id = rel["signal"] 
        if signal_event_id!="":
            """有信号"""
            
            if rel["reltype"]=="PRECONDITION":
                pass
            elif rel["reltype"]=="FALLING_ACTION":
                source_event_id,target_event_id=target_event_id,source_event_id
            else:
                continue
            source_group_index=mid2event[source_event_id].group_id
            target_group_index=mid2event[target_event_id].group_id
            for relation in group_rels:
                if relation.source_group_id==source_group_index and relation.target_group_id==target_group_index:
                    try:
                        relation.signal_token_start_index=mid2event[int(signal_event_id)].token_indexs[0]
                        relation.signal_token_end_index=mid2event[int(signal_event_id)].token_indexs[-1]
                    except Exception as ex:
                        print(ex)
                        continue
    mids=list(mid2event.keys())
    size=len(mids)
    for i in range(size-1):
        for j in range(i+1,size):
            mid1=mids[i]
            mid2=mids[j]
            event1=mid2event[mid1]
            event2=mid2event[mid2]
            if token_index2sentence_index[mid2event[mid1].token_indexs[0]]==token_index2sentence_index[mid2event[mid2].token_indexs[-1]]:
                relations[f"{mid1}_{mid2}"]={
                    "event1_start_index":event1.token_indexs[0],
                    "event1_end_index":event1.token_indexs[-1],
                    "event2_start_index":event2.token_indexs[0],
                    "event2_end_index":event2.token_indexs[-1],
                    "signal_start_index":-1,
                    "signal_end_index":-1,
                    "cause":0
                }
    for group_rel in group_rels:
        group1=event_token_groups[group_rel.source_group_id]
        group2=event_token_groups[group_rel.target_group_id]
        for event1 in group1.events:
            for event2 in group2.events:
                assert event1.mid!=event2.mid
                key=f"{event1.mid}_{event2.mid}"
                if key in relations:
                    relation=relations[key]
                    relation["cause"]=1
                    relation["signal_start_index"]=group_rel.signal_token_start_index
                    relation["signal_end_index"]=group_rel.signal_token_end_index
                else:
                    key=f"{event2.mid}_{event1.mid}"
                    if key in relations:
                        relation=relations[key]
                        relation["cause"]=-1
                        relation["signal_start_index"]=group_rel.signal_token_start_index
                        relation["signal_end_index"]=group_rel.signal_token_end_index
    
    # relations.append(
    #     {"source": event_id2index[source_event_id], "target": event_id2index[target_event_id], "signal_start_token_index": int(id2event[signal_event_id].findAll("token_anchor")[0]["t_id"])-1 if signal_event_id != "" else -1, "signal_end_token_index": int(id2event[signal_event_id].findAll("token_anchor")[-1]["t_id"])-1 if signal_event_id != "" else -1})
    res["relations"] = list(relations.values())
    return res

    


if __name__ == '__main__':

    import time
    t1=time.time()
    res = []
    for path in os.listdir("raw_data/eventstory/data"):
        dir_ = os.path.join("raw_data/eventstory/data", path)
        if (not os.path.isdir(dir_)) or (path in ["37","41"]):
            continue
        for subpath in os.listdir(dir_):
            if not subpath.endswith("xml"):
                continue
            
            temp = process_file(os.path.join(dir_, subpath),os.path.join("raw_data/eventstory/chain",path,subpath.split(".")[0]+".tab"))
            if temp != None:
                res.append(temp)
    f = open("data/eventstory.json", "w")
    json.dump(res, f, ensure_ascii=False, indent=4)
    f.close()
    print(time.time()-t1)
# print(process_file("timebank/ABC19980108.1830.0711.xml"))
