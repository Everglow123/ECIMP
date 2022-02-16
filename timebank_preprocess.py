#!/home/zhouheng/anaconda3/bin/python
# -*- encoding: utf-8 -*-
'''
@文件    :timebank_preprocess.py
@时间    :2021/11/17 21:12:26
@作者    :周恒
@版本    :1.0
@说明    :
'''


import json
import os
from typing import Dict, List, Optional, Union
from bs4 import BeautifulSoup
import bs4
import dataclasses


"""
tokens:List
event:[class,trigger_start_index,trigger_end_index]
events:Dict[id,event]
cause_signals:[]
event_relation:Dict[target,source,type:str,signal_start_token_index,signal_end_token_index]
relations:List[event_relation]
"""


def process_file(path: str) -> Optional[Dict[str, Union[str, Dict, List[str]]]]:
    f = open(path, "r")
    text = f.read()
    f.close()
    bs = BeautifulSoup(text, 'lxml')
    # if len(bs.find_all("clink")) == 0:
    #     return None
    res = {}
    print(path)

    id2event: Dict[str, bs4.element.Tag] = {}
    id2token: Dict[str, bs4.element.Tag] = {}
    tokens = []
    sentences=[]
    token_index2sentence_index=[]
    for token_e in bs.findAll("token"):
        id2token[token_e["id"]] = token_e
        sentence=token_e["sentence"]
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
    id2signal = {}
    for signal_id in bs.findAll("c-signal"):
        id2signal[signal_id["id"]] = signal_id
    events = {}
    event_id2index = {}
    for index, event_e in enumerate(bs.findAll("event")):
        id2event[event_e["id"]] = event_e
        events[int(event_e["id"])]={
            "class": event_e["class"],
            "trigger_start_index": int(id2token[event_e.findAll("token_anchor")[0]["id"]]["number"]),
            "trigger_end_index": int(id2token[event_e.findAll("token_anchor")[-1]["id"]]["number"])
        }
        event_id2index[event_e["id"]] = len(events)-1
    # res["events"] = events
    relations = {}
    event_ids=list(events.keys())
    for i in range(len(event_ids)-1):
        for j in range(i+1,len(event_ids)):
            eventi=events[event_ids[i]] 
            eventj=events[event_ids[j]]

            if token_index2sentence_index[max(eventi["trigger_end_index"],eventj["trigger_end_index"])]-\
                token_index2sentence_index[min(eventi["trigger_start_index"],eventj["trigger_start_index"])]>=1:
                continue
            relations[f"{event_ids[i]}_{event_ids[j]}"]={
                "event1_start_index":eventi["trigger_start_index"],
                "event1_end_index":eventi["trigger_end_index"],
                "event2_start_index":eventj["trigger_start_index"],
                "event2_end_index":eventj["trigger_end_index"],
                "signal_start_index":-1,
                "signal_end_index":-1,
                "cause":0
            }
    for rel in bs.find_all("clink"):
        source_event_id = rel.source["id"]
        target_event_id = rel.target["id"]
        signal_id = rel["c-signalid"] if "c-signalid" in rel.attrs else ""
        signal_start_token_index = -1
        signal_end_token_index = -1
        if signal_id != "":
            signal_start_token_index = int(id2signal[signal_id].findAll(
                "token_anchor")[0]["id"])-1
            signal_end_token_index = int(id2signal[signal_id].findAll(
                "token_anchor")[-1]["id"])-1
        key=f"{source_event_id}_{target_event_id}"
        if key in relations:
            relation=relations[key]
            relation["cause"]=1
            relation["signal_start_index"]=signal_start_token_index
            relation["signal_end_index"]=signal_end_token_index
        else:
            key=f"{target_event_id}_{source_event_id}"
            if key in relations:
                relation=relations[key]
                relation["cause"]=-1
                relation["signal_start_index"]=signal_start_token_index
                relation["signal_end_index"]=signal_end_token_index
    res["relations"] = list(relations.values())
    return res


if __name__ == '__main__':
    count=0
    res = []
    for path in os.listdir("raw_data/timebank"):
        temp = process_file(os.path.join("raw_data/timebank", path))
        if temp != None:
            res.append(temp)
            count+=len(temp["relations"])
    f = open("data/timebank.json", "w")
    json.dump(res, f, ensure_ascii=False, indent=4)
    f.close()
    print(count)

# print(process_file("timebank/ABC19980108.1830.0711.xml"))
