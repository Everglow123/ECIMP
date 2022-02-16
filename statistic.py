import json
from typing import *
def proc_doc(data:Dict[str,any]):
    tokens:List[str]=data["tokens"]
    token_index2sentence_index:List[int]=data["token_index2sentence_index"]
    relations:List[Dict[str,int]]=data["relations"]
    sentences:List[Dict[str,int]]=data["sentences"]
    rel_size=len(relations)
    res=[]
    for i, rel in enumerate(relations):
        tokens1=tokens.copy()
        substr_event_start_index=min(rel["event1_start_index"],rel["event2_start_index"])
        substr_event_end_index=max(rel["event1_end_index"],rel["event2_end_index"])
        tokens1[rel["event1_start_index"]]="</t1> "+tokens1[rel["event1_start_index"]]
        tokens1[rel["event1_end_index"]]=tokens1[rel["event1_end_index"]]+" </t2>"
        tokens1[rel["event2_start_index"]]="</t3> "+tokens1[rel["event2_start_index"]]
        tokens1[rel["event2_end_index"]]=tokens1[rel["event2_end_index"]]+" </t4>"
        sentences_index1=token_index2sentence_index[substr_event_start_index]
        sentences_index2=token_index2sentence_index[substr_event_end_index]
        if sentences_index2-sentences_index1>=2:
            if rel["cause"]:
                res.append({"cause":rel["cause"],"sentence":" ".join(tokens1[sentences[sentences_index1]["start"]:sentences[sentences_index2]["end"]+1])}) 
    return res

js=[]
res=[]
with open("data/timebank.json","r") as f:
    js=json.load(f)
for d in js:
    res.extend(proc_doc(d))

with open("wtf.json","w")  as f:
    json.dump(res,f,indent=4,ensure_ascii=False)   
