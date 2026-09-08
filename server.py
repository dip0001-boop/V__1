from __future__ import annotations
import random,re,uuid
from pathlib import Path
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel,Field
import torch
from model import ModelStore
from memory import MemoryStore
from summary import summarize
from trainer import Trainer

app=FastAPI(title='Verdant-1.0')
ROOT=Path(__file__).parent
store=ModelStore(ROOT/'verdant_state.pt'); memory=MemoryStore(ROOT/'verdant_memory.json'); trainer=Trainer(store,memory)

class ChatRequest(BaseModel):message:str=Field(min_length=1,max_length=12000);chat_id:str|None=None;effort:str='medium'
class TrainRequest(BaseModel):goal:str=Field(min_length=2,max_length=300);minutes:float=Field(default=20,ge=1,le=120)

@app.get('/')
def index():return FileResponse(ROOT/'index.html')

@app.get('/api/status')
def status():return {'training':trainer.snapshot(),'model_step':store.step,'parameters':store.parameter_count()}


def _sample(prompt,max_new,temperature):
    tok=store.tokenizer; ids=tok.encode(prompt,max_len=store.model.cfg.context)
    if not ids:ids=[32]
    x=torch.tensor([ids],dtype=torch.long)
    generated=[]
    hidden=None
    store.model.eval()
    with torch.no_grad():
        logits,hidden=store.model(x,hidden)
        last=x[:,-1:]
        for _ in range(max_new):
            logits,hidden=store.model(last,hidden)
            z=logits[:,-1,:]/max(0.25,temperature)
            probs=torch.softmax(z,dim=-1)
            # stochastic nucleus-ish sampling without hard-coded knowledge.
            vals,idx=torch.sort(probs,descending=True)
            c=torch.cumsum(vals,dim=-1); keep=c<=0.92
            keep[...,0]=True
            vals=vals*keep
            vals=vals/vals.sum(dim=-1,keepdim=True)
            pick=torch.multinomial(vals,1)
            nxt=idx.gather(-1,pick)
            generated.append(int(nxt.item())); last=nxt
    return tok.decode(generated).strip()

def _effort_config(e):
    return {'lite':(80,0.9),'medium':(140,0.82),'max':(220,0.72)}.get((e or 'medium').lower(),(140,0.82))

@app.post('/api/chat')
def chat(req:ChatRequest):
    cid=req.chat_id or str(uuid.uuid4()); c=memory.chat(cid)
    memory.add_message(cid,'user',req.message)
    context='\n'.join(f"{m['role'].title()}: {m['content']}" for m in c['messages'][-8:])
    max_new,temp=_effort_config(req.effort)
    # The model itself determines the output; these are runtime controls only.
    prompt=context+'\nAssistant:'
    answer=_sample(prompt,max_new,temp)
    if not answer: answer=''
    memory.add_message(cid,'assistant',answer); memory.set_summary(cid,summarize(memory.chat(cid)['messages']))
    return {'chat_id':cid,'response':answer,'summary':memory.chat(cid)['summary'],'effort':req.effort,'training_step':store.step}

@app.post('/api/train/start')
def start(req:TrainRequest):
    try:trainer.start(req.goal,req.minutes);return {'ok':True}
    except Exception as e:raise HTTPException(409,str(e))

@app.post('/api/train/stop')
def stop():trainer.halt();return {'ok':True}

@app.get('/api/health')
def health():return {'ok':True,'step':store.step,'parameters':store.parameter_count()}
