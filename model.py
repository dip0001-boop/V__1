from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json, threading, time
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class ModelConfig:
    vocab_size:int=256
    embed_dim:int=384
    hidden_dim:int=512
    layers:int=3
    context:int=384
    dropout:float=0.12

class VerdantCore(nn.Module):
    """Compact from-scratch recurrent language core.

    The model uses learned numerical state; it is not populated with an answer table.
    """
    def __init__(self,cfg:ModelConfig|None=None):
        super().__init__()
        self.cfg=cfg or ModelConfig()
        self.embedding=nn.Embedding(self.cfg.vocab_size,self.cfg.embed_dim)
        self.pre=nn.Linear(self.cfg.embed_dim,self.cfg.embed_dim)
        self.rnn=nn.GRU(self.cfg.embed_dim,self.cfg.hidden_dim,num_layers=self.cfg.layers,batch_first=True,dropout=self.cfg.dropout if self.cfg.layers>1 else 0)
        self.norm=nn.LayerNorm(self.cfg.hidden_dim)
        self.readout=nn.Linear(self.cfg.hidden_dim,self.cfg.vocab_size)
        self.state_gate=nn.Linear(self.cfg.hidden_dim,self.cfg.hidden_dim)

    def forward(self,x,hidden=None):
        e=torch.tanh(self.pre(self.embedding(x)))
        y,h=self.rnn(e,hidden)
        y=self.norm(y)
        return self.readout(y),h

    def state(self,x):
        with torch.no_grad():
            _,h=self.forward(x)
        return torch.tanh(self.state_gate(h[-1]))

class ByteTokenizer:
    vocab_size=256
    def encode(self,text:str,max_len:int|None=None):
        ids=list(text.encode('utf-8',errors='replace'))
        if max_len: ids=ids[-max_len:]
        return ids
    def decode(self,ids):
        return bytes(int(i)%256 for i in ids).decode('utf-8',errors='ignore')

class ModelStore:
    def __init__(self,path='verdant_state.pt'):
        self.path=Path(path); self.lock=threading.RLock(); self.tokenizer=ByteTokenizer()
        self.model=VerdantCore(); self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=8e-4,weight_decay=0.02)
        self.step=0; self.best_val=None; self._load()

    def _load(self):
        if not self.path.exists(): return
        payload=torch.load(self.path,map_location='cpu')
        self.model.load_state_dict(payload['model'])
        if payload.get('optimizer'):
            try:self.optimizer.load_state_dict(payload['optimizer'])
            except Exception: pass
        self.step=int(payload.get('step',0)); self.best_val=payload.get('best_val')

    def save(self,path=None):
        p=Path(path or self.path); p.parent.mkdir(parents=True,exist_ok=True)
        with self.lock:
            torch.save({'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),'step':self.step,'best_val':self.best_val,'config':asdict(self.model.cfg)},p)

    def parameter_count(self): return sum(p.numel() for p in self.model.parameters())

    def snapshot(self):
        with self.lock:return {k:v.detach().clone() for k,v in self.model.state_dict().items()}

    def restore(self,state):
        with self.lock:self.model.load_state_dict(state)
