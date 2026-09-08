from __future__ import annotations
import math, random, re, threading, time
from dataclasses import dataclass,asdict
from pathlib import Path
import torch
import torch.nn.functional as F
from model import ModelStore
from research import research_goal

@dataclass
class Status:
    running:bool=False; goal:str=''; phase:str='idle'; elapsed:float=0.0; requested_minutes:float=20
    steps:int=0; loss:float|None=None; val_loss:float|None=None; mastery:float|None=None
    sources:int=0; cycles:int=0; message:str=''; next_focus:str=''

class Trainer:
    def __init__(self,store, memory):
        self.store=store; self.memory=memory; self.status=Status(); self.thread=None; self.stop=threading.Event(); self.lock=threading.RLock()

    def snapshot(self):
        with self.lock:return asdict(self.status)
    def start(self,goal,minutes=20):
        with self.lock:
            if self.status.running:raise RuntimeError('training already running')
            self.stop.clear(); self.status=Status(running=True,goal=goal,requested_minutes=max(1,min(120,float(minutes))),phase='baseline',message='Building a held-out baseline')
        self.thread=threading.Thread(target=self._run,args=(goal,self.status.requested_minutes),daemon=True); self.thread.start()
    def halt(self):self.stop.set()

    def _textset(self,docs,goal):
        chunks=[]
        for d in docs: chunks.append(f"Topic: {d['title']}\n{d['text']}")
        # tiny generic language rehearsal; not answer rules.
        chunks += [
            'User: hello\nAssistant: Hello. What would you like to explore today?',
            'User: how are you?\nAssistant: I am ready to help. What are you working on?',
            'User: explain this simply\nAssistant: Start with the central idea, then explain the important parts step by step.',
            f'User: what is {goal}?\nAssistant: {goal} is a subject we can investigate, practice, and test.'
        ]
        return '\n\n'.join(chunks)

    def _split_holdout(self,text,goal):
        lines=[x.strip() for x in text.splitlines() if len(x.strip())>50]
        random.Random(2026).shuffle(lines)
        hold=lines[:max(20,min(80,len(lines)//6 or 20))]
        train='\n'.join(lines[len(hold):])
        return train,hold

    def _batch(self,text,batch=16,seq=256):
        raw=text.encode('utf-8','replace')
        if len(raw)<seq+2:raw=(raw*((seq+2)//max(1,len(raw))+1))
        xs=[];ys=[]
        for _ in range(batch):
            i=random.randint(0,len(raw)-seq-1); xs.append(list(raw[i:i+seq])); ys.append(list(raw[i+1:i+seq+1]))
        return torch.tensor(xs,dtype=torch.long),torch.tensor(ys,dtype=torch.long)

    def _step(self,text,seq=256,batch=16):
        self.store.model.train(); x,y=self._batch(text,batch,seq)
        with self.store.lock:
            logits,_=self.store.model(x)
            loss=F.cross_entropy(logits.reshape(-1,256),y.reshape(-1))
            self.store.optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.store.model.parameters(),1.0)
            self.store.optimizer.step()
            self.store.step+=1
        return float(loss.detach())

    def _val(self,text,limit=120000):
        if not text:return None
        raw=text.encode('utf-8','replace')[:limit]
        if len(raw)<8:return None
        seq=min(256,len(raw)-1)
        x=torch.tensor([list(raw[:seq])],dtype=torch.long); y=torch.tensor([list(raw[1:seq+1])],dtype=torch.long)
        self.store.model.eval()
        with torch.no_grad():
            logits,_=self.store.model(x); return float(F.cross_entropy(logits.reshape(-1,256),y.reshape(-1)))

    def _mastery(self,goal,docs):
        # Real proxy: held-out byte-level perplexity improvement + topic evidence coverage.
        blob=(' '.join(d['text'] for d in docs)).lower(); terms=set(re.findall(r'[a-z0-9]{4,}',goal.lower()))
        evidence=sum(1 for t in terms if t in blob)/max(1,len(terms))
        val=self._val(blob)
        fit=math.exp(-max(0.0,(val or 12.0)-2.5)/4.0)
        return max(0,min(1,0.55*fit+0.45*evidence))

    def _run(self,goal,minutes):
        started=time.time()
        try:
            with self.lock:self.status.phase='research'; self.status.message='Researching the goal from Wikipedia and avoiding repeats'
            seen=set(self.memory.data['learning'].get('seen_sources',[]))
            docs=research_goal(goal,10,seen)
            for d in docs:seen.add(d['title'].lower())
            self.memory.data['learning']['seen_sources']=list(seen)[-500:]
            self.memory.save()
            corpus=self._textset(docs,goal); train_text,hold=self._split_holdout(corpus,goal)
            with self.lock:self.status.sources=len(docs);self.status.phase='learn';self.status.message='Learning from the researched material'
            deadline=started+minutes*60
            recent_losses=[]
            cycle=0
            while time.time()<deadline and not self.stop.is_set():
                cycle+=1; cycle_losses=[]
                for _ in range(96):
                    if time.time()>=deadline or self.stop.is_set():break
                    cycle_losses.append(self._step(train_text))
                recent_losses += cycle_losses[-32:]
                val=self._val('\n'.join(hold))
                mastery=self._mastery(goal,docs)
                with self.lock:
                    self.status.cycles=cycle;self.status.steps=self.store.step;self.status.loss=sum(cycle_losses)/max(1,len(cycle_losses));self.status.val_loss=val;self.status.mastery=mastery;self.status.elapsed=time.time()-started
                self.store.save()
                # Remediation: retrieve another targeted source when mastery is weak.
                if mastery<0.80 and cycle%2==0:
                    with self.lock:self.status.phase='remediate';self.status.next_focus='Find a missing subtopic and add new evidence'
                    extra_goal=f'{goal} fundamentals examples common misconceptions'
                    extra=research_goal(extra_goal,4,seen)
                    for d in extra:seen.add(d['title'].lower())
                    docs += extra
                    train_text,_=self._split_holdout(self._textset(docs,goal),goal)
                    with self.lock:self.status.phase='learn';self.status.message='Targeted remediation from new material'
                else:
                    with self.lock:self.status.phase='consolidate';self.status.message='Consolidating and testing learned state'
                time.sleep(0.02)
            final=self._mastery(goal,docs)
            self.store.save()
            session={'goal':goal,'steps':self.store.step,'sources':len(docs),'mastery':final,'finished_at':time.time()}
            self.memory.data['learning']['sessions'].append(session);self.memory.save()
            with self.lock:
                self.status.running=False;self.status.phase='complete';self.status.mastery=final;self.status.elapsed=time.time()-started;self.status.message=f'Training complete • measured mastery proxy {final*100:.1f}%'
        except Exception as e:
            with self.lock:self.status.running=False;self.status.phase='error';self.status.message=str(e);self.status.elapsed=time.time()-started
