from __future__ import annotations
import json, threading
from pathlib import Path

class MemoryStore:
    def __init__(self,path='verdant_memory.json'):
        self.path=Path(path); self.lock=threading.RLock()
        self.data={'chats':{},'learning':{'sessions':[],'skills':{},'seen_sources':[]}}
        self._load()
    def _load(self):
        if self.path.exists():
            try:self.data=json.loads(self.path.read_text('utf-8'))
            except Exception:pass
    def save(self):
        with self.lock:self.path.write_text(json.dumps(self.data,ensure_ascii=False,indent=2),'utf-8')
    def chat(self,chat_id):
        return self.data['chats'].setdefault(chat_id,{'messages':[],'summary':''})
    def add_message(self,chat_id,role,content):
        self.chat(chat_id)['messages'].append({'role':role,'content':content}); self.save()
    def set_summary(self,chat_id,summary):self.chat(chat_id)['summary']=summary; self.save()
    def recent_context(self,chat_id,n=10):return self.chat(chat_id)['messages'][-n:]
