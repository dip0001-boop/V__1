from __future__ import annotations
import re, requests
from bs4 import BeautifulSoup
from urllib.parse import quote

UA='Verdant-1.0/15x-trainer (learning research client)'

def _clean(s):
    s=re.sub(r'\s+',' ',s or '').strip()
    return s

def wiki_search(query,limit=8):
    url='https://en.wikipedia.org/w/api.php'
    params={'action':'query','list':'search','srsearch':query,'format':'json','utf8':1,'srlimit':limit}
    r=requests.get(url,params=params,headers={'User-Agent':UA},timeout=15); r.raise_for_status()
    hits=[]
    for x in r.json().get('query',{}).get('search',[]):
        title=x['title']
        hits.append({'title':title,'snippet':BeautifulSoup(x.get('snippet',''),'html.parser').get_text(' '),'url':'https://en.wikipedia.org/wiki/'+quote(title.replace(' ','_'))})
    return hits

def fetch(url,max_chars=18000):
    r=requests.get(url,headers={'User-Agent':UA},timeout=20); r.raise_for_status()
    soup=BeautifulSoup(r.text,'html.parser')
    for t in soup(['script','style','noscript']):t.decompose()
    return _clean(soup.get_text(' '))[:max_chars]

def research_goal(goal,limit=10,seen_titles=None):
    seen_titles=seen_titles or set(); hits=wiki_search(goal,max(limit*3,12))
    # Relevance is an information-retrieval heuristic, not an intelligence rule.
    terms=set(re.findall(r'[a-z0-9]{3,}',goal.lower()))
    scored=[]
    for h in hits:
        if h['title'].lower() in seen_titles: continue
        blob=(h['title']+' '+h['snippet']).lower()
        overlap=sum(1 for t in terms if t in blob)
        scored.append((overlap,h))
    scored.sort(key=lambda z:(-z[0],z[1]['title']))
    out=[]
    for _,h in scored[:limit]:
        try:
            text=fetch(h['url'])
            if len(text)>500: out.append({**h,'text':text})
        except Exception:
            continue
    return out
