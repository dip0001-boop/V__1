from __future__ import annotations

def summarize(messages):
    # Compact deterministic memory representation; it does not generate intelligence.
    if not messages:return ''
    parts=[]
    for m in messages[-12:]:
        text=' '.join((m.get('content') or '').split())[:220]
        if text:parts.append(('User' if m.get('role')=='user' else 'Verdant')+': '+text)
    return ' | '.join(parts)[-1800:]
