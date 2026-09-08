"""Optional local bootstrap training. This performs genuine gradient updates."""
from model import ModelStore
from trainer import Trainer
from memory import MemoryStore
# A short local bootstrap keeps the model usable before web-guided training.
store=ModelStore('verdant_state.pt')
mem=MemoryStore('verdant_memory.json')
trainer=Trainer(store,mem)
trainer.start('general conversation and clear explanations',minutes=2)
print('Training started in the background. Wait for completion, then run the server.')
