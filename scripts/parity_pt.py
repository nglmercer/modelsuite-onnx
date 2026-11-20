import torch, math
torch.set_num_threads(4)
from reverb_tts import TTS
from reverb_tts.tts import _prompt
from reverb_tts.sampler import Prompt, Batch
from reverb_tts.text import prepare_text, estimate_frames
from reverb_tts.model import MASK
from export_net import Prefix, Step
tts=TTS.from_pretrained(variant="fp16", device="cpu")
v=tts.preset_voice("es_male_standard_1")
words=prepare_text("Buenos días, ¿cómo estás hoy? Me llamo Ana.", "es")
F=estimate_frames(words, v.text, v.pace_frames)
ids=tts.tokenizer(_prompt(words,v.text,"es"),return_tensors="pt").input_ids[0]
print("text tokens",len(ids),"ref frames",v.codes.shape[1],"target frames",F)
p=Prompt(ids,v.codes,F,1)
b=Batch(tts.network,[p],1.0)
g=torch.Generator().manual_seed(0)
tokens=torch.randint(0,1024,(1,8,F),generator=g); tokens[torch.rand(1,8,F,generator=g)<0.6]=MASK
pre=Prefix(tts.network).eval(); step=Step(tts.network).eval()
with torch.inference_mode():
    K,V=pre(ids[None],v.codes[None])
    P=K.shape[3]; print("kv",tuple(K.shape))
    for polish in (False,True):
        ref=b.log_probs(tokens,range(0,8),polish=polish)
        bias=torch.zeros(P)
        if polish: bias[P-v.codes.shape[1]:]=1.0
        out=step(tokens,K,V,bias)
        d=(ref-out).abs()
        print("polish",polish,"max abs diff",d.max().item(),"argmax equal",(ref.argmax(-1)==out.argmax(-1)).float().mean().item())
