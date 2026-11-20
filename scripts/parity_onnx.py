import torch, numpy as np, onnxruntime as ort, time, sys
torch.set_num_threads(4)
from reverb_tts import TTS
from reverb_tts.tts import _prompt
from reverb_tts.sampler import Prompt, Batch, sample, SamplerConfig
from reverb_tts.text import prepare_text, estimate_frames
from reverb_tts.model import MASK
d=sys.argv[1] if len(sys.argv)>1 else "onnx_fp16"
tts=TTS.from_pretrained(variant="fp16", device="cpu")
so=ort.SessionOptions(); so.intra_op_num_threads=6
S={n:ort.InferenceSession(f"{d}/{n}.onnx",so,providers=["CPUExecutionProvider"]) for n in ["core","embed_prefix","embed_target","heads"]}

class OnnxBatch:
    def __init__(self,p,polish_bias):
        self.n=1; self.device=torch.device("cpu"); self.frames=[p.frames]; self.F=p.frames
        self.target=torch.ones(1,self.F,dtype=torch.bool)
        x=S["embed_prefix"].run(None,{"text_ids":p.ids[None].numpy(),"ref_codes":p.reference[None].numpy()})[0]
        P=x.shape[1]; self.P=P; R=p.reference.shape[1]
        empty=np.zeros((24,1,6,0,128),np.float32)
        _,self.k,self.v=S["core"].run(None,{"x":x,"past_k":empty,"past_v":empty,"bias":np.zeros((P,P),np.float32)})
        self.bias={}
        for pol in (False,True):
            b=np.zeros((self.F,P+self.F),np.float32)
            if pol: b[:,P-R:P]=polish_bias
            self.bias[pol]=b
    def log_probs(self,tokens,codebooks,polish=False):
        x=S["embed_target"].run(None,{"tokens":tokens.numpy()})[0]
        h=S["core"].run(None,{"x":x,"past_k":self.k,"past_v":self.v,"bias":self.bias[polish]})[0]
        lp=S["heads"].run(None,{"hidden":h})[0]
        return torch.from_numpy(lp)[:,codebooks.start:codebooks.stop]

v=tts.preset_voice("es_male_standard_1")
cfg=SamplerConfig()
for text in ["Buenos días, ¿cómo estás hoy? Me llamo Ana.","El tren sale a las ocho de la mañana, así que debemos levantarnos temprano."]:
    words=prepare_text(text,"es"); F=estimate_frames(words,v.text,v.pace_frames)
    ids=tts.tokenizer(_prompt(words,v.text,"es"),return_tensors="pt").input_ids[0]
    p=Prompt(ids,v.codes,F,7)
    t=time.time(); ref=sample(Batch(tts.network,[p],cfg.polish_bias),[7],cfg)[0]; t_pt=time.time()-t
    t=time.time(); ob=OnnxBatch(p,cfg.polish_bias); got=sample(ob,[7],cfg)[0]; t_ox=time.time()-t
    eq=(ref==got).float().mean().item()
    print(f"F={F}: code agreement {eq*100:.2f}%  per-codebook {[round((ref[c]==got[c]).float().mean().item()*100,1) for c in range(8)]}  torch {t_pt:.1f}s onnx-cpu {t_ox:.1f}s",flush=True)
    audio_r=tts.codec.decode(ref[None]).audio_values[0,0].detach().numpy(); audio_o=tts.codec.decode(got[None]).audio_values[0,0].detach().numpy()
    n=min(len(audio_r),len(audio_o)); print("   decoded waveform max abs diff", float(np.abs(audio_r[:n]-audio_o[:n]).max()),"(rms ref",float(np.sqrt((audio_r**2).mean())),")")
