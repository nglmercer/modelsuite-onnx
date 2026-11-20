import os
REF_WAV = os.environ.get("REF_WAV", "reference.wav")  # clean 5-10 s speech, 24 kHz or any rate
REF_TEXT = os.environ.get("REF_TEXT", "")  # exact transcript of REF_WAV (gen_eval.py only)
import torch, os, time, numpy as np
import torch.nn.functional as F
torch.set_num_threads(4)
from transformers import HiggsAudioV2TokenizerModel
m=HiggsAudioV2TokenizerModel.from_pretrained("eustlb/higgs-audio-v2-tokenizer").eval()
out="onnx_fp16"; os.makedirs(out,exist_ok=True)
print("semantic_downsample_factor",m.config.semantic_downsample_factor,"targets",m.config.target_bandwidths,"nq",m.quantizer.num_quantizers)

class Dec(torch.nn.Module):
    def __init__(s,m): super().__init__(); s.m=m
    def forward(s,codes):  # [1,8,T] int64 -> [1,1,T*960]
        q=0
        for i,qz in enumerate(s.m.quantizer.quantizers): q=q+qz.decode(codes[:,i])
        return s.m.acoustic_decoder(s.m.fc2(q.transpose(1,2)).transpose(1,2))

class Enc(torch.nn.Module):
    def __init__(s,m,pad_acoustic): super().__init__(); s.m=m; s.pad_acoustic=pad_acoustic
    def forward(s,wav24,wav16):  # [1,1,N], [1,N16] (16 kHz resample done by the caller)
        m=s.m
        x=F.pad(wav16,(160,160))
        hs=m.semantic_model(x,output_hidden_states=True).hidden_states
        sem=torch.stack(list(hs),dim=1).mean(dim=1)
        if m.config.semantic_downsample_factor>1: sem=sem[:,::m.config.semantic_downsample_factor,:]
        e_sem=m.encoder_semantic(sem.transpose(1,2))
        e_ac=m.acoustic_encoder(F.pad(wav24,(m.pad,m.pad)) if s.pad_acoustic else wav24)
        e=torch.cat([e_ac,e_sem],dim=1)
        e=m.fc(e.transpose(1,2)).transpose(1,2)
        return m.quantizer.encode(e,m.config.target_bandwidths[-1]).transpose(0,1)

import soundfile as sf, torchaudio
a,sr=sf.read(REF_WAV); a=torch.tensor(a,dtype=torch.float32)
a=torchaudio.functional.resample(a,sr,24000) if sr!=24000 else a
a=F.pad(a,(0,-a.numel()%960)); w24=a[None,None]
w16=torchaudio.functional.resample(w24,24000,16000)[:,0]
with torch.inference_mode():
    ref_codes=m.encode(w24).audio_codes
    pad_ac=None
    for cand in (False,True):
        try:
            ok=(Enc(m,cand).eval()(w24,w16)==ref_codes).float().mean().item()
        except RuntimeError: ok=-1
        print("pad_acoustic",cand,"agreement",ok)
        if ok>0.99: pad_ac=cand
    assert pad_ac is not None
    enc=Enc(m,pad_ac).eval(); dec=Dec(m).eval()
    ref_wav=m.decode(ref_codes).audio_values; print("wrapper dec == official:",(dec(ref_codes)-ref_wav).abs().max().item())
t=time.time(); torch.onnx.export(dec,(ref_codes,),f"{out}/codec_decoder.onnx",input_names=["codes"],output_names=["audio"],dynamic_axes={"codes":{2:"T"},"audio":{2:"N"}},opset_version=18,dynamo=False); print("decoder exported",round(time.time()-t,1),flush=True)
t=time.time(); torch.onnx.export(enc,(w24,w16),f"{out}/codec_encoder.onnx",input_names=["wav24","wav16"],output_names=["codes"],dynamic_axes={"wav24":{2:"N"},"wav16":{1:"N16"},"codes":{2:"T"}},opset_version=18,dynamo=False); print("encoder exported",round(time.time()-t,1),flush=True)
torch.save({"w24":w24,"w16":w16,"codes":ref_codes,"wav":ref_wav},"codec_ref.pt")
for f in ("codec_decoder.onnx","codec_encoder.onnx"): print(f, round(os.path.getsize(f"{out}/{f}")/1e6,1),"MB")
