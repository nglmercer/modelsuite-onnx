import os
REF_WAV = os.environ.get("REF_WAV", "reference.wav")  # clean 5-10 s speech, 24 kHz or any rate
REF_TEXT = os.environ.get("REF_TEXT", "")  # exact transcript of REF_WAV (gen_eval.py only)
import torch, numpy as np, onnxruntime as ort, time
import torch.nn.functional as F, torchaudio, scipy.signal as ss, soundfile as sf
from transformers import HiggsAudioV2TokenizerModel
m=HiggsAudioV2TokenizerModel.from_pretrained("eustlb/higgs-audio-v2-tokenizer").eval()
so=ort.SessionOptions(); so.intra_op_num_threads=6
dec=ort.InferenceSession("onnx_fp16/codec_decoder.onnx",so,providers=["CPUExecutionProvider"])
enc=ort.InferenceSession("onnx_fp16/codec_encoder.onnx",so,providers=["CPUExecutionProvider"])
a,sr=sf.read(REF_WAV); a=torch.tensor(a,dtype=torch.float32)
if sr!=24000: a=torchaudio.functional.resample(a,sr,24000)
for secs in (7.0,3.0,1.4):
    x=a[:int(secs*24000)]; x=F.pad(x,(0,-x.numel()%960)); w24=x[None,None]
    with torch.inference_mode(): ref=m.encode(w24).audio_codes; refwav=m.decode(ref).audio_values.numpy()
    w16=torchaudio.functional.resample(w24,24000,16000)[:,0].numpy()
    t=time.time(); c1=enc.run(None,{"wav24":w24.numpy(),"wav16":w16})[0]; te=time.time()-t
    w16s=ss.resample_poly(w24[0,0].numpy().astype(np.float64),2,3).astype(np.float32)[None]
    c2=enc.run(None,{"wav24":w24.numpy(),"wav16":w16s})[0]
    t=time.time(); au=dec.run(None,{"codes":ref.numpy()})[0]; td=time.time()-t
    print(f"{secs}s: frames {ref.shape[2]} | encoder(torchaudio rs) agree {np.mean(c1==ref.numpy())*100:.2f}% ({te:.1f}s) | encoder(scipy rs) agree {np.mean(c2==ref.numpy())*100:.2f}% | decoder max diff {np.abs(au-refwav).max():.2e} ({td:.2f}s)")
