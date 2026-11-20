import torch, os, sys, time
torch.set_num_threads(4)
from reverb_tts import TTS
from export_net import CoreGraph, EmbedPrefix, EmbedTarget, Heads
variant=sys.argv[1] if len(sys.argv)>1 else "fp16"
out=f"onnx_{variant}"; os.makedirs(out,exist_ok=True)
tts=TTS.from_pretrained(variant=variant, device="cpu"); n=tts.network
S,Pp=62,183
x=torch.randn(1,S,1024); pk=torch.randn(24,1,6,Pp,128); pv=torch.randn(24,1,6,Pp,128); bias=torch.zeros(S,Pp+S)
def ex(mod,args,name,inn,outn,dyn):
    t=time.time(); torch.onnx.export(mod.eval(),args,f"{out}/{name}.onnx",input_names=inn,output_names=outn,dynamic_axes=dyn,opset_version=18,dynamo=False)
    print(name,"exported",round(time.time()-t,1),"s",flush=True)
ex(CoreGraph(n),(x,pk,pv,bias),"core",["x","past_k","past_v","bias"],["hidden","new_k","new_v"],
   {"x":{1:"S"},"past_k":{3:"Pp"},"past_v":{3:"Pp"},"bias":{0:"S",1:"T"},"hidden":{1:"S"},"new_k":{3:"S"},"new_v":{3:"S"}})
ex(EmbedPrefix(n),(torch.randint(0,1000,(1,53)),torch.randint(0,1024,(1,8,130))),"embed_prefix",["text_ids","ref_codes"],["x"],
   {"text_ids":{1:"Pt"},"ref_codes":{2:"R"},"x":{1:"P"}})
ex(EmbedTarget(n),(torch.randint(0,1024,(1,8,S)),),"embed_target",["tokens"],["x"],{"tokens":{2:"F"},"x":{1:"F"}})
ex(Heads(n),(x,),"heads",["hidden"],["log_probs"],{"hidden":{1:"F"},"log_probs":{2:"F"}})
for f in sorted(os.listdir(out)): print(f, round(os.path.getsize(f"{out}/{f}")/1e6,1),"MB")
