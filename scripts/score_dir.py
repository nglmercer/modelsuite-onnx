import sys, os, re, json, unicodedata, glob
import soundfile as sf
from faster_whisper import WhisperModel
texts=["Buenos días, ¿cómo estás hoy? Me llamo Ana y voy a contarte una historia sobre el mercado.",
"¡Qué sorpresa verte aquí! No esperaba encontrarte en la biblioteca esta tarde.",
"El tren sale a las ocho de la mañana, así que debemos levantarnos temprano.",
"Uno, dos, tres, cuatro, cinco, seis, siete, ocho, nueve, diez."]
def norm(s):
    s=unicodedata.normalize('NFKD',s.lower()); s=''.join(c for c in s if not unicodedata.combining(c))
    for a,b in [('10','diez'),('1','uno'),('2','dos'),('3','tres'),('4','cuatro'),('5','cinco'),('6','seis'),('7','siete'),('8','ocho'),('9','nueve')]: s=re.sub(rf'\b{a}\b',b,s)
    return re.findall(r"[a-zñ]+",s)
def ed(a,b):
    d=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        p,d[0]=d[0],i
        for j,y in enumerate(b,1): p,d[j]=d[j],min(d[j]+1,d[j-1]+1,p+(x!=y))
    return d[-1]
m=WhisperModel('small',device='cpu',compute_type='int8',cpu_threads=6)
for d in sys.argv[1:]:
    errs=words=bad=n=0; dur=0
    for ti,t in enumerate(texts):
        for f in sorted(glob.glob(f"{d}/t{ti}_s*.wav")):
            a,sr=sf.read(f,dtype='float32'); dur+=len(a)/sr
            segs,_=m.transcribe(a if sr==16000 else __import__('scipy.signal',fromlist=['x']).resample_poly(a,16000,sr).astype('float32'),language='es',beam_size=5)
            hyp=' '.join(x.text for x in segs); ref=norm(t); e=ed(ref,norm(hyp)); errs+=e; words+=len(ref); n+=1; bad+= e/len(ref)>0.5
    print(json.dumps({"set":d,"WER":round(errs/words,3),"bad(>50%)":f"{bad}/{n}","avg_audio_s":round(dur/n,2)}),flush=True)
