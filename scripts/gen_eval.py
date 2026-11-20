import os
REF_WAV = os.environ.get("REF_WAV", "reference.wav")  # clean 5-10 s speech, 24 kHz or any rate
REF_TEXT = os.environ.get("REF_TEXT", "")  # exact transcript of REF_WAV (gen_eval.py only)
import os, time, torch
torch.set_num_threads(6)
import soundfile as sf
from reverb_tts import TTS
texts=["Buenos días, ¿cómo estás hoy? Me llamo Ana y voy a contarte una historia sobre el mercado.",
"¡Qué sorpresa verte aquí! No esperaba encontrarte en la biblioteca esta tarde.",
"El tren sale a las ocho de la mañana, así que debemos levantarnos temprano.",
"Uno, dos, tres, cuatro, cinco, seis, siete, ocho, nueve, diez."]
tts=TTS.from_pretrained(variant="fp16", device="cpu")
voices={"preset1":tts.preset_voice("es_male_standard_1"),"preset2":tts.preset_voice("es_male_standard_2"),
 "stref":tts.voice(REF_WAV,REF_TEXT,"es")}
tot=0;aud=0
for name,v in voices.items():
    os.makedirs(f"gen/{name}",exist_ok=True)
    for ti,t in enumerate(texts):
        for s in range(6):
            f=f"gen/{name}/t{ti}_s{s}.wav"
            if os.path.exists(f): continue
            t0=time.time(); a=tts.generate(t,v,"es",seed=1000+s); dt=time.time()-t0
            sf.write(f,a,tts.sample_rate); tot+=dt; aud+=len(a)/tts.sample_rate
    print(name,"done; cumulative RTF",round(tot/max(aud,1e-9),2),flush=True)
