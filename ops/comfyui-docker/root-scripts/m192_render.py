import json, time, urllib.request, urllib.error
BASE="http://127.0.0.1:8188"
CKPT="krea2TurboNSFWAIO_v10.safetensors"
POS="a portrait photograph of a young woman, neutral expression, plain grey background, soft studio lighting, head and shoulders, looking at camera"
NEG="blurry, lowres, deformed, watermark, text"
SEED=424242

def graph(lora, prefix):
    g = {
     "1": {"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":CKPT}},
     "3": {"class_type":"CLIPTextEncode","inputs":{"clip":["2",1],"text":POS}},
     "4": {"class_type":"CLIPTextEncode","inputs":{"clip":["2",1],"text":NEG}},
     "5": {"class_type":"EmptySD3LatentImage","inputs":{"width":1024,"height":1024,"batch_size":1}},
     "6": {"class_type":"KSampler","inputs":{"model":["2",0],"positive":["3",0],"negative":["4",0],"latent_image":["5",0],
           "seed":SEED,"steps":14,"cfg":1.1,"sampler_name":"euler","scheduler":"beta","denoise":1.0}},
     "7": {"class_type":"VAEDecode","inputs":{"samples":["6",0],"vae":["1",2]}},
     "8": {"class_type":"SaveImage","inputs":{"images":["7",0],"filename_prefix":prefix}},
    }
    if lora:
        g["2"] = {"class_type":"LoraLoader","inputs":{"model":["1",0],"clip":["1",1],"lora_name":lora,"strength_model":1.0,"strength_clip":1.0}}
    else:
        g["2"] = {"class_type":"LoraLoader","inputs":{"model":["1",0],"clip":["1",1],"lora_name":"90s_Grit_v01.safetensors","strength_model":0.0,"strength_clip":0.0}}
    return g

def post(path, obj):
    d=json.dumps(obj).encode()
    r=urllib.request.Request(BASE+path,data=d,headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r,timeout=60).read().decode())

jobs=[("krea2_emmamyers_v1_onetrainer.safetensors","m192_emma"),
      ("krea2_sadiesink_v1_onetrainer.safetensors","m192_sadie")]
pids={}
for lora,pref in jobs:
    res=post("/prompt",{"prompt":graph(lora,pref),"client_id":"m192"})
    pids[res["prompt_id"]]=(pref,lora)
    print("queued",pref,res["prompt_id"])
    time.sleep(1)

done={}
t0=time.time()
while pids and time.time()-t0 < 420:
    for pid in list(pids):
        try:
            h=json.loads(urllib.request.urlopen(BASE+"/history/"+pid,timeout=30).read().decode())
        except Exception:
            continue
        if pid in h and h[pid].get("outputs"):
            outs=h[pid]["outputs"]
            imgs=[]
            for n in outs.values():
                for im in (n.get("images") or []):
                    if im.get("type")=="output": imgs.append(im)
            pref,lora=pids.pop(pid)
            done[pref]=imgs
            print("DONE",pref,[im["filename"] for im in imgs])
    time.sleep(4)
print("RESULT",json.dumps({k:[i["filename"] for i in v] for k,v in done.items()}))
