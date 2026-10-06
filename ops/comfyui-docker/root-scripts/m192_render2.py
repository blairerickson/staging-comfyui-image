import json, time, urllib.request
BASE="http://127.0.0.1:8188"
CKPT="krea2TurboNSFWAIO_v10.safetensors"
POS="a portrait photograph of a young woman, neutral expression, plain grey background, soft studio lighting, head and shoulders, looking at camera"
NEG="blurry, lowres, deformed, watermark, text"
SEED=424242
def graph(lora, sm, sc, prefix):
    g = {
     "1": {"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":CKPT}},
     "2": {"class_type":"LoraLoader","inputs":{"model":["1",0],"clip":["1",1],"lora_name":lora,"strength_model":sm,"strength_clip":sc}},
     "3": {"class_type":"CLIPTextEncode","inputs":{"clip":["2",1],"text":POS}},
     "4": {"class_type":"CLIPTextEncode","inputs":{"clip":["2",1],"text":NEG}},
     "5": {"class_type":"EmptySD3LatentImage","inputs":{"width":1024,"height":1024,"batch_size":1}},
     "6": {"class_type":"KSampler","inputs":{"model":["2",0],"positive":["3",0],"negative":["4",0],"latent_image":["5",0],
           "seed":SEED,"steps":14,"cfg":1.1,"sampler_name":"euler","scheduler":"beta","denoise":1.0}},
     "7": {"class_type":"VAEDecode","inputs":{"samples":["6",0],"vae":["1",2]}},
     "8": {"class_type":"SaveImage","inputs":{"images":["7",0],"filename_prefix":prefix}},
    }
    return g
jobs=[("krea2_zendaya_v1_onetrainer.safetensors",1.0,1.0,"m192_zendaya"),
      ("krea2_emmawatson_v1_onetrainer.safetensors",1.0,1.0,"m192_watson"),
      ("90s_Grit_v01.safetensors",0.0,0.0,"m192_plain")]
pids={}
for lora,sm,sc,pref in jobs:
    res=json.loads(urllib.request.urlopen(urllib.request.Request(BASE+"/prompt",data=json.dumps({"prompt":graph(lora,sm,sc,pref),"client_id":"m192"}).encode(),headers={"Content-Type":"application/json"}),timeout=60).read().decode())
    pids[res["prompt_id"]]=pref; print("queued",pref)
done={}
t0=time.time()
while pids and time.time()-t0<420:
    for pid in list(pids):
        try: h=json.loads(urllib.request.urlopen(BASE+"/history/"+pid,timeout=30).read().decode())
        except Exception: continue
        if pid in h and h[pid].get("outputs"):
            imgs=[im for n in h[pid]["outputs"].values() for im in (n.get("images") or []) if im.get("type")=="output"]
            done[pids.pop(pid)]=[im["filename"] for im in imgs]
    time.sleep(4)
print(json.dumps(done))
