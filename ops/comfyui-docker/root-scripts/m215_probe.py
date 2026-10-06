#!/usr/bin/env python3
"""M215 P0 throwaway probe: H3 native single-pass length ceiling.
Submits a MINIMAL single-pass t2v graph (NO chunk-2 nodes) to ComfyUI and
measures wall time + peak VRAM for increasing `length` (frames @24fps).
Does not touch the production two-chunk graph or any live-job artifacts."""
import json, time, subprocess, threading, urllib.request, sys

BASE="http://127.0.0.1:8188"

def api(path, data=None):
    url=BASE+path
    if data is None:
        return json.load(urllib.request.urlopen(url, timeout=15))
    req=urllib.request.Request(url, data=json.dumps(data).encode(),
                               headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))

PROMPT_TEXT=("A slow cinematic push-in on a rain-slick alley at night. "
             "A woman in a red coat walks toward camera, neon reflections on wet asphalt. "
             "Native audio: steady rain and distant traffic.")

def build(L, w=544, h=288, steps=8, seed=424242):
    return {
      "1":{"class_type":"UNETLoader","inputs":{"unet_name":"minimax_h3_fl2va_pruned_fp8_scaled.safetensors","weight_dtype":"default"}},
      "2":{"class_type":"LoraLoaderModelOnly","inputs":{"model":["1",0],"lora_name":"minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors","strength_model":1.0}},
      "3":{"class_type":"CLIPLoader","inputs":{"clip_name":"qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors","type":"minimax","device":"default"}},
      "4":{"class_type":"VAELoader","inputs":{"vae_name":"minimax_h3_video_vae_fp16.safetensors"}},
      "5":{"class_type":"VAELoader","inputs":{"vae_name":"minimax_h3_audio_vae_fp32.safetensors"}},
      "11":{"class_type":"MiniMaxH3ImageToVideo","inputs":{"clip":["3",0],"vae":["4",0],"prompt":PROMPT_TEXT,"width":w,"height":h,"length":L}},
      "12":{"class_type":"BasicScheduler","inputs":{"model":["2",0],"scheduler":"simple","steps":steps,"denoise":1.0}},
      "13":{"class_type":"KSamplerSelect","inputs":{"sampler_name":"res_multistep"}},
      "14":{"class_type":"RandomNoise","inputs":{"noise_seed":seed}},
      "15":{"class_type":"BasicGuider","inputs":{"model":["2",0],"conditioning":["11",0]}},
      "16":{"class_type":"SamplerCustomAdvanced","inputs":{"noise":["14",0],"guider":["15",0],"sampler":["13",0],"sigmas":["12",0],"latent_image":["11",1]}},
      "17":{"class_type":"VAEDecode","inputs":{"samples":["16",0],"vae":["4",0]}},
      "18":{"class_type":"VAEDecodeAudio","inputs":{"samples":["16",0],"vae":["5",0]}},
      "34":{"class_type":"CreateVideo","inputs":{"fps":24.0,"images":["17",0],"audio":["18",0]}},
      "35":{"class_type":"SaveVideo","inputs":{"video":["34",0],"filename_prefix":"m215_probe","format":"mp4","codec":"h264"}},
    }

def vram_sampler(stop, out):
    while not stop.is_set():
        try:
            r=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).strip().split("\n")[0]
            out.append(int(r))
        except Exception: pass
        time.sleep(0.5)

def run(L):
    wf=build(L)
    stop=threading.Event(); samples=[]
    t=threading.Thread(target=vram_sampler,args=(stop,samples),daemon=True); t.start()
    t0=time.time()
    try:
        r=api("/prompt",{"prompt":wf,"client_id":"m215-probe"})
        pid=r["prompt_id"]
    except urllib.error.HTTPError as e:
        stop.set()
        body=e.read().decode()[:1500]
        return {"length":L,"status":"REJECTED","http":e.code,"detail":body}
    status="unknown"
    for _ in range(3600):  # up to ~30 min
        time.sleep(2)
        try: h=api("/history/"+pid)
        except Exception: continue
        if pid in h:
            st=h[pid].get("status",{})
            status=st.get("status_str","done")
            break
    elapsed=time.time()-t0
    stop.set(); time.sleep(0.6)
    peak=max(samples) if samples else None
    st=h.get(pid,{}).get("status",{}) if 'h' in dir() else {}
    msgs=st.get("messages",[])
    ok_video=False
    outs=h.get(pid,{}).get("outputs",{}) if 'h' in dir() else {}
    if "35" in outs:
        ok_video=bool(outs["35"].get("videos") or outs["35"].get("gifs"))
    return {"length":L,"status":status,"elapsed_s":round(elapsed,1),
            "seconds_of_video":round(L/24.0,2),"s_per_video_s":round(elapsed/(L/24.0),3),
            "s_per_frame":round(elapsed/L,4),"vram_peak_mib":peak,"output_ok":ok_video,
            "messages":[m for m in msgs if m[0]!="execution_start"][-3:]}

if __name__=="__main__":
    # constraint on length
    oi=api("/object_info/MiniMaxH3ImageToVideo")
    keys=list(oi.keys())
    print("node:",keys)
    print("length spec:",json.dumps(oi[keys[0]]["input"]["required"].get("length")))
    results=[]
    for L in [int(x) for x in sys.argv[1:]] or [362]:
        print("=== length",L,"===",flush=True)
        r=run(L); results.append(r); print(json.dumps(r),flush=True)
        if r.get("status")=="REJECTED": break
    print("RESULTS_JSON="+json.dumps(results))
