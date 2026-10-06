# M187 — decode every Malcolm Rey H3 refmod into character reference frames (Blair: "go with #1 decode").
# Resumable: a character dir with frames is skipped. Run under /root/venv/bin/python from /root/ComfyUI.
import sys,os,time,glob,traceback
sys.path.insert(0,'/root/ComfyUI')
import torch
from PIL import Image
import comfy.utils, comfy.sd

VAE='/root/ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors'
REFMODS='/root/ComfyUI/models/refmods'
OUT='/root/ComfyUI/models/h3_char_tokens'
MAX_FRAMES=int(os.environ.get('MAX_FRAMES','0'))  # 0 = all
os.makedirs(OUT,exist_ok=True)
vae=comfy.sd.VAE(sd=comfy.utils.load_torch_file(VAE))
files=sorted(glob.glob(REFMODS+'/minimaxh3_*.safetensors'))
done=skip=fail=0; t0=time.time()
print(f'[{time.strftime("%H:%M:%S")}] start: {len(files)} refmods, vae device {vae.device}',flush=True)
for i,src in enumerate(files,1):
    name=os.path.basename(src)[:-len('.safetensors')]
    d=os.path.join(OUT,name)
    try:
        if glob.glob(d+'/f*.jpg'):
            skip+=1; continue
        os.makedirs(d,exist_ok=True)
        lat=comfy.utils.load_torch_file(src)['latent']
        with torch.no_grad(): img=vae.decode(lat)
        frames=img[0] if img.dim()==5 else img
        if MAX_FRAMES: frames=frames[:MAX_FRAMES]
        for k in range(frames.shape[0]):
            a=(frames[k].clamp(0,1).cpu().numpy()*255).astype('uint8')
            Image.fromarray(a).save(os.path.join(d,f'f{k:03d}.jpg'),'JPEG',quality=90)
        done+=1
        if done%25==0 or i==len(files):
            print(f'[{time.strftime("%H:%M:%S")}] {i}/{len(files)} decoded={done} skipped={skip} fail={fail} '
                  f'{(time.time()-t0)/max(done,1):.1f}s/char',flush=True)
    except Exception as e:
        fail+=1
        print(f'[{time.strftime("%H:%M:%S")}] FAIL {name}: {e}',flush=True)
        traceback.print_exc()
print(f'[{time.strftime("%H:%M:%S")}] DONE decoded={done} skipped={skip} fail={fail} in {(time.time()-t0)/60:.1f} min',flush=True)
