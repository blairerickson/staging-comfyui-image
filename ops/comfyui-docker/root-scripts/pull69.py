import os,sys,time,urllib.request
BASE='https://huggingface.co/malcolmrey/krea2/resolve/main/'
DST='/root/ComfyUI/models/loras'
names=[l.strip() for l in open(sys.argv[1]) if l.strip()]
os.makedirs(DST,exist_ok=True)
ok=skip=fail=0; t0=time.time(); bytes_=0
for i,n in enumerate(names,1):
    p=os.path.join(DST,n)
    if os.path.exists(p) and os.path.getsize(p)>50_000_000: skip+=1; continue
    try:
        req=urllib.request.Request(BASE+n,headers={'User-Agent':'chief'})
        tmp=p+'.part'
        with urllib.request.urlopen(req,timeout=300) as r, open(tmp,'wb') as f:
            while True:
                b=r.read(1<<20)
                if not b: break
                f.write(b); bytes_+=len(b)
        os.replace(tmp,p); ok+=1
        print(f'[{i}/{len(names)}] {n} {os.path.getsize(p)/1e6:.1f} MB  ({bytes_/1e9:.2f} GB, {(time.time()-t0)/60:.1f} min)',flush=True)
    except Exception as e:
        fail+=1; print(f'[{i}/{len(names)}] FAIL {n}: {e}',flush=True)
print(f'DONE ok={ok} skip={skip} fail={fail} {bytes_/1e9:.2f} GB in {(time.time()-t0)/60:.1f} min')
