set -u
D=/root/ComfyUI/models/loras
cd "$D"
mv -v sololo_emmamyers_kr2v1_raw.safetensors krea2_emmamyers_v1_onetrainer.safetensors 2>/dev/null
mv -v hmnsfw_spankme_h3_v1.safetensors SpankMe_H3_v1.safetensors 2>/dev/null
ls -la krea2_emmamyers_v1_onetrainer.safetensors SpankMe_H3_v1.safetensors
echo "=== inventory probe ==="
curl -s -m 30 "http://127.0.0.1:8188/playspace/lora/inventory" | python3 -c "
import sys,json
d=json.load(sys.stdin)
ls=d.get('loras',[])
print('count',len(ls),'free_GB',round((d.get('free_bytes') or 0)/1e9,1))
for x in ls:
    if 'emmamyers' in x['filename'] or 'SpankMe' in x['filename']:
        print(' ',x['filename'], round(x['size']/1e6,1),'MB')
" 2>&1 | tail -5
