
import json,base64,sys,urllib.request
MODEL=sys.argv[2] if len(sys.argv)>2 else "ps-promptwriter:latest"
Q=("You are a strict visual auditor looking at a single generated image. Answer with ONLY compact JSON.\n"
   "Report presence/absence of each element and the count of zombies:\n"
   "naked_woman (any nude/partly-nude female figure visible), rope_on_wrists, carrot, red_apple_in_mouth,\n"
   "giant_apple_centerpiece (a huge apple dominating the frame), zombies (integer count of rotting/undead figures),\n"
   "forks_or_knives, silver_platter, bed_of_vegetables, restaurant_interior, sunlight.\n"
   "Then subjects_description = <=25 words describing what is actually in the image.\n"
   'JSON schema: {"naked_woman":bool,"rope_on_wrists":bool,"carrot":bool,"red_apple_in_mouth":bool,'
   '"giant_apple_centerpiece":bool,"zombies":int,"forks_or_knives":bool,"silver_platter":bool,'
   '"bed_of_vegetables":bool,"restaurant_interior":bool,"sunlight":bool,"subjects_description":"str"}')
b64=base64.b64encode(open(sys.argv[1],'rb').read()).decode()
body=json.dumps({"model":MODEL,"prompt":Q,"images":[b64],"stream":False,"options":{"temperature":0}}).encode()
req=urllib.request.Request("http://127.0.0.1:11434/api/generate",data=body,headers={"Content-Type":"application/json"})
d=json.loads(urllib.request.urlopen(req,timeout=600).read())
print(d.get("response",""))
