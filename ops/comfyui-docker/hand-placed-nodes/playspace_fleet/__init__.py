"""PlaySpace fleet helper — on-demand lora pull, inventory, eviction, cache refresh.

Routes (served by ComfyUI's aiohttp app on :8188):
  POST /playspace/lora/pull      {"filename": "...", "url": "..."}  -> {"state": ...}
  GET  /playspace/lora/status?filename=...                          -> {"state","size","bytes","error"}
  GET  /playspace/lora/inventory                                    -> {"dir","free_bytes","loras":[...]}
  POST /playspace/lora/delete    {"filename": "..."}                -> {"deleted","bytes"}
  POST /playspace/lora/refresh                                      -> {"ok","count"}

State per filename: absent -> downloading -> present | failed
Downloads are serialised (one at a time) and written atomically (.part -> os.replace).
"""
import asyncio
import os
import time
import urllib.request

from aiohttp import web

import folder_paths
from server import PromptServer

HF_BASE = "https://huggingface.co/malcolmrey/krea2/resolve/main/"
LORA_DIR = folder_paths.get_folder_paths("loras")[0]
MIN_BYTES = 50_000_000
USER_AGENT = "playspace-fleet/1"
TIMEOUT = 300

_LOCK = asyncio.Lock()
_STATE: dict = {}

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}


def lora_path(filename: str) -> str:
    return os.path.join(LORA_DIR, filename)


def lora_present(filename: str) -> bool:
    try:
        return os.path.getsize(lora_path(filename)) >= MIN_BYTES
    except OSError:
        return False


def refresh_cache() -> int:
    folder_paths.cache_helper.clear()
    try:
        folder_paths.filename_list_cache.clear()
    except Exception:
        pass
    return len(folder_paths.get_filename_list("loras"))


def _download_blocking(filename: str, url: str) -> int:
    dest = lora_path(filename)
    tmp = dest + ".part"
    os.makedirs(LORA_DIR, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    written = 0
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            written += len(chunk)
    if written < MIN_BYTES:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise RuntimeError(f"short download: {written} bytes")
    os.replace(tmp, dest)
    return written


async def _pull_task(filename: str, url: str) -> None:
    st = _STATE[filename]
    try:
        async with _LOCK:
            if lora_present(filename):
                st.update(state="present", bytes=os.path.getsize(lora_path(filename)))
                return
            st.update(state="downloading", started=time.time())
            n = await asyncio.to_thread(_download_blocking, filename, url)
            refresh_cache()
            st.update(state="present", bytes=n, finished=time.time())
    except Exception as e:  # noqa: BLE001
        st.update(state="failed", error=str(e), finished=time.time())


routes = PromptServer.instance.routes


@routes.post("/playspace/lora/pull")
async def pull(request):
    data = await request.json()
    filename = str(data.get("filename", "")).strip()
    if not filename or "/" in filename or "\\" in filename or not filename.endswith(".safetensors"):
        return web.json_response({"error": "bad filename"}, status=400)
    url = data.get("url") or (HF_BASE + filename)
    if lora_present(filename):
        return web.json_response({"state": "present", "size": os.path.getsize(lora_path(filename))})
    st = _STATE.get(filename)
    if st is None or st.get("state") == "failed":
        _STATE[filename] = st = {"state": "downloading", "bytes": 0, "error": None, "queued": time.time()}
        asyncio.ensure_future(_pull_task(filename, url))
    return web.json_response({"state": st["state"], "bytes": st.get("bytes", 0)})


@routes.get("/playspace/lora/status")
async def status(request):
    filename = request.query.get("filename", "")
    if lora_present(filename):
        return web.json_response({"state": "present", "size": os.path.getsize(lora_path(filename))})
    st = _STATE.get(filename)
    if st is None:
        return web.json_response({"state": "absent"})
    if st.get("state") == "downloading":
        try:
            st["bytes"] = os.path.getsize(lora_path(filename) + ".part")
        except OSError:
            pass
    return web.json_response({"state": st["state"], "bytes": st.get("bytes", 0), "error": st.get("error")})


@routes.get("/playspace/lora/inventory")
async def inventory(request):
    items = []
    try:
        for fn in os.listdir(LORA_DIR):
            p = os.path.join(LORA_DIR, fn)
            if not (fn.endswith(".safetensors") and os.path.isfile(p)):
                continue
            s = os.stat(p)
            items.append({"filename": fn, "size": s.st_size, "mtime": s.st_mtime})
    except OSError:
        pass
    try:
        du = os.statvfs(LORA_DIR)
        free = du.f_bavail * du.f_frsize
    except OSError:
        free = None
    return web.json_response({"dir": LORA_DIR, "min_bytes": MIN_BYTES, "free_bytes": free, "loras": items})


@routes.post("/playspace/lora/delete")
async def delete(request):
    data = await request.json()
    filename = str(data.get("filename", "")).strip()
    if not filename or "/" in filename or "\\" in filename or filename.endswith(".part"):
        return web.json_response({"error": "bad filename"}, status=400)
    p = lora_path(filename)
    try:
        n = os.path.getsize(p)
        os.remove(p)
    except OSError as e:
        return web.json_response({"deleted": False, "error": str(e)}, status=404)
    refresh_cache()
    return web.json_response({"deleted": True, "bytes": n})


@routes.post("/playspace/lora/refresh")
async def refresh(request):
    return web.json_response({"ok": True, "count": refresh_cache()})
