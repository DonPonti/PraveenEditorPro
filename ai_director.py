"""Story-led AI edit planner for Praveen Editor."""
import copy, json
from pathlib import Path
import cv2
import numpy as np
import vlog_core as core

MODEL = "gemini-2.5-flash"
STYLE_MAP = {"Cinematic":"cinema","Documentary":"clean","Fast social":"gritty","Emotional":"warm","Product advertisement":"teal","Vlog":"clean"}

def _frames(path, kind, duration):
    found = []
    if kind == "photo":
        frame = core.thumbnail(path, 0, 720)
        if frame is not None: found.append((0.0, frame))
    else:
        cap = cv2.VideoCapture(str(path))
        for fraction in (0.22, 0.72):
            t = max(0.0, min(duration - 0.05, duration * fraction))
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                if w > 720: frame = cv2.resize(frame, (720, max(1, int(h * 720 / w))))
                found.append((t, frame))
        cap.release()
    return found

def _parse(text):
    value = (text or "").strip()
    if value.startswith("```"): value = value.split("\n", 1)[-1].removesuffix("```").strip()
    try: return json.loads(value)
    except Exception as exc: raise core.EditorError("Gemini returned an invalid edit plan. Please try again.") from exc

def create_ai_edit(media_paths, base_project, brief, target_length=30, visual_style="Cinematic", api_key=None, cb=None):
    cb = cb or (lambda *args: None)
    key = (api_key or "").strip()
    if not key: raise core.EditorError("Add your Gemini API key in the AI Creator tab.")
    if not brief.strip(): raise core.EditorError("Describe the video you want the AI Director to make.")
    if not base_project.music or not Path(base_project.music).exists(): raise core.EditorError("Choose background music in Settings first.")
    try: from google import genai
    except ImportError as exc: raise core.EditorError("Install Gemini SDK: python -m pip install -U google-genai") from exc
    paths = core.scan_paths(media_paths)[:18]
    if not paths: raise core.EditorError("Add video clips or photos in the Media tab first.")
    records, parts = [], []
    for n, path in enumerate(paths):
        kind = core.media_kind(path)
        try: duration = core.media_duration(path) if kind == "video" else 5.0
        except Exception: continue
        local_parts = []
        for t, frame in _frames(path, kind, duration):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 76])
            if ok: local_parts.append((t, f"CLIP {len(records)} {Path(path).name} sample at {t:.2f}s; sharpness {min(sharp,1000):.1f}; brightness {gray.mean():.1f}", encoded.tobytes()))
        if local_parts:
            records.append({"id":len(records),"name":Path(path).name,"kind":kind,"duration":round(duration,2),"path":str(path)})
            parts.extend((label, data) for _, label, data in local_parts)
        cb(f"Inspecting footage {n+1}/{len(paths)}", 0.05 + 0.4*(n+1)/len(paths))
    if not records or not parts: raise core.EditorError("Could not read usable frames from your media.")
    inventory = [{k:v for k,v in r.items() if k != "path"} for r in records]
    aspect = "9:16 vertical" if base_project.format == "vertical" else "16:9 horizontal"
    prompt = f"""You are a premium film editor. Create a coherent, story-led edit from the supplied sampled frames.
CREATIVE BRIEF: {brief}
TARGET: {int(target_length)} seconds. FORMAT: {aspect}. STYLE: {visual_style}.
Use only visible content; do not invent events, dialogue, locations, or claims. Build hook, context, development, strongest moment and ending. Vary shot length and avoid random montage sequencing. Keep shots 0.7–8 seconds; speed 0.85–1.25 normally. For video source_start/source_end are seconds in that clip; timeline duration=(end-start)/speed. For photos set source range to 0 and provide duration.
CLIP INVENTORY: {json.dumps(inventory, ensure_ascii=False)}
Return ONLY JSON: {{"title":"...","hook":"...","outro":"...","editorial_summary":"...","shots":[{{"clip_id":0,"source_start":0.5,"source_end":3.5,"duration":3,"speed":1,"section":"intro","role":"opening hook","reason":"...","move":"push","effects":[]}}]}}
Allowed section: intro/build/peak/cooldown. Allowed move: none/push/pull/punch/shake. Allowed effects: flash/glitch/whip/ramp/dip_in/dip_out. For photos use source_start=0, source_end=0 and duration. Use enough shots to approach target length. No markdown."""
    cb("Sending sampled still frames and creative brief to Gemini…", 0.55)
    contents = [prompt]
    try:
        for label, data in parts:
            contents.extend([label, genai.types.Part.from_bytes(data=data, mime_type="image/jpeg")])
        client = genai.Client(api_key=key)
        response = client.models.generate_content(model=MODEL, contents=contents, config=genai.types.GenerateContentConfig(response_mime_type="application/json", temperature=0.35))
    except Exception as exc: raise core.EditorError(f"Gemini AI Director failed: {type(exc).__name__}: {exc}") from exc
    plan = _parse(getattr(response, "text", ""))
    shots = plan.get("shots")
    if not isinstance(shots, list) or not shots: raise core.EditorError("Gemini did not return any usable shots. Try a clearer brief.")
    section_ids = {"intro":0,"build":1,"peak":2,"cooldown":3}
    moves = set(core.MOVE_CHOICES); fx_allowed = {"flash","glitch","whip","ramp","dip_in","dip_out"}
    items, notes = [], []
    for shot in shots[:36]:
        try:
            cid = int(shot.get("clip_id"))
            if cid < 0 or cid >= len(records): continue
            rec = records[cid]; speed = min(2.0,max(0.5,float(shot.get("speed",1))))
            section = section_ids.get(str(shot.get("section","build")).lower(),1); kind = rec["kind"]
            if kind == "video":
                start = max(0.0,float(shot.get("source_start",0))); end = min(rec["duration"],float(shot.get("source_end",0)))
                if end <= start + 0.25: continue
                span = min(end-start,16.0); end = start+span; duration = span/speed
            else: start=0.0; speed=1.0; duration=min(6.0,max(1.2,float(shot.get("duration",3))))
            duration = min(8.0,max(0.7,duration))
            effects = set(shot.get("effects",[])) & fx_allowed if isinstance(shot.get("effects",[]),list) else set()
            move = str(shot.get("move","none")).lower(); move = move if move in moves else "none"
            role = str(shot.get("role","")).lower(); hero = section==2 or "hook" in role or "climax" in role or "hero" in role
            items.append(core.Item(source=rec["path"],kind=kind,src_start=round(start,3),duration=round(duration,3),speed=round(speed,3),section=section,hero=hero,move=move if move!="none" else None,flash="flash" in effects,glitch="glitch" in effects,whip="whip" in effects,ramp="ramp" in effects,dip_in="dip_in" in effects,dip_out="dip_out" in effects,reframe="smart" if base_project.format=="vertical" else "fill"))
            notes.append({"clip":rec["name"],"role":str(shot.get("role","")),"reason":str(shot.get("reason","")),"duration":round(duration,2)})
        except (TypeError,ValueError,IndexError,AttributeError): continue
    if not items: raise core.EditorError("The AI plan did not contain valid source ranges. Try again.")
    target=max(5.0,float(target_length)); total=sum(i.duration for i in items)
    if total < target*0.55: raise core.EditorError(f"The AI planned only {total:.1f}s. Add more footage or request a shorter video.")
    overflow=total-target
    for item in reversed(items):
        if overflow <= 0: break
        reducible=max(0.0,item.duration-0.7); reduction=min(reducible,overflow); item.duration=round(item.duration-reduction,3); overflow-=reduction
    project=copy.deepcopy(base_project); project.items=items; project.style=STYLE_MAP.get(visual_style,"cinema")
    project.effects="hype" if visual_style=="Fast social" else "subtle"; project.title=str(plan.get("title",""))[:80]; project.subtitle=str(plan.get("hook",""))[:120]; project.outro=str(plan.get("outro",""))[:120]; project.captions=[]
    project.reframe="smart" if project.format=="vertical" else "fill"
    cb("AI edit ready — review the timeline before exporting.",1.0)
    return {"project":project,"plan":{"editorial_summary":str(plan.get("editorial_summary","")),"shots":notes,"model":MODEL,"sampled_frames_sent":len(parts),"original_media_uploaded":False}}

