"""Dataset captions written by a vision-language model through OpenRouter, in one of two styles.

"vgl": structured captions for Fibo 2.0, which is trained on Bria's VGL (Visual GenAI Language): one JSON object per
image. The prompt follows the one Bria's own captioner uses (github.com/Bria-AI/vgl, `Caption` in the Thales
dialect), reworded to describe a real image, and the result is put in the same key order Bria serialises VGL in.
"natural": a few plain, flowing sentences, for MOSS-Image, Krea 2 and other models trained on natural language.
"""
from __future__ import annotations

import base64
import io
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-3.5-flash-lite"  # the model bria-vgl uses by default
MAX_PIXELS = 1024 * 1024  # long enough to read small text in the image

SYSTEM_PROMPT = """\
You write image captions in Bria's VGL format: one JSON object that describes an image so precisely that an image \
model could redraw it from the caption alone.

How to write it
- Describe only what is actually visible in the image. Don't guess, invent or add things that aren't there.
- Use plain, concrete words: "rust-red wool coat with brass buttons", not "nice coat".
- Describe a person's left and right from the person's own point of view, not the viewer's.
- Don't repeat the same detail in several fields.
- Leave out keys that don't apply (for example the person-only keys for an object that isn't a person).
- Reply with the JSON object only: no markdown, no code fences, no text before or after it.

The keys, in this order
1. "short_description": a summary of the whole image in one to three sentences (200 words at most).
2. "objects": the prominent things in the scene (people, animals, items), most important first. Each one is an \
object with:
   - "description": what it is and what it looks like (100 words at most)
   - "location": where it is in the frame, e.g. "center", "top-left", "bottom-right foreground"
   - "relationship": how it relates to the other objects, e.g. "held in the woman's right hand"
   - "relative_size": e.g. "small", "medium", "large within frame"
   - "shape_and_color": its basic shape and main colors
   - "texture": e.g. "smooth", "rough", "metallic", "furry"
   - "appearance_details": any other notable visual details
   - "orientation": e.g. "upright", "tilted 45 degrees", "facing left", "lying on its side"
   - only for a group of similar things: "number_of_objects" (an integer)
   - only for a person or human-like figure: "pose", "expression", "clothing", "action", "gender" and \
"skin_tone_and_texture"
3. "background_setting": the environment and background, including background elements not listed in "objects".
4. "lighting": an object with "conditions" (e.g. "golden hour", "studio lighting"), "direction" (e.g. "backlit", \
"side-lit from left") and "shadows" (e.g. "long, soft shadows").
5. "aesthetics": an object with "composition" (e.g. "rule of thirds", "centered", "close-up"; give the shot type when \
people are the subject), "color_scheme" (e.g. "warm complementary colors", "muted pastels") and "mood_atmosphere" \
(e.g. "serene", "mysterious").
6. "photographic_characteristics": an object with "depth_of_field" (e.g. "shallow", "deep"), "focus" (e.g. "sharp \
focus on subject", "motion blur"), "camera_angle" (e.g. "eye-level", "low angle") and "lens_focal_length" (e.g. \
"wide-angle", "portrait lens (50mm-85mm)", "macro").
7. "style_medium": the medium, e.g. "photograph", "oil painting", "3D render", "digital illustration", "anime", \
"pencil sketch".
8. "text_render": every piece of readable text in the image, one entry per line, label or sign, copied exactly as \
written. Use [] when the image has no text. Each entry is an object with "text", "location", "size", "color", \
"font" (e.g. "bold sans-serif", "handwritten script") and "appearance_details".
9. "context": what kind of image this is, e.g. "a candid street photograph", "concept art for a fantasy video game".
10. "artistic_style": the style in 3 words at most, e.g. "impressionistic, vibrant, textured"."""

USER_PROMPT = "Analyze the provided image and write its VGL caption as specified in your instructions."

NATURAL_PROMPT = """\
You write captions for training an image generation model. Describe the image in simple, natural English: two to \
four flowing sentences, 40 to 90 words, that read like a person describing the picture to a friend.

How to write it
- Start with the main subject, then describe, in this order: their hair, clothing and accessories (with colors), \
any tattoos or other distinctive features, what they are holding or doing, their pose and expression, then the \
setting, the time of day and the lighting.
- Describe only what is actually visible. Don't guess names, ages, identities or what isn't in the picture.
- Use plain, concrete words: "a fitted white mini skirt", not "a stunning outfit". No flowery or poetic language.
- Describe a person's left and right from the person's own point of view, not the viewer's.
- If it isn't a photograph, say what it is near the start (e.g. "a 2D cartoon illustration of…", "a 3D render of…").
- Write out any clearly readable text in quotes.
- Name a brand only if its name or logo is clearly readable.
- Don't start with "The image shows" or "This is a photo of", and don't use lists, line breaks or headings.

Example of the style (for a different picture):
A man with short curly brown hair and a trimmed beard sits at a small café table, wearing a navy denim jacket over \
a white T-shirt. He holds a white coffee cup in his right hand and looks out the window with a relaxed smile. \
Warm afternoon sunlight falls across the wooden table, and a blurred street with parked bicycles is visible \
behind the glass.

Reply with the caption only."""

NATURAL_USER_PROMPT = "Write the caption for this image."
STYLES = ("vgl", "natural")

# Bria's VGL key order (bria_vgl.model.VGL), used to serialise captions the way Fibo 2.0 saw them.
_ORDER = ["short_description", "objects", "background_setting", "lighting", "aesthetics",
          "photographic_characteristics", "style_medium", "text_render", "context", "artistic_style"]
_SUB_ORDER = {
    "objects": ["description", "location", "relationship", "relative_size", "shape_and_color", "texture",
                "appearance_details", "number_of_objects", "pose", "expression", "clothing", "action", "gender",
                "skin_tone_and_texture", "orientation"],
    "lighting": ["conditions", "direction", "shadows"],
    "aesthetics": ["composition", "color_scheme", "mood_atmosphere"],
    "photographic_characteristics": ["depth_of_field", "focus", "camera_angle", "lens_focal_length"],
    "text_render": ["text", "location", "size", "color", "font", "appearance_details"],
}


def _schema() -> dict:
    """JSON schema of the reply (Bria's output schema; optional keys aren't required)."""
    s = {"type": "string"}

    def obj(keys, required):
        return {"type": "object", "properties": {k: dict(s) for k in keys}, "required": required}
    objects = obj(_SUB_ORDER["objects"], _SUB_ORDER["objects"][:7])
    objects["properties"]["number_of_objects"] = {"type": "integer"}
    return {"type": "object", "properties": {
        "short_description": s, "objects": {"type": "array", "items": objects}, "background_setting": s,
        "lighting": obj(_SUB_ORDER["lighting"], _SUB_ORDER["lighting"]),
        "aesthetics": obj(_SUB_ORDER["aesthetics"], _SUB_ORDER["aesthetics"]),
        "photographic_characteristics": obj(_SUB_ORDER["photographic_characteristics"],
                                            _SUB_ORDER["photographic_characteristics"]),
        "style_medium": s, "text_render": {"type": "array", "items": obj(_SUB_ORDER["text_render"],
                                                                         _SUB_ORDER["text_render"][:5])},
        "context": s, "artistic_style": s,
    }, "required": _ORDER}


class CaptionError(Exception):
    pass


def image_data_url(path: Path) -> str:
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        im = im.convert("RGB")
        if im.width * im.height > MAX_PIXELS:
            scale = (MAX_PIXELS / (im.width * im.height)) ** 0.5
            im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def tidy(caption: dict) -> dict:
    """Bria's key order, without empty values (Fibo 2.0's pipeline drops those anyway); text_render stays even
    when empty, since an empty list says the image has no text."""
    def empty(v):
        return v is None or (isinstance(v, (str, dict, list)) and not v)

    def ordered(d: dict, order: list[str]) -> dict:
        out = {k: d[k] for k in order if k in d}
        out.update({k: v for k, v in d.items() if k not in out})
        return {k: v for k, v in out.items() if not empty(v)}
    out = {}
    for key in _ORDER + [k for k in caption if k not in _ORDER]:
        if key not in caption:
            continue
        v = caption[key]
        sub = _SUB_ORDER.get(key)
        if sub and isinstance(v, list):
            v = [ordered(x, sub) for x in v if isinstance(x, dict)]
            v = [x for x in v if x]
        elif sub and isinstance(v, dict):
            v = ordered(v, sub)
        if key == "text_render" and v == []:
            out[key] = v
        elif not empty(v):
            out[key] = v
    return out


def parse_reply(text: str) -> dict:
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise CaptionError("the model didn't reply with a JSON object")
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise CaptionError(f"the model's JSON is invalid ({e})") from None
    if not isinstance(obj, dict) or not obj.get("short_description"):
        raise CaptionError("the model's JSON has no short_description")
    return tidy(obj)


def _post(api_key: str, body: dict, timeout: float = 180) -> dict:
    req = urllib.request.Request(OPENROUTER_URL, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
        "X-Title": "MOSS LoRA Trainer",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail).get("error", {}).get("message", detail)
        except (json.JSONDecodeError, AttributeError):
            pass
        raise CaptionError(f"OpenRouter HTTP {e.code}: {str(detail)[:300]}") from None
    except (urllib.error.URLError, TimeoutError) as e:
        raise CaptionError(f"OpenRouter unreachable: {e}") from None


def parse_natural(text: str) -> str:
    """A plain caption: one paragraph, without quotes, fences or a leading "Caption:"."""
    text = re.sub(r"```\w*", "", text or "").strip()
    text = re.sub(r"^(caption|description)\s*:\s*", "", text, flags=re.I)
    text = " ".join(text.split()).strip().strip('"“”').strip()
    if len(text) < 10:
        raise CaptionError("the model's caption is empty")
    return text


def caption_image(path: Path, api_key: str, model: str = DEFAULT_MODEL, extra: str = "", attempts: int = 4,
                  style: str = "vgl") -> dict | str:
    """Caption one image: a VGL dict, or a string for style "natural". `extra` is appended to the instructions
    (e.g. "The woman is called Mira.")."""
    if not api_key:
        raise CaptionError("Add your OpenRouter API key in Settings first.")
    natural = style == "natural"
    system = (NATURAL_PROMPT if natural else SYSTEM_PROMPT) + (
        f"\n\nAdditional instructions from the user:\n{extra.strip()}" if extra.strip() else "")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": [{"type": "text", "text": NATURAL_USER_PROMPT if natural else USER_PROMPT},
                                             {"type": "image_url", "image_url": {"url": image_data_url(path)}}]}]
    formats = [None] if natural else [
        {"type": "json_schema", "json_schema": {"name": "vgl_caption", "schema": _schema()}}, {"type": "json_object"}, None]
    last = None
    for attempt in range(attempts):
        body = {"model": model or DEFAULT_MODEL, "messages": messages, "temperature": 0.2,
                "max_tokens": 1000 if natural else 8000}
        if formats[0]:
            body["response_format"] = formats[0]
        try:
            reply = _post(api_key, body)
            if reply.get("error"):
                raise CaptionError(f"OpenRouter: {reply['error'].get('message', reply['error'])}")
            content = reply["choices"][0]["message"].get("content") or ""
            return parse_natural(content) if natural else parse_reply(content)
        except CaptionError as e:
            last = e
            msg = str(e)
            if "HTTP 401" in msg or "HTTP 402" in msg or "HTTP 403" in msg:
                raise  # bad key / no credits: retrying won't help
            if "HTTP 400" in msg and len(formats) > 1 and ("response_format" in msg or "schema" in msg.lower()):
                formats.pop(0)  # this model doesn't take structured output: ask for plain JSON instead
                continue
            time.sleep(2 * (attempt + 1))
        except (KeyError, IndexError, TypeError):
            last = CaptionError("unexpected reply from OpenRouter")
    raise last or CaptionError("captioning failed")


def check_key(api_key: str) -> dict:
    """OpenRouter's info about a key (label, usage, limit); raises CaptionError if it's invalid."""
    req = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode()).get("data", {})
    except urllib.error.HTTPError as e:
        raise CaptionError(f"OpenRouter rejected the key (HTTP {e.code})") from None
    except (urllib.error.URLError, TimeoutError) as e:
        raise CaptionError(f"OpenRouter unreachable: {e}") from None


def caption_text(caption: dict | str) -> str:
    """How a caption is stored in the dataset's .txt file (VGL as readable JSON; training re-serialises it)."""
    return caption if isinstance(caption, str) else json.dumps(caption, ensure_ascii=False, indent=2)
