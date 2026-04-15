import os, json, io
import numpy as np
from PIL import Image, ImageFilter
from flask import Flask, render_template_string, request, jsonify
import sys

try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow as tf
        tflite = tf.lite

if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
    MODEL_PATH = os.path.join(BASE_DIR, "model_cnn.tflite")
    META_PATH = os.path.join(BASE_DIR, "model_meta.json")
else:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    MODEL_PATH = os.path.join(BASE_DIR, "data", "model_cnn.tflite")
    META_PATH = os.path.join(BASE_DIR, "data", "model_meta.json")

with open(META_PATH, encoding="utf-8") as f:
    meta = json.load(f)

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CATEGORIES = meta["categories"]
CAT_NAMES  = meta["category_names"]
IMG_SIZE   = meta["img_size"]

CAT_COLORS = {
    "plastic": "#f5c542",
    "paper":   "#4287f5",
    "glass":   "#42c960",
    "bio":     "#8B4513",
    "mixed":   "#555"
}
CAT_EMOJIS = {
    "plastic": "🟡",
    "paper":   "🔵",
    "glass":   "🟢",
    "bio":     "🟤",
    "mixed":   "⚫"
}
FEAT_LABELS = {
    "brightness":       "Brightness",
    "contrast":         "Contrast",
    "saturation":       "Saturation",
    "color_uniformity": "Color uniformity",
    "warm_ratio":       "Warm ratio",
    "transparency":     "Transparency",
    "dark_ratio":       "Dark ratio",
    "edge_density":     "Edge density",
    "edge_intensity":   "Edge intensity",
    "texture_roughness":"Texture roughness",
    "smoothness":       "Smoothness",
    "entropy":          "Entropy",
    "edge_entropy":     "Edge entropy",
    "channel_variance": "Channel variance",
    "highlights":       "Highlights",
}

def extract_features_for_display(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img).astype("float32")
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    bri = 0.299*r + 0.587*g + 0.114*b
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = np.where(mx > 0, (mx - mn)/(mx+1e-8), 0)
    gray = img.convert("L")
    ea = np.asarray(gray.filter(ImageFilter.FIND_EDGES)).astype("float32")
    da = np.asarray(gray.filter(ImageFilter.DETAIL)).astype("float32")
    emba = np.asarray(gray.filter(ImageFilter.EMBOSS)).astype("float32")
    h = np.histogram(bri.flatten(), bins=64, range=(0,255))[0]
    h = h / h.sum()
    h = h[h > 0]
    ent = float(-np.sum(h * np.log2(h)))
    eh = np.histogram(ea.flatten(), bins=32, range=(0,255))[0]
    eh = eh / eh.sum()
    eh = eh[eh > 0]
    eent = float(-np.sum(eh * np.log2(eh)))
    return {
        "brightness":       round(float(bri.mean()), 2),
        "contrast":         round(float(bri.std()), 2),
        "saturation":       round(float(sat.mean()), 4),
        "color_uniformity": round(float(sat.std()), 4),
        "warm_ratio":       round(float((r > b + 15).mean()), 4),
        "transparency":     round(float((bri > 210).mean()), 4),
        "dark_ratio":       round(float((bri < 40).mean()), 4),
        "edge_density":     round(float(ea.mean()), 2),
        "edge_intensity":   round(float(ea.std()), 2),
        "texture_roughness":round(float(da.std()), 2),
        "smoothness":       round(float(emba.std()), 2),
        "entropy":          round(ent, 4),
        "edge_entropy":     round(eent, 4),
        "channel_variance": round(float(np.var([r.mean(), g.mean(), b.mean()])), 2),
        "highlights":       round(float((bri > 240).mean()), 4),
    }

app = Flask(__name__)

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Waste Classifier</title>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#f0f4f0;--card:#fff;--border:#d4ddd4;--text:#1a2e1a;--dim:#6b8068;--accent:#2d7a3a;--al:#e8f5e9;--r:16px}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Outfit',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;display:flex;justify-content:center;padding:24px 16px}
.w{width:100%;max-width:540px}
h1{font-size:1.6rem;font-weight:800;text-align:center;margin-bottom:2px;color:var(--accent)}
.sub{color:var(--dim);font-size:.85rem;text-align:center;margin-bottom:20px}
.card{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:22px;margin-bottom:14px;box-shadow:0 2px 12px rgba(0,0,0,.04)}
.dz{border:3px dashed var(--border);border-radius:var(--r);padding:36px 20px;text-align:center;cursor:pointer;transition:all .3s;position:relative;background:var(--al)}
.dz:hover{border-color:var(--accent)}.dz input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.di{font-size:2.5rem;margin-bottom:6px}.dt{font-size:.88rem;color:var(--dim)}.dt strong{color:var(--text)}
.pv{display:none;margin-top:14px;text-align:center}.pv img{max-width:100%;max-height:200px;border-radius:12px}
.btn{width:100%;padding:13px;margin-top:14px;background:var(--accent);color:#fff;font-family:'Outfit',sans-serif;font-size:.95rem;font-weight:600;border:none;border-radius:12px;cursor:pointer;transition:all .2s}
.btn:hover{background:#236b2e}.btn:disabled{opacity:.4;cursor:not-allowed}
#result{display:none}@keyframes su{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:none}}
.rh{text-align:center;padding:16px 0 12px}.re{font-size:3rem;margin-bottom:6px}
.rc{font-size:1.4rem;font-weight:800;margin-bottom:4px}
.rb{font-size:.88rem;font-weight:600;padding:5px 18px;border-radius:999px;display:inline-block;color:#fff;margin-bottom:6px}
.rcf{font-size:.8rem;color:var(--dim)}
.ft{font-size:.72rem;color:var(--dim);text-transform:uppercase;letter-spacing:.06em;margin:14px 0 8px;font-weight:600}
.fg{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.fi{background:var(--al);border-radius:8px;padding:6px 10px;display:flex;justify-content:space-between;align-items:center}
.fn{font-size:.72rem;color:var(--dim)}.fv{font-family:'JetBrains Mono',monospace;font-size:.78rem;font-weight:500}
.pr{display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:.82rem}
.pl{width:62px;color:var(--dim);font-weight:500}.pb{flex:1;height:7px;background:#e8ece8;border-radius:4px;overflow:hidden}
.pf{height:100%;border-radius:4px;transition:width .6s}.pvl{width:36px;text-align:right;font-family:'JetBrains Mono',monospace;font-size:.76rem;color:var(--dim)}
.tp{margin-top:12px;padding:12px 14px;background:var(--al);border-radius:10px;font-size:.82rem;line-height:1.6}
.tp strong{color:var(--accent)}.tp ul{margin:4px 0 0 16px}.tp li{margin-bottom:3px}
.ag{display:block;text-align:center;margin-top:10px;color:var(--accent);font-weight:600;cursor:pointer;font-size:.86rem}
.inf{font-size:.74rem;color:var(--dim);text-align:center;line-height:1.6;padding:6px 0}.inf strong{color:var(--text)}
</style>
</head>
<body>
<div class="w">
<h1> Waste Classifier</h1>
<p class="sub">Take a photo of waste → AI tells you which bin it goes to</p>
<div class="card">
  <div class="dz"><input type="file" id="fi" accept="image/*" capture="environment" onchange="onP(this)">
    <div class="di"></div><div class="dt"><strong>Tap to take photo</strong> or drag & drop</div></div>
  <div class="pv" id="pv"><img id="pi"></div>
  <button class="btn" id="btn" onclick="go()" disabled>Classify waste</button>
</div>
<div id="result">
<div class="card" style="animation:su .4s ease">
  <div class="rh"><div class="re" id="re"></div><div class="rc" id="rc"></div>
    <div class="rb" id="rb"></div><div class="rcf" id="rcf"></div></div>
  <div class="ft">Image properties</div><div class="fg" id="fg"></div>
  <div class="ft" style="margin-top:14px">Category probabilities</div><div id="pb"></div>
  <div class="tp" id="tp"></div>
  <a class="ag" onclick="rs()">← Try another</a>
</div></div>
<div class="inf">CNN model accuracy: <strong>{{ acc }}%</strong>. Trained on <strong>{{ nd }}</strong> photos.</div>
</div>
<script>
let sf=null;
const cs={{ categories|tojson }},cn={{ cn|tojson }},cc={{ cc|tojson }},ce={{ ce|tojson }},fl={{ fl|tojson }};
const tips={plastic:'<strong>Plastic → yellow bin</strong><ul><li>Crush PET bottles</li><li>Greasy plastic → mixed</li></ul>',
  paper:'<strong>Paper → blue bin</strong><ul><li>Flatten boxes</li><li>Wet paper → mixed</li></ul>',
  glass:'<strong>Glass → green bin</strong><ul><li>Remove caps</li><li>Ceramics do NOT belong here</li></ul>',
  bio:'<strong>Bio → brown bin</strong><ul><li>Fruit peels, food scraps</li><li>Tea bags OK</li></ul>',
  mixed:'<strong>Mixed → black bin</strong><ul><li>Anything that doesn\'t fit elsewhere</li><li>Contaminated packaging</li></ul>'};
function onP(i){if(!i.files[0])return;sf=i.files[0];const r=new FileReader();
  r.onload=e=>{document.getElementById('pi').src=e.target.result;document.getElementById('pv').style.display='block';
  document.getElementById('btn').disabled=false;document.getElementById('result').style.display='none'};r.readAsDataURL(sf)}
async function go(){if(!sf)return;const b=document.getElementById('btn');b.disabled=true;b.textContent='Classifying...';
  try{const fd=new FormData();fd.append('photo',sf);const r=await fetch('/classify',{method:'POST',body:fd});const d=await r.json();
    if(d.error){alert(d.error);return}const cat=d.category;
    document.getElementById('re').textContent=ce[cat];document.getElementById('rc').textContent=cn[cat];
    document.getElementById('rb').textContent=cn[cat].split('(')[1]?.replace(')','');
    document.getElementById('rb').style.background=cc[cat];
    document.getElementById('rcf').textContent='Confidence: '+d.confidence+'%';
    let fg='';for(const[k,v]of Object.entries(d.features)){fg+=`<div class="fi"><span class="fn">${fl[k]||k}</span><span class="fv">${v}</span></div>`}
    document.getElementById('fg').innerHTML=fg;
    let pb='';for(let i=0;i<cs.length;i++){const c=cs[i],p=Math.round(d.probabilities[i]*100);
      pb+=`<div class="pr"><span class="pl">${c}</span><div class="pb"><div class="pf" style="width:${p}%;background:${cc[c]}"></div></div><span class="pvl">${p}%</span></div>`}
    document.getElementById('pb').innerHTML=pb;
    document.getElementById('tp').innerHTML=tips[cat]||'';
    document.getElementById('result').style.display='block';
    document.getElementById('result').scrollIntoView({behavior:'smooth'})
  }catch(e){alert('Error: '+e.message)}finally{b.disabled=false;b.textContent='Classify waste'}}
function rs(){sf=null;document.getElementById('pv').style.display='none';document.getElementById('result').style.display='none';
  document.getElementById('btn').disabled=true;document.getElementById('fi').value='';window.scrollTo({top:0,behavior:'smooth'})}
</script>
</body></html>"""

@app.route("/")
def index():
    return render_template_string(
        HTML,
        nd=meta["n_total"],
        acc=round(meta.get("accuracy_cnn", meta.get("accuracy", 0)) * 100, 1),
        categories=CATEGORIES,
        cn=CAT_NAMES,
        cc=CAT_COLORS,
        ce=CAT_EMOJIS,
        fl=FEAT_LABELS,
    )

@app.route("/classify", methods=["POST"])
def classify():
    if "photo" not in request.files:
        return jsonify({"error": "No photo"})
    try:
        img = Image.open(io.BytesIO(request.files["photo"].read())).convert("RGB")
        feats = extract_features_for_display(img)

        img_r = img.resize((IMG_SIZE, IMG_SIZE))
        arr = np.asarray(img_r).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)

        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()
        proba = interpreter.get_tensor(output_details[0]['index'])[0]

        cat_idx = int(np.argmax(proba))
        cat = CATEGORIES[cat_idx]
        return jsonify({
            "category": cat,
            "confidence": round(float(proba[cat_idx]) * 100, 1),
            "probabilities": [round(float(p), 4) for p in proba],
            "features": feats,
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    acc = meta.get("accuracy_cnn", meta.get("accuracy", 0))
    print(f"Waste Classifier (TFLite) | accuracy: {acc*100:.1f}% | http://localhost:5000")
    app.run(debug=False, port=5000)