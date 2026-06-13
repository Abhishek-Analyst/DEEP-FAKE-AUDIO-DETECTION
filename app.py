# app.py — AuraShield deepfake-audio detector (3-seed LFCC ensemble)
import os, tempfile, numpy as np, streamlit as st
import torch, torch.nn as nn, torchaudio, librosa, soundfile as sf

# ---------- constants (must match training) ----------
SR=16000; TARGET_LEN=int(SR*4.0); N_LFCC=60; TRIM_DB=30
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
FIXED_THRESHOLD=0.06   # prior-median decision threshold (for single-file inference)

# ---------- model (identical to the training notebook) ----------
class AuraNet(nn.Module):
    def __init__(self, n_lfcc=N_LFCC):
        super().__init__()
        self.lfcc=torchaudio.transforms.LFCC(sample_rate=SR, n_lfcc=n_lfcc,
                    speckwargs={"n_fft":512,"win_length":400,"hop_length":160})
        self.bn0=nn.BatchNorm2d(3)
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),
                                           nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.body=nn.Sequential(blk(3,16),blk(16,32),blk(32,64),blk(64,128))
        self.drop=nn.Dropout(0.3); self.fc=nn.Linear(128,1)
    def forward(self, wave):
        x=self.lfcc(wave)
        d1=torchaudio.functional.compute_deltas(x); d2=torchaudio.functional.compute_deltas(d1)
        x=torch.stack([x,d1,d2],1); x=self.bn0(x)
        x=self.body(x); x=torch.amax(x,dim=(2,3)); x=self.drop(x)
        return self.fc(x).squeeze(1)

def load_fixed_wave(path, target_len=TARGET_LEN, sr=SR, trim_db=TRIM_DB):
    try:
        wav,fsr=torchaudio.load(path)
    except Exception:
        y,fsr=sf.read(path); wav=torch.tensor(np.asarray(y).T if np.ndim(y)>1 else y).float()
        if wav.ndim==1: wav=wav.unsqueeze(0)
    if wav.shape[0]>1: wav=wav.mean(0,keepdim=True)
    if fsr!=sr: wav=torchaudio.functional.resample(wav,fsr,sr)
    y=wav.squeeze(0).numpy().astype(np.float32)
    yt,_=librosa.effects.trim(y,top_db=trim_db)
    if yt.size<sr*0.25: yt=y
    if yt.size<target_len: yt=np.tile(yt,int(np.ceil(target_len/yt.size)))
    s=(yt.size-target_len)//2; yt=yt[s:s+target_len]
    m=np.max(np.abs(yt)); yt = yt/m if m>0 else yt
    return yt.astype(np.float32)

@st.cache_resource
def load_models():
    ms=[]
    for s in range(3):
        m=AuraNet()
        m.load_state_dict(torch.load(f"ens_seed{s}.pt", map_location=DEVICE, weights_only=True))
        m.eval().to(DEVICE); ms.append(m)
    return ms

@torch.no_grad()
def predict(path, models, thr=FIXED_THRESHOLD):
    x=torch.from_numpy(load_fixed_wave(path)[None]).to(DEVICE)
    scores=[float(torch.sigmoid(m(x)).item()) for m in models]
    p_fake=float(np.mean(scores))
    return ("FAKE (AI-generated)" if p_fake>=thr else "REAL (human)"), p_fake, scores

# ---------- UI ----------
st.set_page_config(page_title="AuraShield", page_icon="🛡️")
st.title("🛡️ AuraShield — Deepfake Audio Detector")
st.caption("3-seed LFCC ensemble · Test EER 10.73% · Accuracy 89.2% · Macro-F1 0.892")

try:
    models=load_models()
except Exception as e:
    st.error(f"Could not load models — are ens_seed0/1/2.pt next to app.py?\n\n{e}"); st.stop()

up=st.file_uploader("Upload an audio clip (.wav / .flac recommended)", type=["wav","flac","mp3"])
if up:
    st.audio(up)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as tmp:
        tmp.write(up.read()); path=tmp.name
    if st.button("Analyze", type="primary"):
        with st.spinner("Analyzing..."):
            label,p,scores=predict(path, models)
        (st.error if "FAKE" in label else st.success)(f"### {label}")
        thr = FIXED_THRESHOLD                                # map threshold -> 50% confidence
        conf = 0.5 + 0.5*((p-thr)/max(1e-6,1-thr) if p >= thr else (thr-p)/max(1e-6,thr))
        c1, c2 = st.columns(2)
        c1.metric("Verdict confidence", f"{conf*100:.1f}%")
        c2.metric("P(fake)", f"{p:.3f}", help=f"decision threshold = {FIXED_THRESHOLD:.3f}")
        st.progress(float(min(max(conf, 0.0), 1.0)))
        st.caption("per-model P(fake): " + ", ".join(f"{s:.3f}" for s in scores))

st.divider()
if os.path.exists("confusion_matrix.png"):
    with st.expander("Model performance (test set)"):
        st.image("confusion_matrix.png")
