import streamlit as st
import os
import io
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="NyanCheck", page_icon="🐾", layout="wide")

# -------------------------------------------------
# predict.py & selectimage.py 読み込み
# -------------------------------------------------
try:
    from predict import predict  # 提供ファイル
except Exception:
    predict = None

try:
    from selectimage import randomselect
except Exception:
    randomselect = None

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "./uploads")).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = Path("results/nyancheck.h5").resolve()
LABELS_PATH = Path("results/labels.txt").resolve()
TARGET_SIZE = (200, 150)

# -------------------------------------------------
# 高速化モデル読込
# -------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_model_once():
    try:
        if not MODEL_PATH.exists():
            return None
        from tensorflow.keras.models import load_model  # type: ignore
        return load_model(str(MODEL_PATH))
    except Exception:
        return None

def _load_labels():
    if LABELS_PATH.exists():
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    return None

def _preprocess_pil(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(TARGET_SIZE, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

def _predict_fast_batch(paths: list[Path]) -> list[str]:
    model = _load_model_once()
    if model is None:
        raise RuntimeError("Model not available")
    arrays, valid_idx = [], []
    for i, p in enumerate(paths):
        try:
            img = Image.open(p)
            arr = _preprocess_pil(img)
            arrays.append(arr)
            valid_idx.append(i)
        except Exception:
            arrays.append(None)
    batch = np.stack([a for a in arrays if a is not None], axis=0)
    preds = model.predict(batch, verbose=0)
    idxs = preds.argmax(axis=1) if preds.ndim == 2 else np.zeros((preds.shape[0],), dtype=np.int64)
    label_table = _load_labels()
    mapped = [label_table[i] if label_table and 0 <= i < len(label_table) else str(i) for i in idxs]
    out = ["?"] * len(paths)
    j = 0
    for i in range(len(paths)):
        if i in valid_idx:
            out[i] = mapped[j]
            j += 1
    return out

# ===============================
# Check ページ
# ===============================

def page_check():
    st.header("Check（画像推論）")
    with st.form(key="check_form"):
        uploaded_files = st.file_uploader("画像をアップロード（複数可）", type=["png","jpg","jpeg","webp"], accept_multiple_files=True)
        save_to_uploads = st.checkbox("/uploads に保存する", value=True)
        use_fast = st.checkbox("高速モード", value=True)
        submitted = st.form_submit_button("推論を実行")

    if submitted and uploaded_files:
        file_paths = []
        for uf in uploaded_files:
            dst = UPLOAD_DIR / uf.name
            dst.write_bytes(uf.getvalue())
            file_paths.append(dst)

        rows = []
        with st.spinner("推論中..."):
            fast_available = use_fast and (_load_model_once() is not None)
            if fast_available:
                try:
                    labels = _predict_fast_batch(file_paths)
                    for p, lb in zip(file_paths, labels):
                        rows.append({"filename": p.name, "label": lb, "mode": "fast"})
                except Exception as e:
                    st.info(f"高速モード失敗: {e}")
                    fast_available = False
            if not fast_available:
                for p in file_paths:
                    if predict is not None:
                        try:
                            try:
                                out = predict(p.name)
                            except Exception:
                                out = predict(str(p))
                            rows.append({"filename": p.name, "label": out, "mode": "fallback"})
                        except Exception as e:
                            rows.append({"filename": p.name, "label": None, "error": str(e)})
                    else:
                        rows.append({"filename": p.name, "label": "dummy_label"})
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.subheader("プレビュー")
        cols = st.columns(3)
        for i, f in enumerate(uploaded_files):
            with cols[i % 3]:
                st.image(f, caption=f.name, use_container_width=True)

# ===============================
# Select Image ページ
# ===============================

def page_select_image():
    st.header("Select Image（ランダム選択 + 推論）")
    num = st.number_input("選択する画像枚数", min_value=1, max_value=50, value=10)
    with st.form("select_form"):
        do_predict = st.checkbox("選んだ画像も推論する", value=True)
        use_fast = st.checkbox("高速モード", value=True)
        run = st.form_submit_button("ランダム選択")

    if run:
        imgs = [p for p in UPLOAD_DIR.glob("**/*") if p.suffix.lower() in (".png",".jpg",".jpeg",".webp")]
        if randomselect is not None:
            try:
                selected_paths = [Path(p) for p in randomselect(str(UPLOAD_DIR), int(num))]
            except Exception:
                selected_paths = imgs[:int(num)]
        else:
            selected_paths = imgs[:int(num)]

        labels = None
        if do_predict:
            fast_available = use_fast and (_load_model_once() is not None)
            try:
                if fast_available:
                    labels = _predict_fast_batch(selected_paths)
                else:
                    labels = []
                    for p in selected_paths:
                        if predict is not None:
                            try:
                                try:
                                    out = predict(p.name)
                                except Exception:
                                    out = predict(str(p))
                                labels.append(str(out))
                            except Exception:
                                labels.append("?")
                        else:
                            labels.append("dummy_label")
            except Exception:
                labels = ["?"] * len(selected_paths)

        cols = st.columns(5)
        for i, p in enumerate(selected_paths):
            with cols[i % 5]:
                st.image(str(p), caption=p.name, use_container_width=True)
                if labels:
                    st.caption(f"label: {labels[i]}")

# ===============================
# ページ切替
# ===============================
PAGES = {"Check": page_check, "Select Image": page_select_image}
with st.sidebar:
    st.title("NyanCheck")
    choice = st.radio("ページを選択", list(PAGES.keys()))
    st.markdown(f"**Upload dir**: `{UPLOAD_DIR}`")
    st.caption("モデル: " + ("OK" if _load_model_once() else "未検出"))
PAGES[choice]()
