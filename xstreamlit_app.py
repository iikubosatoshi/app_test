# 先頭付近の import よりも前が理想（INFO抑制）
import os as _os
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

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

# これをファイル内のどこか（_load_model_once の直前が分かりやすい）に追加
def _record_model_error(e: Exception):
    import traceback
    # 例外情報をセッションに保存（サイドバーで見られる）
    st.session_state["model_load_error"] = repr(e)
    st.session_state["model_load_trace"] = traceback.format_exc()

def _is_hdf5_file(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"\x89HDF"
    except Exception:
        return False

# よくあるカスタム名の救済（必要に応じて追加）
def _common_custom_objects():
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import LeakyReLU
        return {
            "swish": tf.nn.swish,
            "relu6": tf.nn.relu6,
            "LeakyReLU": LeakyReLU,
        }
    except Exception:
        return {}

# -------------------------------------------------
# 高速化モデル読込
# -------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_model_once():
    if not MODEL_PATH.exists():
        return None

    # 1) tf.keras のローダ（古いH5互換に強い）
    try:
        from tensorflow.keras.models import load_model as tf_load_model
        return tf_load_model(str(MODEL_PATH), compile=False)
    except Exception as e1:
        _record_model_error(e1)

    # 2) tf.keras + よくある custom_objects（活性化・LeakyReLU など）
    try:
        from tensorflow.keras.models import load_model as tf_load_model
        return tf_load_model(str(MODEL_PATH), compile=False, custom_objects=_common_custom_objects())
    except Exception as e2:
        _record_model_error(e2)

    # 3) Keras 3 推奨API（安全モードOFFで Lambda/カスタム許可）
    try:
        import keras
        return keras.saving.load_model(str(MODEL_PATH), compile=False, safe_mode=False)
    except Exception as e3:
        _record_model_error(e3)

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
        st.dataframe(df, width="stretch")
        st.subheader("プレビュー")
        cols = st.columns(3)
        for i, f in enumerate(uploaded_files):
            with cols[i % 3]:
                st.image(f, caption=f.name, width="stretch")

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
                st.image(str(p), caption=p.name, width="stretch")
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

    # 現在のモデル状態を表示
    _m = None
    try:
        _m = _load_model_once()
    except Exception:
        _m = None
    st.caption("モデル: " + ("OK" if _m else "未検出"))

    # 一時診断（トラブル時にだけ開けばOK）
    st.divider()
    with st.expander("Diagnostics (一時表示)"):
        st.write("**MODEL_PATH**:", str(MODEL_PATH))
        st.write("exists:", MODEL_PATH.exists())
        st.write("parent exists:", MODEL_PATH.parent.exists())

        try:
            st.write("loadable (cache):", _m is not None)
        except Exception as e:
            st.write("loadable: False")
            st.exception(e)

        # predict.py の有無
        st.write("predict.py available:", predict is not None)

        # TensorFlow の存在確認（任意）
        try:
            import tensorflow as tf  # 重いので診断の中で遅延 import
            st.write("tensorflow:", tf.__version__)
        except Exception as e:
            st.write("tensorflow import error:", e)

        # results/ の中身を確認（空ならファイル未配置の可能性）
        try:
            from pathlib import Path as _Path
            files = [str(p) for p in _Path(MODEL_PATH.parent).glob("*")]
            st.write("results/ files:", files if files else "(empty)")
        except Exception as e:
            st.write("list error:", e)

    st.divider()
    with st.expander("Diagnostics (model loader)", expanded=False):
        st.write("MODEL_PATH:", str(MODEL_PATH))
        st.write("exists:", MODEL_PATH.exists(), "| is_dir:", MODEL_PATH.is_dir())
        st.write("seems_hdf5:", _is_hdf5_file(MODEL_PATH))
        st.write("loadable (cache):", _load_model_once() is not None)
        if "model_load_error" in st.session_state:
            st.write("last error:", st.session_state["model_load_error"])
        if "model_load_trace" in st.session_state:
            st.code(st.session_state["model_load_trace"])
            
    with st.expander("Diagnostics (model loader)", expanded=False):
        st.write("MODEL_PATH:", str(MODEL_PATH))
        st.write("exists:", MODEL_PATH.exists(), "| is_dir:", MODEL_PATH.is_dir())
        st.write("seems_hdf5:", _is_hdf5_file(MODEL_PATH))
        st.write("loadable (cache):", _load_model_once() is not None)
        if "model_load_error" in st.session_state:
            st.write("last error:", st.session_state["model_load_error"])
        if "model_load_trace" in st.session_state:
            st.code(st.session_state["model_load_trace"])

PAGES[choice]()
