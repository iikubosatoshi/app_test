import streamlit as st
import os
import io
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import random

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
VALIDATION_DIR = Path(os.environ.get("VALIDATION_DIR", "./validation_data")).resolve()
MODEL_PATH = Path("results/nyancheck.h5").resolve()
LABELS_PATH = Path("results/labels.txt").resolve()
TARGET_SIZE = (200, 150)
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

def allowed_file(name: str) -> bool:
    return Path(name).suffix.lower() in ALLOWED_EXTENSIONS

# -------------------------------------------------
# Flask版 check.py のロジックを支える補助（ラベル・クイズ生成）
# -------------------------------------------------
DEFAULT_CLASSES = [
    "アビシニアン", "犬", "エジプシャンマウ", "メインクーン",
    "ノルウェージャンフォレストキャット", "ロシアンブルー",
    "アメリカンショートヘアー", "日本猫",
]

def _answer_choices() -> list[str]:
    labels = _load_labels()
    return labels if labels else DEFAULT_CLASSES

def _infer_label_from_filename(filename: str, choices: list[str]) -> str | None:
    # ファイル名に含まれていれば優先
    for c in choices:
        if c and c in filename:
            return c
    # ディレクトリ名にも含まれていないかチェック
    parts = Path(filename).parts
    for c in choices:
        if any(c in p for p in parts):
            return c
    return None

def _make_quiz_df(k: int = 8) -> pd.DataFrame:
    """validation_data/<class>/* から優先的に抽出。無ければ /uploads から抽出。
    返り値: filename, path, 猫の種類
    """
    # 1) validation_data を最優先（ディレクトリ名=正解ラベル）
    items: list[tuple[Path, str]] = []
    try:
        if VALIDATION_DIR.exists():
            for cls_dir in sorted([d for d in VALIDATION_DIR.iterdir() if d.is_dir()]):
                cls = cls_dir.name
                for p in cls_dir.rglob("*"):
                    if p.is_file() and allowed_file(p.name):
                        items.append((p, cls))
    except Exception:
        items = []
    if items:
        random.shuffle(items)
        chosen = items[: int(k)]
        return pd.DataFrame([
            {"filename": p.name, "path": str(p), "猫の種類": cls} for p, cls in chosen
        ])

    # 2) フォールバック: selectimage.randomselect_df → /uploads 直下
    df = None
    try:
        from selectimage import randomselect_df  # type: ignore
        df = randomselect_df(str(UPLOAD_DIR), k=int(k))
    except Exception:
        df = None
    if df is None or df.empty:
        imgs = [p for p in UPLOAD_DIR.glob("**/*") if allowed_file(p.name)]
        random.shuffle(imgs)
        imgs = imgs[: int(k)]
        df = pd.DataFrame([{"filename": p.name, "path": str(p)} for p in imgs])
    if "猫の種類" not in df.columns:
        choices = _answer_choices()
        df["猫の種類"] = [
            _infer_label_from_filename(str(fn), choices) for fn in df["filename"]
        ]
    return df

# -------------------------------------------------
# 高速化モデル読込
# -------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_model_once():
    try:
        if not MODEL_PATH.exists():
            return None
        from tensorflow.keras.models import load_model  # type: ignore
        return load_model(str(MODEL_PATH), compile=False)
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

    tab_multi, tab_single, tab_quiz = st.tabs([
        "複数アップロード", "単発アップロード（Flask互換）", "対戦（クイズ）"
    ])

    # --- 複数アップロード（既存まとめ推論） ---
    with tab_multi:
        with st.form(key="check_form_multi"):
            uploaded_files = st.file_uploader(
                "画像をアップロード（複数可）",
                type=["png", "jpg", "jpeg", "webp", "gif"],
                accept_multiple_files=True,
            )
            save_to_uploads = st.checkbox("/uploads に保存する", value=True)
            use_fast = st.checkbox("高速モード", value=True)
            submitted = st.form_submit_button("推論を実行")

        if submitted and uploaded_files:
            file_paths = []
            for uf in uploaded_files:
                dst = UPLOAD_DIR / uf.name
                # 保存する/しないにかかわらず、一時配置（predictの引数で使う）
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

    # --- 単発アップロード（Flask: /api/v1/send 相当） ---
    with tab_single:
        c1, c2 = st.columns([3, 2])
        with c1:
            uf_single = st.file_uploader(
                "画像を1枚アップロード",
                type=["png", "jpg", "jpeg", "webp", "gif"],
                accept_multiple_files=False,
            )
            use_fast_single = st.checkbox("高速モードで推論", value=True, key="use_fast_single")
            run_single = st.button("アップロードして推論")
        with c2:
            st.caption("既存ファイルを使う（/uploads 内）")
            existing = sorted([p for p in UPLOAD_DIR.glob("**/*") if allowed_file(p.name)])
            pick = st.selectbox("ファイルを選択", ["（選択しない）"] + [p.name for p in existing])
            run_pick = st.button("選択したファイルで推論")

        target_path = None
        target_name = None
        if run_single and uf_single is not None:
            if not allowed_file(uf_single.name):
                st.error("許可されていない拡張子です")
            else:
                dst = UPLOAD_DIR / uf_single.name
                dst.write_bytes(uf_single.getvalue())
                target_path = dst
                target_name = uf_single.name
        elif run_pick and pick != "（選択しない）":
            target_path = UPLOAD_DIR / pick
            target_name = pick

        if target_path is not None:
            st.image(str(target_path), caption=target_name, width="stretch")
            row = {"filename": target_name}
            with st.spinner("推論中..."):
                fast_available = use_fast_single and (_load_model_once() is not None)
                try:
                    if fast_available:
                        label = _predict_fast_batch([target_path])[0]
                        row.update({"label": label, "mode": "fast"})
                    else:
                        if predict is not None:
                            try:
                                try:
                                    out = predict(target_name)
                                except Exception:
                                    out = predict(str(target_path))
                                row.update({"label": str(out), "mode": "fallback"})
                            except Exception as e:
                                row.update({"label": None, "error": str(e)})
                        else:
                            row.update({"label": "dummy_label"})
                except Exception as e:
                    row.update({"label": None, "error": str(e)})
            st.write(pd.DataFrame([row]))

    # --- 対戦（クイズ）: Flask index/check 相当 ---
    with tab_quiz:
        st.caption(f"validation_data（{VALIDATION_DIR}）配下の各ディレクトリ（=猫の種類）から画像をランダムに出題します。")
        k = st.number_input("出題数", min_value=1, max_value=30, value=8, step=1)
        colq1, colq2 = st.columns([1, 1])
        with colq1:
            if st.button("問題を作る"):
                st.session_state["quiz_df"] = _make_quiz_df(int(k))
                # 回答をリセット
                dfq = st.session_state["quiz_df"]
                for i in range(len(dfq)):
                    st.session_state.pop(f"ans_{i}", None)
        with colq2:
            use_fast_quiz = st.checkbox("高速モードで採点", value=True)

        dfq = st.session_state.get("quiz_df")
        if isinstance(dfq, pd.DataFrame) and not dfq.empty:
            choices = _answer_choices()
            ans_list = ["（選択してください）"] + choices
            cols = st.columns(4)
            for i, row in dfq.reset_index(drop=True).iterrows():
                img_path = row.get("path") or str(UPLOAD_DIR / row["filename"])  # type: ignore
                with cols[i % 4]:
                    st.image(img_path, caption=row["filename"], width="stretch")
                    st.selectbox("あなたの答え", ans_list, key=f"ans_{i}")

            if st.button("採点する"):
                # 予測
                paths = [Path(r.get("path") or (UPLOAD_DIR / r["filename"])) for _, r in dfq.iterrows()]
                # AI 推論
                ai_labels: list[str]
                try:
                    if use_fast_quiz and (_load_model_once() is not None):
                        ai_labels = _predict_fast_batch(paths)
                    else:
                        ai_labels = []
                        for p in paths:
                            if predict is not None:
                                try:
                                    try:
                                        out = predict(p.name)
                                    except Exception:
                                        out = predict(str(p))
                                    ai_labels.append(str(out))
                                except Exception:
                                    ai_labels.append("?")
                            else:
                                ai_labels.append("dummy_label")
                except Exception:
                    ai_labels = ["?"] * len(paths)

                # 正解（可能なら）
                gt = []
                for _, r in dfq.iterrows():
                    g = r.get("猫の種類")
                    if not g or pd.isna(g):
                        g = _infer_label_from_filename(r["filename"], choices)
                    gt.append(g)

                your = [st.session_state.get(f"ans_{i}", "（選択してください）") for i in range(len(dfq))]

                # 集計（正解が不明な行は除外）
                eval_rows = [i for i, g in enumerate(gt) if g and g != "（選択してください）"]
                correct_human = sum(1 for i in eval_rows if your[i] == gt[i])
                correct_ai = sum(1 for i in eval_rows if ai_labels[i] == gt[i])

                if correct_human > correct_ai:
                    youwin = "あなたの勝ち"
                elif correct_human == correct_ai:
                    youwin = "引き分け"
                else:
                    youwin = "あなたの負け"

                # 結果テーブル
                disp = pd.DataFrame({
                    "filename": [r["filename"] for _, r in dfq.iterrows()],
                    "正解は": gt,
                    "あなたの答え": your,
                    "nyancheckの答え": ai_labels,
                })
                st.subheader("採点結果")
                st.dataframe(disp, width="stretch")

                m1, m2, m3 = st.columns(3)
                m1.metric("あなたの正解数", correct_human)
                m2.metric("AIの正解数", correct_ai)
                m3.metric("判定", youwin)

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
        imgs = [p for p in UPLOAD_DIR.glob("**/*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".gif")]
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
    st.markdown(f"**Validation dir**: `{VALIDATION_DIR}`")
    st.caption("validation_data: " + ("検出" if VALIDATION_DIR.exists() else "未検出"))
    st.caption("モデル: " + ("OK" if _load_model_once() else "未検出"))
PAGES[choice]()
