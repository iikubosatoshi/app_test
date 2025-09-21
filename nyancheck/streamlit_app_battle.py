import streamlit as st
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import random

st.set_page_config(page_title="NyanCheck", page_icon="🐾", layout="wide")

# -------------------------------------------------
# 環境設定
# -------------------------------------------------
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "./nyancheck/uploads")).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_DIR = Path(os.environ.get("VALIDATION_DIR", "./nyancheck/validation_data")).resolve()
MODEL_PATH = Path("nyancheck/results/nyancheck.h5").resolve()
LABELS_PATH = Path("nyancheck/results/labels.txt").resolve()
TARGET_SIZE = (200, 150)
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

# 表示言語: ja / en（環境変数で切替）
LABEL_LANG = os.environ.get("LABEL_LANG", "ja").lower()
# クイズの出題数（UIなし・固定）。必要に応じて環境変数で変更
K_DEFAULT = int(os.environ.get("QUIZ_NUM", "8"))

def allowed_file(name: str) -> bool:
    return Path(name).suffix.lower() in ALLOWED_EXTENSIONS

# -------------------------------------------------
# 予測（フォールバック用）
# -------------------------------------------------
try:
    from predict import predict  # type: ignore
except Exception:
    predict = None

# -------------------------------------------------
# ラベル読み込み（日本語/英語 両対応）
# -------------------------------------------------
DEFAULT_CLASSES = [
    "アビシニアン", "犬", "エジプシャンマウ", "メインクーン",
    "ノルウェージャンフォレストキャット", "ロシアンブルー",
    "アメリカンショートヘアー", "日本猫",
]

@st.cache_data(show_spinner=False)
def _load_labels_legacy():
    if LABELS_PATH.exists():
        txt = LABELS_PATH.read_text(encoding="utf-8")
        labs = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        return labs if labs else None
    return None

@st.cache_data(show_spinner=False)
def _load_labels_bilingual():
    """labels.txt を「日本語 | 英語」形式（区切り: | / タブ / カンマ）で読む。
    戻り値: [{"ja": str, "en": str}, ...] または None
    """
    if not LABELS_PATH.exists():
        return None
    lines = [ln.strip() for ln in LABELS_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out: list[dict[str, str]] = []
    for ln in lines:
        parts = re.split(r'[|	,]', ln, maxsplit=1)
        if len(parts) >= 2:
            ja, en = parts[0].strip(), parts[1].strip()
        else:
            ja = parts[0].strip()
            en = ja
        out.append({"ja": ja, "en": en})
    return out if out else None

def _labels_for_display() -> list[str] | None:
    bi = _load_labels_bilingual()
    if bi:
        return [d.get(LABEL_LANG, d["ja"]) for d in bi]
    return _load_labels_legacy()

def _name_to_id() -> dict[str, int]:
    bi = _load_labels_bilingual()
    mapping: dict[str, int] = {}
    if bi:
        for i, d in enumerate(bi):
            for v in {d["ja"], d["en"]}:
                mapping[v] = i
    else:
        legacy = _load_labels_legacy() or []
        for i, v in enumerate(legacy):
            mapping[v] = i
    return mapping

# UI 表示用の選択肢
def _answer_choices() -> list[str]:
    labels = _labels_for_display()
    if labels:
        return labels
    return DEFAULT_CLASSES

# ファイル名/パスからクラス名を推測（fallback 用）
def _infer_label_from_filename(filename: str, choices: list[str]) -> str | None:
    for c in choices:
        if c and c in filename:
            return c
    parts = Path(filename).parts
    for c in choices:
        if any(c in p for p in parts):
            return c
    return None

# -------------------------------------------------
# モデル読み込み & 高速推論
# -------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_model_once():
    try:
        if not MODEL_PATH.exists():
            # パス自体が無い
            try:
                st.session_state["_model_error"] = f"NotFound: {MODEL_PATH}"
            except Exception:
                pass
            return None
        from tensorflow.keras.models import load_model  # type: ignore
        model = load_model(str(MODEL_PATH), compile=False)
        # 正常にロードできたらエラー情報を消す
        try:
            st.session_state.pop("_model_error", None)
        except Exception:
            pass
        return model
    except Exception as e:
        # 失敗理由を保存しておく（UIで警告表示に使う）
        try:
            st.session_state["_model_error"] = f"{type(e).__name__}: {e}"
        except Exception:
            pass
        return None
        from tensorflow.keras.models import load_model  # type: ignore
        return load_model(str(MODEL_PATH), compile=False)
    except Exception:
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
            arrays.append(_preprocess_pil(img))
            valid_idx.append(i)
        except Exception:
            arrays.append(None)
    batch = np.stack([a for a in arrays if a is not None], axis=0)
    preds = model.predict(batch, verbose=0)
    idxs = preds.argmax(axis=1) if preds.ndim == 2 else np.zeros((preds.shape[0],), dtype=np.int64)
    label_table = _labels_for_display()
    mapped = [label_table[i] if label_table and 0 <= i < len(label_table) else str(i) for i in idxs]
    out = ["?"] * len(paths)
    j = 0
    for i in range(len(paths)):
        if i in valid_idx:
            out[i] = mapped[j]
            j += 1
    return out

# -------------------------------------------------
# 出題データ作成（validation_data/<class>/... を優先）
# -------------------------------------------------
def _make_quiz_df(k: int = K_DEFAULT) -> pd.DataFrame:
    """validation_data/<class>/* から優先的に抽出。返り値: filename, path, 猫の種類"""
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
        return pd.DataFrame(
            [{"filename": p.name, "path": str(p), "猫の種類": cls} for p, cls in chosen]
        )

    # フォールバック: uploads から抽出（正解はファイル名/ディレクトリから推測）
    imgs = [p for p in UPLOAD_DIR.glob("**/*") if p.is_file() and allowed_file(p.name)]
    random.shuffle(imgs)
    imgs = imgs[: int(k)]
    df = pd.DataFrame([{"filename": p.name, "path": str(p)} for p in imgs])
    if "猫の種類" not in df.columns:
        choices = _answer_choices()
        df["猫の種類"] = [_infer_label_from_filename(str(fn), choices) for fn in df["filename"]]
    return df

# -------------------------------------------------
# 画面（クイズのみ / ボタン最小）
# -------------------------------------------------

def page_check():
    st.header("AIと対戦。猫の種類を当ててください。")
    st.caption(f"validation_data（{VALIDATION_DIR}）配下の各ディレクトリ（=猫の種類）から画像をランダムに出題します。")

    # モデルのロード状況を警告表示
    _model_obj = _load_model_once()
    if _model_obj is None:
        err = st.session_state.get("_model_error")
        if MODEL_PATH.exists():
            st.warning(f"モデルの読み込みに失敗しました（{MODEL_PATH.name}）。predict.py にフォールバックします。詳細: {err}")
        else:
            st.warning(f"モデルが見つかりません（{MODEL_PATH}）。predict.py にフォールバックします。")
    else:
        st.success(f"モデルの読み込みに成功しました（{MODEL_PATH.name}）")

    # 初回のみ自動で問題を生成（ページ再実行でも保持）
    if "quiz_df" not in st.session_state:
        st.session_state["quiz_df"] = _make_quiz_df()
        dfq_init = st.session_state["quiz_df"]
        for i in range(len(dfq_init)):
            st.session_state.pop(f"ans_{i}", None)

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

            # AI 推論（モデルがあればバッチ／無ければ predict.py）
            try:
                if _load_model_once() is not None:
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

            # 日本語/英語をクラスIDへ正規化して比較
            name2id = _name_to_id()
            def to_id(x: str | None):
                if not x or x == "（選択してください）":
                    return None
                return name2id.get(str(x))

            gt_ids   = [to_id(g) for g in gt]
            your_ids = [to_id(y) for y in your]
            ai_ids   = [to_id(a) for a in ai_labels]

            eval_rows = [i for i, gid in enumerate(gt_ids) if gid is not None]
            correct_human = sum(1 for i in eval_rows if your_ids[i] is not None and your_ids[i] == gt_ids[i])
            correct_ai    = sum(1 for i in eval_rows if ai_ids[i]   is not None and ai_ids[i]   == gt_ids[i])

            if correct_human > correct_ai:
                youwin = "あなたの勝ち"
            elif correct_human == correct_ai:
                youwin = "引き分け"
            else:
                youwin = "あなたの負け"

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

# 単一ページ運用
page_check()
