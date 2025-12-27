import streamlit as st
import yaml
import tempfile
from pathlib import Path

from extractor import extract_courses_from_pdf
from judge import load_master, apply_master, judge_graduation, judge_thesis_start

st.set_page_config(page_title="卒業・卒論着手要件チェッカー", layout="wide")
st.title("卒業・卒論着手要件チェッカー（成績表PDF）")

# -----------------------------
# data/ 配下から選択して読む
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

if not DATA_DIR.exists():
    st.error(f"data ディレクトリが見つかりません: {DATA_DIR}")
    st.stop()

master_candidates = sorted(DATA_DIR.glob("*.csv"))
req_candidates = sorted([*DATA_DIR.glob("*.yaml"), *DATA_DIR.glob("*.yml")])

if not master_candidates:
    st.error(f"data/ に CSV がありません（course_master.csv を置いてください）: {DATA_DIR}")
    st.stop()

if not req_candidates:
    st.error(f"data/ に requirements の yaml/yml がありません: {DATA_DIR}")
    st.stop()

def _label(p: Path) -> str:
    return p.name

selected_master_path = st.selectbox(
    "科目マスタ（CSV）",
    master_candidates,
    format_func=_label,
    index=0,
)

selected_req_path = st.selectbox(
    "要件（YAML）",
    req_candidates,
    format_func=_label,
    index=0,
)

with st.expander("読み込みファイル（data/）", expanded=False):
    st.code(f"master: {selected_master_path}\nrequirements: {selected_req_path}")

pdf = st.file_uploader("成績表PDF（テキストPDF）", type=["pdf"])

course_key = st.selectbox(
    "コース",
    ["material", "mech_elec"],
    format_func=lambda x: {"material": "物質材料", "mech_elec": "機械電気"}[x],
)

admission_year = st.number_input(
    "入学年度（西暦）", min_value=2000, max_value=2100, value=2022, step=1
)

# -----------------------------
# 実行
# -----------------------------
if pdf:
    # PDFだけアップロード。requirements/master は data/ の固定ファイルを使う。
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf.read())
        pdf_path = f.name

    # requirements.yaml 読み込み
    try:
        with open(selected_req_path, "r", encoding="utf-8") as f:
            req = yaml.safe_load(f)
    except Exception as e:
        st.error(f"requirements の読み込みに失敗しました: {selected_req_path}\n{e}")
        st.stop()

    # 必須キーの存在チェック（落ち方を分かりやすく）
    if "term_order" not in req or "courses" not in req:
        st.error("requirements.yaml の形式が不正です（term_order / courses がありません）")
        st.stop()

    if course_key not in req["courses"]:
        st.error(f"requirements.yaml に courses.{course_key} がありません")
        st.stop()

    term_order = req["term_order"]
    req_course = req["courses"][course_key]

    # master 読み込み
    try:
        master = load_master(str(selected_master_path))
    except Exception as e:
        st.error(f"course_master の読み込みに失敗しました: {selected_master_path}\n{e}")
        st.stop()

    # 抽出 → マスタ付与
    raw_courses = extract_courses_from_pdf(pdf_path)
    courses = apply_master(raw_courses, master)

    # 抽出結果
    st.subheader("抽出結果（確認）")
    st.caption("※ area/kind が None の行はマスタ未登録です（集計には含めません）")
    st.dataframe(courses, use_container_width=True)

    unknown = [c for c in courses if c.get("area") is None or c.get("kind") is None]
    if unknown:
        with st.expander("未分類科目（マスタ未登録）を表示", expanded=True):
            st.write(f"未分類：{len(unknown)}件")
            st.dataframe(
                [
                    {
                        "name": c.get("name"),
                        "credits": c.get("credits"),
                        "year": c.get("year"),
                        "term": c.get("term"),
                        "raw": c.get("raw"),
                    }
                    for c in unknown
                ],
                use_container_width=True,
            )

    # 卒業要件
    st.subheader("卒業要件 判定（選択必修なし）")
    grad = judge_graduation(courses, req_course, selected_course_key=course_key)
    st.success("卒業要件OK ✅" if grad["ok"] else "卒業要件NG ❌")
    for r in grad["totals"]:
        st.write(("✅ " if r["ok"] else "❌ ") + f"{r['label']}: {r['got']} / {r['need']}（不足 {r['lack']}）")
        
        # ← この4行を追加
        if "taken" in r:
            if r["taken"]:
                st.caption("取得科目: " + ", ".join(r["taken"]))
            else:
                st.caption("取得科目: なし")

    # 卒論着手要件
    st.subheader("卒論着手要件（3年後期終了時点） 判定")
    th = judge_thesis_start(
        courses=courses,
        master=master,
        req_course=req_course,
        selected_course_key=course_key,
        term_order=term_order,
        admission_year=admission_year,
    )
    st.success("卒論着手OK ✅" if th["ok"] else "卒論着手NG ❌")
    for r in th["totals"]:
        st.write(("✅ " if r["ok"] else "❌ ") + f"{r['label']}: {r['got']} / {r['need']}（不足 {r['lack']}）")

    st.markdown("### 必修（実験・演習）の未修得")
    if th["missing_labs"]:
        for x in th["missing_labs"]:
            st.write(f"- {x}")
    else:
        st.write("なし")

    st.markdown("### 必修（4年次開講除く）の未修得単位")
    st.write(f"{th['missing_required_credits']} 単位（許容 {th['max_missing_required_credits']} 単位）")
    if th["missing_required_courses"]:
        st.markdown("未修得科目：")
        for x in th["missing_required_courses"]:
            st.write(f"- {x}")
else:
    st.info("成績表PDF をアップロードすると判定できます。")