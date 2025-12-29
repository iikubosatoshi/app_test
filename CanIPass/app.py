import os
import re
import yaml
import unicodedata
import tempfile
from typing import Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st

from extractor import extract_courses_from_pdf
from judge import load_master, apply_master, judge_graduation, judge_thesis_start

# =========================================================
# Config
# =========================================================
st.set_page_config(page_title="卒業・卒論着手要件チェッカー", layout="wide")
st.title("卒業・卒論着手要件チェッカー")

DATA_DIR = "CanIPass/data"

# =========================================================
# CSV Encoding setting
# =========================================================
CSV_ENCODING = "utf-8-sig"   # ← ここを変えるだけで全体に反映

# =========================================================
# Helpers (common)
# =========================================================
_ROMAN = str.maketrans({"Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III", "Ⅳ": "IV"})


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace(" ", "").replace("　", "")
    s = s.translate(_ROMAN)
    return s.strip()


def list_data_files(exts: Tuple[str, ...]) -> List[str]:
    if not os.path.isdir(DATA_DIR):
        return []
    out = []
    for fn in os.listdir(DATA_DIR):
        if fn.lower().endswith(exts):
            out.append(fn)
    return sorted(out)


def load_yaml_from_data(filename: str) -> dict:
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_bytes_to_tempfile(uploaded_file, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded_file.read())
        return f.name


# =========================================================
# Batch CSV/Excel helpers
# =========================================================
def parse_star_number(x) -> float:
    """
    CSVにある '*44.5' のような値を 44.5 として扱う。
    空/NaN は 0。
    """
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        if pd.isna(x):
            return 0.0
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return 0.0
    s = s.lstrip("*")  # 先頭の不足印を外す
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def build_code_to_label_map(df: pd.DataFrame) -> Dict[str, str]:
    """
    ヘッダFLG==1 の行から、001.. の列名->日本語ラベル を作る。
    例: "001" -> "修得単位数計"
    """
    if "ヘッダFLG" not in df.columns:
        raise ValueError("このCSV/Excelには 'ヘッダFLG' 列がありません。")

    header_rows = df[df["ヘッダFLG"] == 1]
    if header_rows.empty:
        raise ValueError("ヘッダFLG==1 の行が見つかりません。")

    hdr = header_rows.iloc[0].to_dict()

    code_cols = [c for c in df.columns if re.fullmatch(r"\d{3}", str(c))]
    mapping = {}
    for c in code_cols:
        label = hdr.get(c)
        if label is None or str(label).strip() == "":
            continue
        mapping[str(c)] = str(label).strip()
    return mapping


def detect_course_key(row: pd.Series) -> Optional[str]:
    """
    行からコース(material / mech_elec)を推定。
    - 所属略称 / カリキュラム略称 に '物質材' や '機械電' が入っている想定
    """
    for col in ["所属略称", "カリキュラム略称"]:
        if col in row and pd.notna(row[col]):
            s = str(row[col])
            if "物質" in s or "物質材" in s:
                return "material"
            if "機械" in s or "機械電" in s:
                return "mech_elec"
    return None


def get_value_by_label(row: pd.Series, code_to_label: Dict[str, str], label: str) -> float:
    """
    ラベル名から該当する 001.. 列の値を拾って数値化
    """
    label_n = norm(label)
    for code, lab in code_to_label.items():
        if norm(lab) == label_n:
            return parse_star_number(row.get(code))
    return 0.0


def sum_by_labels(row: pd.Series, code_to_label: Dict[str, str], labels: List[str]) -> float:
    return sum(get_value_by_label(row, code_to_label, lb) for lb in labels)


def judge_graduation_from_row(
    row: pd.Series,
    code_to_label: Dict[str, str],
    req_course: dict,
) -> dict:
    """
    CSV/Excel1行（学生）から卒業要件を判定（集計ベース）
    requirements.yaml:
      graduation:
        min_credits: ...
        label_sets:
          required_labels: [...]
          elective_labels: [...]
        elective_required_groups: [...]
    """
    mins = req_course["graduation"]["min_credits"]
    label_sets = req_course["graduation"].get("label_sets", {})
    req_labels = label_sets.get("required_labels", [])
    ele_labels = label_sets.get("elective_labels", [])

    kikan_total = get_value_by_label(row, code_to_label, "基幹教育科目")
    senko_total = get_value_by_label(row, code_to_label, "専攻教育科目")
    overall_total = get_value_by_label(row, code_to_label, "修得単位数計")
    if overall_total == 0.0:
        overall_total = kikan_total + senko_total

    senko_required = sum_by_labels(row, code_to_label, req_labels)
    senko_elective = sum_by_labels(row, code_to_label, ele_labels)

    results = []

    def add(label, got, need):
        need = float(need)
        results.append(
            {
                "label": label,
                "got": float(got),
                "need": need,
                "ok": float(got) >= need,
                "lack": max(0.0, need - float(got)),
            }
        )

    add("基幹教育科目 合計", kikan_total, mins["kikan_total"])
    add("専攻教育科目 合計", senko_total, mins["senko_total"])
    add("専攻教育（必修）", senko_required, mins["senko_required"])
    add("専攻教育（選択）", senko_elective, mins["senko_elective"])
    add("総取得単位 合計", overall_total, mins["overall_total"])

    # 選択必修グループ（コースごと）
    egroups = req_course["graduation"].get("elective_required_groups", [])
    group_results = []
    for g in egroups:
        g_label = g.get("label", "選択必修")
        g_need = float(g.get("min_credits", 0))
        g_courses = g.get("courses", [])
        g_got = sum_by_labels(row, code_to_label, g_courses)
        group_results.append(
            {
                "label": g_label,
                "got": g_got,
                "need": g_need,
                "ok": g_got >= g_need,
                "lack": max(0.0, g_need - g_got),
                "details": g_courses,
            }
        )

    ok = all(r["ok"] for r in results) and all(gr["ok"] for gr in group_results)

    return {
        "ok": ok,
        "totals": results,
        "elective_required_groups": group_results,
        # 主要値も返す（CSV出力用）
        "metrics": {
            "kikan_total": kikan_total,
            "senko_total": senko_total,
            "senko_required": senko_required,
            "senko_elective": senko_elective,
            "overall_total": overall_total,
        },
    }


def judge_thesis_start_from_row(
    row: pd.Series,
    code_to_label: Dict[str, str],
    req_course: dict,
    master: dict,
    course_key: str,
) -> dict:
    """
    CSV/Excel1行（学生）から卒論着手要件を判定（集計＋必修未修得判定）
    - min_credits: 基幹/専攻（4年次開講除外）
    - constraints: 実験・演習必修の未修得、必修未修得単位<=4
    """
    mins = req_course["thesis_start"]["min_credits"]
    cons = req_course["thesis_start"]["constraints"]

    kikan_total = get_value_by_label(row, code_to_label, "基幹教育科目")
    senko_total = get_value_by_label(row, code_to_label, "専攻教育科目")

    # 4年次開講科目の合計（列が存在するものだけ）
    fourth_year_sum = 0.0
    for m in master.values():
        if m.get("area") != "専攻":
            continue
        if str(m.get("is_fourth_year", "")).lower() != "true":
            continue

        scope = m.get("scope")
        if scope == "物質材料" and course_key != "material":
            continue
        if scope == "機械電気" and course_key != "mech_elec":
            continue
        if scope not in ("共通", "物質材料", "機械電気"):
            continue

        fourth_year_sum += get_value_by_label(row, code_to_label, m.get("name", ""))

    senko_excl_4 = max(0.0, senko_total - fourth_year_sum)

    results = []

    def add(label, got, need):
        need = float(need)
        results.append(
            {
                "label": label,
                "got": float(got),
                "need": need,
                "ok": float(got) >= need,
                "lack": max(0.0, need - float(got)),
            }
        )

    add("基幹教育科目（3年後期終了まで）", kikan_total, mins["kikan_total"])
    add("専攻教育（4年次開講除外・3年後期終了まで）", senko_excl_4, mins["senko_excluding_4th_year"])

    # ② 実験・演習（必修）を全て修得（4年次開講除外）
    missing_labs = []
    if cons.get("required_labs_exercises_all_passed", False):
        for m in master.values():
            if m.get("area") != "専攻":
                continue
            if m.get("kind") != "必修":
                continue
            if str(m.get("is_fourth_year", "")).lower() == "true":
                continue
            if not m.get("is_lab_or_exercise", False):
                continue

            scope = m.get("scope")
            if scope == "物質材料" and course_key != "material":
                continue
            if scope == "機械電気" and course_key != "mech_elec":
                continue
            if scope not in ("共通", "物質材料", "機械電気"):
                continue

            name = m.get("name", "")
            need_credits = float(m.get("credits") or 0.0)
            got = get_value_by_label(row, code_to_label, name)
            if got + 1e-9 < need_credits:
                missing_labs.append(name)

    labs_ok = (len(missing_labs) == 0)

    # ③ 必修（4年次開講除く）の未修得単位数 <= 4
    missing_required_courses = []
    missing_required_credits = 0.0
    for m in master.values():
        if m.get("area") != "専攻":
            continue
        if m.get("kind") != "必修":
            continue
        if str(m.get("is_fourth_year", "")).lower() == "true":
            continue

        scope = m.get("scope")
        if scope == "物質材料" and course_key != "material":
            continue
        if scope == "機械電気" and course_key != "mech_elec":
            continue
        if scope not in ("共通", "物質材料", "機械電気"):
            continue

        name = m.get("name", "")
        need_credits = float(m.get("credits") or 0.0)
        got = get_value_by_label(row, code_to_label, name)
        if got + 1e-9 < need_credits:
            missing_required_courses.append(name)
            missing_required_credits += need_credits

    max_missing = float(cons.get("required_missing_credits_max", 0))
    missing_ok = (missing_required_credits <= max_missing)

    ok = all(r["ok"] for r in results) and labs_ok and missing_ok

    return {
        "ok": ok,
        "totals": results,
        "missing_labs": missing_labs,
        "missing_required_credits": missing_required_credits,
        "max_missing_required_credits": max_missing,
        "missing_required_courses": missing_required_courses,
        # 主要値も返す（CSV出力用）
        "metrics": {
            "kikan_total_upto_y3w": kikan_total,
            "senko_total": senko_total,
            "senko_excl_4": senko_excl_4,
            "fourth_year_sum": fourth_year_sum,
            "missing_labs_count": len(missing_labs),
            "missing_required_credits": missing_required_credits,
            "max_missing_required_credits": max_missing,
        },
    }

# =========================================================
# Display (重要列だけ表示＆ラベルを分かりやすく)
# =========================================================

DISPLAY_COLS_BASE = [
    "ID",
    "氏名",
    "学年",
    "コース",
    "判定種別",
    "判定",
    "不足サマリ",
]

# 卒業（4年）で特に見たいもの
DISPLAY_COLS_GRAD = [
    "kikan_total",
    "senko_total",
    "senko_required",
    "senko_elective",
    "overall_total",
    # 選択必修グループ（ある場合）
    "elective_group1_label",
    "elective_group1_got",
    "elective_group1_need",
    "elective_group1_ok",
]

# 卒論着手（3年）で特に見たいもの
DISPLAY_COLS_THESIS = [
    "kikan_total_upto_y3w",
    "senko_excl_4",
    "missing_required_credits",
    "max_missing_required_credits",
    "missing_labs_count",
    # 任意：どれが足りないか（長いので非表示でもOK）
    # "missing_labs_list",
    # "missing_required_courses_list",
]

RENAME_MAP = {
    "ID": "ID（学籍番号 or 仮ID）",
    "氏名": "氏名",
    "学年": "学年",
    "コース": "コース",
    "判定種別": "判定種別",
    "判定": "判定",
    "不足サマリ": "不足ポイント",

    # graduation metrics
    "kikan_total": "基幹 合計(取得)",
    "senko_total": "専攻 合計(取得)",
    "senko_required": "専攻 必修(取得)",
    "senko_elective": "専攻 選択(取得)",
    "overall_total": "総取得(取得)",

    # elective required group
    "elective_group1_label": "選択必修グループ",
    "elective_group1_got": "選択必修 取得",
    "elective_group1_need": "選択必修 必要",
    "elective_group1_ok": "選択必修 OK",

    # thesis metrics
    "kikan_total_upto_y3w": "基幹(3年後期まで取得)",
    "senko_excl_4": "専攻(4年次除外・取得)",
    "missing_required_credits": "必修 未修得単位(4年次除外)",
    "max_missing_required_credits": "必修 未修得許容(単位)",
    "missing_labs_count": "実験・演習 未修得数",

    # optional long fields
    "missing_labs_list": "実験・演習 未修得(一覧)",
    "missing_required_courses_list": "必修 未修得(一覧)",
}

# =========================================================
# UI: data file pickers
# =========================================================
st.sidebar.header("データファイル（data/）")
req_files = list_data_files((".yaml", ".yml"))
master_files = list_data_files((".csv",))

if not req_files:
    st.sidebar.error("data/ に requirements.yaml が見つかりません。")
if not master_files:
    st.sidebar.error("data/ に course_master.csv が見つかりません。")

req_sel = st.sidebar.selectbox("requirements.yaml", req_files, index=0 if req_files else None)
master_sel = st.sidebar.selectbox("course_master.csv", master_files, index=0 if master_files else None)

req = load_yaml_from_data(req_sel) if req_files else {}
master_path = os.path.join(DATA_DIR, master_sel) if master_files else None

term_order = req.get("term_order", {})
courses_req = req.get("courses", {})

# =========================================================
# Tabs
# =========================================================
tab_pdf, tab_batch = st.tabs(["PDF（1人）", "CSV/Excel（一括）"])

# =========================================================
# Tab: PDF (single)
# =========================================================
with tab_pdf:
    st.subheader("PDF（1人）判定")

    pdf = st.file_uploader("成績表PDF（テキストPDF）", type=["pdf"], key="pdf_uploader")

    course_key = st.selectbox(
        "コース",
        ["material", "mech_elec"],
        format_func=lambda x: {"material": "物質材料", "mech_elec": "機械電気"}[x],
        key="pdf_course_key",
    )
    admission_year = st.number_input(
        "入学年度（西暦）",
        min_value=2000,
        max_value=2100,
        value=2022,
        step=1,
        key="pdf_admission_year",
    )

    if pdf and master_path and courses_req:
        pdf_path = load_bytes_to_tempfile(pdf, ".pdf")

        req_course = courses_req[course_key]
        master = load_master(master_path)

        raw_courses = extract_courses_from_pdf(pdf_path)
        courses = apply_master(raw_courses, master)

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

        st.subheader("卒業要件 判定")
        grad = judge_graduation(courses, req_course, selected_course_key=course_key)
        st.success("卒業要件OK ✅" if grad["ok"] else "卒業要件NG ❌")
        for r in grad["totals"]:
            st.write(("✅ " if r["ok"] else "❌ ") + f"{r['label']}: {r['got']} / {r['need']}（不足 {r['lack']}）")

        eg = grad.get("elective_required_groups")
        if eg:
            st.markdown("### 選択必修（グループ）")
            for g in eg:
                st.write(("✅ " if g["ok"] else "❌ ") + f"{g['label']}: {g['got']} / {g['need']}（不足 {g['lack']}）")

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
        st.info("PDFをアップロードすると判定できます（requirements.yaml / course_master.csv は data/ のものを使用）。")


# =========================================================
# Tab: Batch (CSV/Excel)
# =========================================================
with tab_batch:
    st.subheader("CSV/Excel（一括）判定（全学生）")

    st.caption(
        "・ヘッダFLG==1 の行から 001.. の見出しを復元して判定します。"
        "・3年生は卒論着手、4年生は卒業要件を判定します。"
    )

    batch_file = st.file_uploader(
        "判定用CSV/Excel（3年/4年の成績リスト）",
        type=["csv", "xlsx", "xls"],
        key="batch_uploader",
    )

    default_encoding = st.selectbox("CSVエンコーディング", ["utf-8", "cp932", "utf-8-sig"], index=0, key="csv_encoding")

    if batch_file and master_path and courses_req:
        # Load
        if batch_file.name.lower().endswith(".csv"):
            df = pd.read_csv(batch_file, encoding=default_encoding)
        else:
            df = pd.read_excel(batch_file)

        # Build mapping
        code_to_label = build_code_to_label_map(df)

        # Student rows
        students = df[df["ヘッダFLG"] != 1].copy()
        students = students.reset_index(drop=True)
        students["student_id"] = students.index.map(lambda i: f"anon_{i+1:06d}")

        # master
        master = load_master(master_path)

        out_rows = []

        for _, row in students.iterrows():
            grade = row.get("学年")
            if pd.isna(grade):
                continue

            ck = detect_course_key(row)
            if ck is None:
                ck = "unknown"

            sid_raw = row.get("学籍番号", "")
            sid = "" if pd.isna(sid_raw) else str(sid_raw).strip()
            
            # 学籍番号が空なら仮IDを使う
            if sid == "":
                sid = str(row.get("student_id", "")).strip()
            
            name_raw = row.get("氏名", "")
            name = "" if pd.isna(name_raw) else str(name_raw).strip()
            curriculum = row.get("カリキュラム略称", "")
            dept = row.get("所属略称", "")

            req_course = courses_req.get(ck) if ck in courses_req else None

            # 判定種別：3年=卒論着手、4年=卒業
            totals = []
            egroups = []
            metrics = {}

            if int(grade) == 3:
                kind = "卒論着手"
                if req_course is None:
                    ok = False
                    detail = "コース判定不能"
                else:
                    res = judge_thesis_start_from_row(row, code_to_label, req_course, master, ck)
                    ok = res["ok"]
                    totals = res["totals"]
                    metrics = res.get("metrics", {})
                    detail = "OK" if ok else "NG"
                    # 追加の詳細（必要なら展開）
                    metrics["missing_labs_list"] = " / ".join(res.get("missing_labs", []))
                    metrics["missing_required_courses_list"] = " / ".join(res.get("missing_required_courses", []))

            elif int(grade) == 4:
                kind = "卒業"
                if req_course is None:
                    ok = False
                    detail = "コース判定不能"
                else:
                    res = judge_graduation_from_row(row, code_to_label, req_course)
                    ok = res["ok"]
                    totals = res["totals"]
                    egroups = res.get("elective_required_groups", [])
                    metrics = res.get("metrics", {})
                    # 選択必修グループの取得/必要もメトリクスに落とす
                    for i, g in enumerate(egroups, start=1):
                        metrics[f"elective_group{i}_label"] = g["label"]
                        metrics[f"elective_group{i}_got"] = g["got"]
                        metrics[f"elective_group{i}_need"] = g["need"]
                        metrics[f"elective_group{i}_ok"] = g["ok"]
                    detail = "OK" if ok else "NG"

            else:
                kind = "対象外"
                ok = False
                detail = "対象外学年"

            # totals（判定条件）を列に展開： got/need/ok/lack
            flat_totals = {}
            for t in totals:
                # 列名を安定化（スペースや記号を軽く整理）
                base = re.sub(r"\s+", "_", str(t["label"]).strip())
                base = base.replace("（", "_").replace("）", "").replace("・", "_").replace("/", "_")
                flat_totals[f"{base}__got"] = t["got"]
                flat_totals[f"{base}__need"] = t["need"]
                flat_totals[f"{base}__ok"] = t["ok"]
                flat_totals[f"{base}__lack"] = t["lack"]

            # 選択必修グループも列に展開（複数対応）
            flat_groups = {}
            for i, g in enumerate(egroups, start=1):
                base = f"選択必修グループ{i}"
                flat_groups[f"{base}__label"] = g["label"]
                flat_groups[f"{base}__got"] = g["got"]
                flat_groups[f"{base}__need"] = g["need"]
                flat_groups[f"{base}__ok"] = g["ok"]
                flat_groups[f"{base}__lack"] = g["lack"]

            # サマリ（見やすさ用）
            lack_parts = []
            for t in totals:
                if not t["ok"]:
                    lack_parts.append(f"{t['label']} 不足{t['lack']}")
            for g in egroups:
                if not g["ok"]:
                    lack_parts.append(f"{g['label']} 不足{g['lack']}")
            lack_summary = " / ".join(lack_parts) if lack_parts else ""

            base_row = {
                "ID": sid,  # 学籍番号 or anon_...
                "氏名": name,
                "学年": int(grade),
                "コース": {"material": "物質材料", "mech_elec": "機械電気"}.get(ck, "不明"),
                "判定種別": kind,
                "判定": "OK" if ok else "NG",
                "不足サマリ": lack_summary,
                "所属略称": dept,
                "カリキュラム略称": curriculum,
            }
            
            # metrics を列化（主要集計値＋未修得リスト等）
            # ※ 列数が増えるので、必要ならこのブロックをコメントアウトしてもOK
            metrics_row = {}
            for k, v in (metrics or {}).items():
                metrics_row[str(k)] = v

            out_rows.append({**base_row, **metrics_row, **flat_totals, **flat_groups})

        result_df = pd.DataFrame(out_rows)

        st.subheader("判定結果（重要項目のみ）")

        # 重要列だけに絞る（存在する列だけ）
        want_cols = DISPLAY_COLS_BASE + DISPLAY_COLS_GRAD + DISPLAY_COLS_THESIS
        use_cols = [c for c in want_cols if c in result_df.columns]
        display_df = result_df[use_cols].copy()
        
        # ラベル（列名）を分かりやすく
        display_df = display_df.rename(columns=RENAME_MAP)
        
        # 並び順：NGを上に
        if "判定" in display_df.columns:
            display_df = display_df.sort_values(by=["判定"], ascending=True)
            
        # =========================================================
        # NG（未達）だけ表示するかどうか
        # =========================================================
        only_ng = st.checkbox("❌ 未達（NG）の学生のみ表示", value=False)
        
        if only_ng and "判定" in display_df.columns:
            display_df_view = display_df[display_df["判定"] == "NG"]
        else:
            display_df_view = display_df
        
        st.dataframe(display_df_view, use_container_width=True)

        # ダウンロード
        csv_bytes = result_df.to_csv(index=False).encode(CSV_ENCODING)
        st.download_button(
            "結果CSVをダウンロード（UTF-8 BOM）",
            data=csv_bytes,
            file_name="judgement_results_with_details.csv",
            mime="text/csv",
        )
        
        # 詳細（全部）ダウンロード
        csv_full = result_df.to_csv(index=False).encode(CSV_ENCODING)
        st.download_button(
            "結果CSVをダウンロード（詳細：全列）",
            data=csv_full,
            file_name="judgement_results_full.csv",
            mime="text/csv",
        )
        
        # サマリ（重要列のみ）ダウンロード
        csv_summary = display_df.to_csv(index=False).encode(CSV_ENCODING)
        st.download_button(
            "結果CSVをダウンロード（サマリ：重要列のみ）",
            data=csv_summary,
            file_name="judgement_results_summary.csv",
            mime="text/csv",
        )

        st.markdown("### フィルタ")
        col1, col2, col3 = st.columns(3)
        with col1:
            f_grade = st.multiselect("学年", sorted(result_df["学年"].unique().tolist()))
        with col2:
            f_course = st.multiselect("コース", sorted(result_df["コース"].unique().tolist()))
        with col3:
            f_ok = st.multiselect("判定", ["OK", "NG"], default=["NG"])

        filtered = result_df.copy()
        if f_grade:
            filtered = filtered[filtered["学年"].isin(f_grade)]
        if f_course:
            filtered = filtered[filtered["コース"].isin(f_course)]
        if f_ok:
            filtered = filtered[filtered["判定"].isin(f_ok)]

        st.dataframe(filtered, use_container_width=True)

        st.caption(
            "注意：この一括判定は、学務系の一覧（001..列）を“合算ベース”で評価します。"
            "PDF版のような個別履修履歴の照合よりは粗くなるため、運用上はNGのみ個別確認（PDF）を推奨します。"
        )

    else:
        st.info("CSV/Excelをアップロードすると全学生の判定結果を一覧で表示します（requirements.yaml / course_master.csv は data/ のものを使用）。")
