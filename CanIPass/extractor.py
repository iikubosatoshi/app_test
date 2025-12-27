# extractor.py（修正版：見出し誤削除を修正 + 省略名補完 + 実験I〜IV推定）
import re
import unicodedata
from collections import deque
import pdfplumber

TERM_PAT = r"(春学期|夏学期|秋学期|冬学期|前期集中|後期集中|前|後)"
END_OF_RECORD = re.compile(rf"(20\d{{2}})\s+{TERM_PAT}")

RECORD_PAT = re.compile(
    rf"^(?P<name>.*?)\s*"
    rf"(?P<credits>\d+(?:\.\d+)?)\s+"
    rf"(?P<grade>[A-ZＡ-ＺＲＦ])\s+"
    rf"(?P<gp>\*|\d+(?:\.\d+)?)\s+"
    rf"(?P<year>20\d{{2}})\s+"
    rf"(?P<term>{TERM_PAT})"
)

# PDFのヘッダ/説明文
NOISE_KEYWORDS = [
    "ログインユーザ", "成績照会", "科目ごとの成績", "第1外国語", "第２外国語", "第2外国語",
    "分野系列名", "科目名", "単位", "評価", "GP", "年度", "期間",
    "科目ナンバ", "講義コード", "成績担当者", "最終更新日",
]

def is_noise_line(s: str) -> bool:
    if not s:
        return True
    if any(k in s for k in NOISE_KEYWORDS):
        return True
    if len(s) > 80:
        return True
    return False

# 区分の見出し（※ここに「基幹教育セミナー」「課題協学科目」は入れない：科目名そのものだから）
CATEGORY_HEAD_PAT = re.compile(
    r"(言語文化基礎科目|文系ディシプリン科目|理系ディシプリン科目|"
    r"サイバーセキュリティ科目|健康・スポーツ科目|フロンティア科目|高年次基幹教育科目|"
    r"\(工\)専攻教育科目|専攻教育科目)"
)

# 末尾に付く科目コード（例：KED-HSS13）
CODE_TAIL_PAT = re.compile(r"(?:KED|ENG)-[A-Z0-9]+")

# 先頭に混入する短いコード（23J/24W/A09J/L294J など）
HEAD_CODE_PAT = re.compile(r"^(?:\d{1,3}J|\d{2}W|[A-Z]\d{2,3}J|L\d{2,4}J)+")

def normalize_grade(g: str) -> str:
    g = unicodedata.normalize("NFKC", g or "")
    g = g.translate(str.maketrans(
        "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ))
    g = g.replace("R", "Ｒ")
    return g.strip()

# 省略形→正式名の補完（必要に応じて増やせる）
ALIAS_SUFFIX = [
    ("自然科学総合実", "自然科学総合実験"),
    ("健康・スポーツ科", "健康・スポーツ科学演習"),  # あなたの要件上の必修名
    ("サイバーセキュリ", "サイバーセキュリティ基礎論"),
    ("常微分方程式とラ", "常微分方程式とラプラス変換"),
    ("フーリエ解析と偏", "フーリエ解析と偏微分方程式"),
    ("融合基礎工学展", "融合基礎工学展望"),
    ("弾性・塑性変形工", "弾性・塑性変形工学"),
    ("先端技術入門B", "先端技術入門Ｂ"),
]

def clean_name_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = CODE_TAIL_PAT.sub("", s)
    s = HEAD_CODE_PAT.sub("", s)
    s = CATEGORY_HEAD_PAT.sub("", s)  # 区分見出しだけ除去（科目名は消さない）
    s = s.replace(" ", "").replace("　", "")
    s = (s.replace("Ⅰ", "I").replace("Ⅱ", "II").replace("Ⅲ", "III").replace("Ⅳ", "IV"))
    return s.strip()

def apply_alias(name: str, year: int, term: str) -> str:
    # ---- 学術英語系 ----
    if name.startswith("学術英語・アカデミ"):
        return "学術英語・アカデミックイシューズ"
    if name.startswith("学術英語・グロー"):
        return "学術英語・グローバルイシューズ"
    if name.startswith("学術英語・CAL"):
        # CALL1/2の厳密判定が必要なら追加。今はCALL1に寄せる
        return "学術英語・CALL1"
    if name.startswith("学術英語・プロダク"):
        if term == "秋学期":
            return "学術英語・プロダクション1"
        if term == "冬学期":
            return "学術英語・プロダクション2"
        return "学術英語・プロダクション1"

    # ---- 先端技術入門 A/B ----
    if name == "先端技術入門":
        return "先端技術入門A"

    # ---- 自然災害と防災 ----
    if name == "自然災害と防":
        return "自然災害と防災"

    # ---- 省略名（専攻/共通） ----
    if name == "融合基礎情報学":
        if year == 2023 and term == "後":
            return "融合基礎情報学I"
        if year == 2024 and term == "前":
            return "融合基礎情報学II"
        if year == 2024 and term == "後":
            return "融合基礎情報学III"
        return "融合基礎情報学I"

    if name == "電子情報工学基":
        if term == "春学期":
            return "電子情報工学基礎I"
        if term == "夏学期":
            return "電子情報工学基礎II"
        return "電子情報工学基礎I"

    if "安全学" in name:
        return "安全学"

    if name.startswith("機械工学大意第"):
        return "機械工学大意第一"
    
    if name == "基礎化学熱力学":
        return "基礎化学熱力学I" if term == "春学期" else "基礎化学熱力学II" if term == "夏学期" else "基礎化学熱力学I"
    if name == "自然科学総合実":
        return "自然科学総合実験"
    if name == "健康・スポーツ科":
        return "健康・スポーツ科学演習"
    if name == "サイバーセキュリ":
        return "サイバーセキュリティ基礎論"
    if name == "弾性・塑性変形工":
        return "弾性・塑性変形工学"
    if name == "フーリエ解析と偏":
        return "フーリエ解析と偏微分方程式"
    if name == "融合基礎工学展":
        return "融合基礎工学展望"
    if name == "セラミックス材料学":
        return "セラミックス材料学I" if term == "秋学期" else "セラミックス材料学II" if term == "冬学期" else "セラミックス材料学I"

    return name
            
def looks_like_course_name_fragment(s: str) -> bool:
    if not s or is_noise_line(s):
        return False
    s2 = clean_name_text(s)
    if not s2:
        return False
    # 区分見出し単体は recent に入れない
    if CATEGORY_HEAD_PAT.fullmatch(s2):
        return False
    # 日本語がほぼ無いものは捨てる
    if not re.search(r"[一-龯ぁ-んァ-ヶ]", s2):
        return False
    if len(s2) > 40:
        return False
    return True

def looks_truncated(x: str) -> bool:
    if not x:
        return True
    return x.endswith(("ラ", "基", "実", "展", "第", "防", "偏")) or len(x) <= 3

def infer_mm_experiment_suffix(term: str) -> str:
    # あなたのサンプルに合わせた推定：春=I, 夏=II, 秋=III, 冬=IV
    return {"春学期": "I", "夏学期": "II", "秋学期": "III", "冬学期": "IV"}.get(term, "")

def extract_courses_from_pdf(pdf_path: str):
    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for ln in text.splitlines():
                ln = ln.strip()
                if ln:
                    lines.append(ln)

    recent = deque(maxlen=3)
    courses = []

    for ln in lines:
        if is_noise_line(ln):
            recent.clear()
            continue

        if END_OF_RECORD.search(ln):
            m = RECORD_PAT.search(ln)
            if not m:
                recent.clear()
                continue

            name_part = clean_name_text(m.group("name") or "")
            credits = float(m.group("credits"))
            grade = normalize_grade(m.group("grade"))
            gp_raw = m.group("gp")
            year = int(m.group("year"))
            term = m.group("term")

            candidates = []
            if name_part:
                candidates.append(name_part)

            # name が空/途中切れっぽいなら直前行と結合
            if (not name_part) or looks_truncated(name_part):
                if recent:
                    prev = clean_name_text(recent[-1])
                    if prev:
                        candidates.append(clean_name_text(prev + name_part))

            # “nameが完全に空”のときだけ prev 単体も許可
            if (not name_part) and recent:
                prev = clean_name_text(recent[-1])
                if prev:
                    candidates.append(prev)

            name = max(candidates, key=len) if candidates else name_part
            name = clean_name_text(name) 
            name = apply_alias(name, year, term)

            # 特殊：物質材料科学実験I〜IV がPDFで後半が落ちる場合の救済
            if name == "物質材料科学実":
                suf = infer_mm_experiment_suffix(term)
                if suf:
                    name = f"物質材料科学実験{suf}"

            courses.append({
                "name": name,
                "credits": credits,
                "grade": grade,
                "gp": None if gp_raw == "*" else float(gp_raw),
                "year": year,
                "term": term,
                "passed": True,
                "raw": ln,
            })

            recent.clear()
        else:
            if looks_like_course_name_fragment(ln):
                recent.append(ln)

    # 重複排除（name+year+term）
    uniq = {}
    for c in courses:
        uniq[(c["name"], c["year"], c["term"])] = c
    return list(uniq.values())