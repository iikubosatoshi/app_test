import csv
import unicodedata

# -----------------------------
# 正規化ユーティリティ
# -----------------------------
_ROMAN = str.maketrans({"Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III", "Ⅳ": "IV"})


def norm(s: str) -> str:
    """course_master と抽出結果を同じルールで突合するための正規化（強化版）"""
    s = s or ""

    # 目に見えない文字対策（BOM / ゼロ幅など）
    s = s.replace("\ufeff", "").replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")

    # NFKC（全角→半角、互換文字の統一）
    s = unicodedata.normalize("NFKC", s)

    # ありがちな表記差を強制統一（NFKCで落ちないケースの保険）
    s = s.replace("（", "(").replace("）", ")")

    # 空白除去（半角/全角）
    s = s.replace(" ", "").replace("　", "")

    # ローマ数字統一
    s = s.translate(_ROMAN)

    return s.strip()


# -----------------------------
# normalize_name（互換API）：norm + alias で最終キーに寄せる
# -----------------------------
_ALIAS_AFTER_NORM = {
    # 基幹（短縮）
    "自然科学総合実": "自然科学総合実験",
    "健康・スポーツ科": "健康・スポーツ科学演習",
    "サイバーセキュリ": "サイバーセキュリティ基礎論",
    "先端技術入門B": "先端技術入門B",  # NFKCで揃う想定（masterがＢでもOK）
    # 専攻/共通（短縮）
    "弾性・塑性変形工": "弾性・塑性変形工学",
    "フーリエ解析と偏": "フーリエ解析と偏微分方程式",
    "融合基礎工学展": "融合基礎工学展望",
    "常微分方程式とラ": "常微分方程式とラプラス変換",
    # これで未分類が残りやすい3つ
    "グローバル科目I": "グローバル科目I(論文)",
    "グローバル科目II": "グローバル科目II(討論)",
    "グローバル科目I(論文)": "グローバル科目I(論文)",
    "グローバル科目II(討論)": "グローバル科目II(討論)",
    "データサイエンス": "データサイエンス序論",
}


def normalize_name(s: str) -> str:
    s = norm(s)
    return _ALIAS_AFTER_NORM.get(s, s)


# -----------------------------
# マスタ読み込み / 付与
# -----------------------------
def load_master(path: str):
    """
    course_master.csv:
    name,credits,area,kind,is_lab_or_exercise,is_fourth_year,scope
    """
    master = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_name = row.get("name", "")
            key = normalize_name(raw_name)
            if not key:
                continue

            master[key] = {
                "name": key,
                "credits": float(row["credits"]) if row.get("credits") else None,
                "area": row.get("area"),  # 基幹 / 専攻
                "kind": row.get("kind"),  # 必修 / 選択
                "is_lab_or_exercise": str(row.get("is_lab_or_exercise", "")).strip().lower() == "true",
                "is_fourth_year": str(row.get("is_fourth_year", "")).strip().lower() == "true",
                "scope": row.get("scope"),  # 共通 / 物質材料 / 機械電気
            }
    return master


def apply_master(courses, master):
    """
    courses: extractor.py の結果（name, credits, year, term, grade...）
    master が見つかれば area/kind/is_fourth_year... を付与
    """
    out = []
    for c in courses:
        key = normalize_name(c.get("name", ""))
        info = master.get(key)
        if info:
            out.append({**c, **info, "name": key})
        else:
            out.append(
                {
                    **c,
                    "name": key,
                    "area": None,
                    "kind": None,
                    "is_lab_or_exercise": False,
                    "is_fourth_year": False,
                    "scope": None,
                }
            )
    return out


# -----------------------------
# 期間（3年後期終了時点）フィルタ
# -----------------------------
def filter_by_thesis_point(courses, term_order: dict, admission_year: int, max_term: str = "冬学期"):
    y3 = admission_year + 2
    max_ord = term_order.get(max_term, 999)

    out = []
    for c in courses:
        y = c.get("year")
        t = c.get("term")
        if y is None or t is None:
            continue
        if y < y3:
            out.append(c)
        elif y == y3 and term_order.get(t, 999) <= max_ord:
            out.append(c)
    return out


# -----------------------------
# コース適用（scope）
# -----------------------------
def scope_allows(scope_value: str, selected_course_key: str) -> bool:
    if scope_value is None:
        return False
    if scope_value == "共通":
        return True
    if selected_course_key == "material" and scope_value == "物質材料":
        return True
    if selected_course_key == "mech_elec" and scope_value == "機械電気":
        return True
    return False


# -----------------------------
# 集計ユーティリティ
# -----------------------------
def sum_credits(courses, *, area=None, kind=None, exclude_fourth_year=False, selected_course_key=None):
    total = 0.0
    for c in courses:
        if c.get("area") is None or c.get("kind") is None:
            continue
        if area and c.get("area") != area:
            continue
        if kind and c.get("kind") != kind:
            continue
        if exclude_fourth_year and c.get("is_fourth_year"):
            continue
        if selected_course_key and c.get("area") == "専攻":
            if not scope_allows(c.get("scope"), selected_course_key):
                continue
        total += float(c.get("credits", 0.0))
    return total


def required_catalog(master, selected_course_key: str, *, labs_only=False, exclude_fourth_year=False):
    req = []
    for m in master.values():
        if m.get("area") != "専攻":
            continue
        if m.get("kind") != "必修":
            continue
        if exclude_fourth_year and m.get("is_fourth_year"):
            continue
        if not scope_allows(m.get("scope"), selected_course_key):
            continue
        if labs_only and not m.get("is_lab_or_exercise"):
            continue
        req.append(m)
    return req


# -----------------------------
# 判定：卒業要件（選択必修なし）
# -----------------------------
def judge_graduation(courses, req_course, selected_course_key: str):
    mins = req_course["graduation"]["min_credits"]
    results = []

    kikan_total = sum_credits(courses, area="基幹")
    senko_total = sum_credits(courses, area="専攻", selected_course_key=selected_course_key)
    senko_required = sum_credits(courses, area="専攻", kind="必修", selected_course_key=selected_course_key)
    senko_elective = sum_credits(courses, area="専攻", kind="選択", selected_course_key=selected_course_key)

    overall_total = kikan_total + senko_total

    def add(label, got, need):
        need = float(need)
        results.append(
            {"label": label, "got": got, "need": need, "ok": got >= need, "lack": max(0.0, need - got)}
        )

    add("基幹教育科目 合計", kikan_total, mins["kikan_total"])
    add("専攻教育科目 合計（共通＋自コース）", senko_total, mins["senko_total"])
    add("専攻教育（必修）", senko_required, mins["senko_required"])
    add("専攻教育（選択）", senko_elective, mins["senko_elective"])
    add("総取得単位 合計", overall_total, mins["overall_total"])
    
    # 選択必修グループ判定（コース別）
    groups = req_course["graduation"].get("elective_required_groups", [])
    for g in groups:
        got, taken = sum_credits_for_named_set(
            courses,
            g.get("courses", []),
            selected_course_key=selected_course_key
        )
        need = float(g.get("min_credits", 0))
        results.append({
            "label": g.get("label", "選択必修"),
            "got": got,
            "need": need,
            "ok": got >= need,
            "lack": max(0.0, need - got),
            "taken": taken,   # UIで表示したい場合に使える
        })

    ok = all(r["ok"] for r in results)
    return {"ok": ok, "totals": results}


# -----------------------------
# 判定：卒論着手要件（選択必修なし）
# -----------------------------
def judge_thesis_start(courses, master, req_course, selected_course_key: str, term_order: dict, admission_year: int):
    upto = filter_by_thesis_point(courses, term_order, admission_year, max_term="冬学期")

    mins = req_course["thesis_start"]["min_credits"]
    cons = req_course["thesis_start"]["constraints"]
    results = []

    def add(label, got, need):
        need = float(need)
        results.append(
            {"label": label, "got": got, "need": need, "ok": got >= need, "lack": max(0.0, need - got)}
        )

    kikan_total = sum_credits(upto, area="基幹")
    add("基幹教育科目（3年後期終了まで）", kikan_total, mins["kikan_total"])

    senko_excl_4 = sum_credits(upto, area="専攻", exclude_fourth_year=True, selected_course_key=selected_course_key)
    add("専攻教育（4年次開講除外・3年後期終了まで）", senko_excl_4, mins["senko_excluding_4th_year"])

    taken_names = set(normalize_name(c.get("name", "")) for c in upto)

    labs_required = required_catalog(master, selected_course_key, labs_only=True, exclude_fourth_year=True)
    missing_labs = [m["name"] for m in labs_required if m["name"] not in taken_names]
    labs_ok = True
    if cons.get("required_labs_exercises_all_passed", False):
        labs_ok = (len(missing_labs) == 0)

    reqs = required_catalog(master, selected_course_key, labs_only=False, exclude_fourth_year=True)
    missing_required = [m for m in reqs if m["name"] not in taken_names]
    missing_required_credits = sum(m["credits"] for m in missing_required if m.get("credits") is not None)

    max_missing = float(cons.get("required_missing_credits_max", 0))
    missing_ok = (missing_required_credits <= max_missing)

    ok = all(r["ok"] for r in results) and labs_ok and missing_ok

    return {
        "ok": ok,
        "totals": results,
        "missing_labs": missing_labs,
        "missing_required_credits": missing_required_credits,
        "max_missing_required_credits": max_missing,
        "missing_required_courses": [m["name"] for m in missing_required],
    }

def sum_credits_for_named_set(courses, names, *, selected_course_key=None):
    """
    courses: apply_master 後の course list
    names: 選択必修対象科目名のリスト（requirements.yaml）
    """
    target = set(normalize_name(n) for n in names)
    total = 0.0
    taken = []

    for c in courses:
        if c.get("area") is None:
            continue
        if c.get("area") != "専攻":
            continue

        # scope フィルタ（共通＋自コースのみ）
        if selected_course_key:
            if not scope_allows(c.get("scope"), selected_course_key):
                continue

        if normalize_name(c.get("name", "")) in target:
            total += float(c.get("credits", 0.0))
            taken.append(c.get("name"))

    return total, taken