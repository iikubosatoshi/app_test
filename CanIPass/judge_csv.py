from typing import Dict, Any, List, Optional

# judge.py の既存関数を活用
from judge import required_catalog, scope_allows


def _get(values: Dict[str, Any], label: str) -> Dict[str, Any]:
    """
    values[label] があれば {"value": float|None, "star": bool, "raw": str}
    なければ value=None 扱い
    """
    return values.get(label, {"value": None, "star": False, "raw": ""})


def _val(values: Dict[str, Any], label: str, default: float = 0.0) -> float:
    v = _get(values, label).get("value")
    return float(v) if isinstance(v, (int, float)) else float(default)


def _star(values: Dict[str, Any], label: str) -> bool:
    return bool(_get(values, label).get("star", False))


def _add_result(results: List[Dict[str, Any]], label: str, got: float, need: float, star: bool = False):
    need_f = float(need)
    ok = (got >= need_f) and (not star)  # '*' が付いていたら未達扱い
    results.append({
        "label": label,
        "got": float(got),
        "need": need_f,
        "ok": ok,
        "lack": max(0.0, need_f - float(got)),
        "star": bool(star),
    })


def judge_graduation_from_values(
    values: Dict[str, Any],
    req_course: Dict[str, Any],
    selected_course_key: str,
) -> Dict[str, Any]:
    """
    CSV（判定表）から卒業要件判定。

    必須:
      - values に以下のラベルが存在している想定
        基幹教育科目, 専攻教育科目

    推奨:
      - req_course["graduation"]["min_credits"] は既存と同じ:
        kikan_total, senko_total, senko_required, senko_elective, overall_total

    専攻(必修/選択)は CSVのラベルに合わせて requirements.yaml 側で定義してもらう方式にする。
      req_course["graduation"]["label_sets"]:
        required_labels: [...]
        elective_labels: [...]
    """
    mins = req_course["graduation"]["min_credits"]
    label_sets = req_course["graduation"].get("label_sets", {})

    required_labels = label_sets.get("required_labels", [])
    elective_labels = label_sets.get("elective_labels", [])

    # 基幹・専攻合計（CSV上の集計）
    kikan_total = _val(values, "基幹教育科目")
    kikan_star = _star(values, "基幹教育科目")

    senko_total = _val(values, "専攻教育科目")
    senko_star = _star(values, "専攻教育科目")

    # 専攻(必修/選択)はラベル集合の合計で作る（*付きがあればその項目は未達扱いに回す）
    senko_required = sum(_val(values, x) for x in required_labels)
    senko_required_star = any(_star(values, x) for x in required_labels)

    senko_elective = sum(_val(values, x) for x in elective_labels)
    senko_elective_star = any(_star(values, x) for x in elective_labels)

    overall_total = kikan_total + senko_total
    overall_star = kikan_star or senko_star  # 総計自体の*が無いので保守的に

    results = []
    _add_result(results, "基幹教育科目 合計", kikan_total, mins["kikan_total"], star=kikan_star)
    _add_result(results, "専攻教育科目 合計", senko_total, mins["senko_total"], star=senko_star)
    _add_result(results, "専攻教育（必修）", senko_required, mins["senko_required"], star=senko_required_star)
    _add_result(results, "専攻教育（選択）", senko_elective, mins["senko_elective"], star=senko_elective_star)
    _add_result(results, "総取得単位 合計", overall_total, mins["overall_total"], star=overall_star)

    # 選択必修（復活）
    selreq = judge_select_required_from_values(values, req_course, selected_course_key)
    ok = all(r["ok"] for r in results) and selreq["ok"]

    return {"ok": ok, "totals": results, "select_required": selreq}


def judge_thesis_start_from_values(
    values: Dict[str, Any],
    master: Dict[str, Any],
    req_course: Dict[str, Any],
    selected_course_key: str,
) -> Dict[str, Any]:
    """
    CSV（判定表）から卒論着手要件判定。
    ※ CSVは「ある時点までの集計」なので、judge.py の term_filter は使わない。

    前提:
      - values に「基幹教育科目」「専攻教育科目」がある
      - 4年次開講科目を除外したい場合は、requirements.yaml 側で除外ラベル（または科目）を指定する
    """
    mins = req_course["thesis_start"]["min_credits"]
    cons = req_course["thesis_start"]["constraints"]

    # 基幹（3年後期終了時点のCSVであることを前提）
    kikan_total = _val(values, "基幹教育科目")
    kikan_star = _star(values, "基幹教育科目")

    # 専攻（4年次開講除外）
    # ここも label_sets にして柔軟化（CSVが「専攻教育科目」しか持たない場合があるため）
    label_sets = req_course["thesis_start"].get("label_sets", {})
    senko_excl_labels = label_sets.get("senko_excluding_4th_year_labels", [])

    if senko_excl_labels:
        senko_excl_4 = sum(_val(values, x) for x in senko_excl_labels)
        senko_excl_star = any(_star(values, x) for x in senko_excl_labels)
    else:
        # fallback: 専攻教育科目の数値をそのまま使う（運用上は推奨しないが動く）
        senko_excl_4 = _val(values, "専攻教育科目")
        senko_excl_star = _star(values, "専攻教育科目")

    results = []
    _add_result(results, "基幹教育科目（CSV集計時点まで）", kikan_total, mins["kikan_total"], star=kikan_star)
    _add_result(
        results,
        "専攻教育（4年次開講除外・CSV集計時点まで）",
        senko_excl_4,
        mins["senko_excluding_4th_year"],
        star=senko_excl_star
    )

    # ② 必修（実験・演習）を全て修得：master を使って列（科目名）があるかチェック
    labs_required = required_catalog(master, selected_course_key, labs_only=True, exclude_fourth_year=True)

    missing_labs = []
    for m in labs_required:
        name = m["name"]
        need = float(m.get("credits") or 0.0)
        got = _val(values, name, default=0.0)
        star = _star(values, name)
        # '*' が付いていたら未達、あるいは got < need なら未達
        if star or got < need:
            missing_labs.append(name)

    labs_ok = True
    if cons.get("required_labs_exercises_all_passed", False):
        labs_ok = (len(missing_labs) == 0)

    # ③ 必修（4年次開講除く）の未修得単位数 <= 4
    reqs = required_catalog(master, selected_course_key, labs_only=False, exclude_fourth_year=True)

    missing_required_courses = []
    missing_required_credits = 0.0

    for m in reqs:
        name = m["name"]
        need = float(m.get("credits") or 0.0)
        got = _val(values, name, default=0.0)
        star = _star(values, name)

        # '*' または got < need を未修得扱い
        if star or got < need:
            missing_required_courses.append(name)
            missing_required_credits += need

    max_missing = float(cons.get("required_missing_credits_max", 0.0))
    missing_ok = (missing_required_credits <= max_missing)

    # 選択必修（復活）
    selreq = judge_select_required_from_values(values, req_course, selected_course_key)

    ok = all(r["ok"] for r in results) and labs_ok and missing_ok and selreq["ok"]

    return {
        "ok": ok,
        "totals": results,
        "missing_labs": missing_labs,
        "missing_required_credits": missing_required_credits,
        "max_missing_required_credits": max_missing,
        "missing_required_courses": missing_required_courses,
        "select_required": selreq,
    }


def judge_select_required_from_values(values: Dict[str, Any], req_course: Dict[str, Any], selected_course_key: str) -> Dict[str, Any]:
    """
    選択必修（復活）:
      物質材料:
        常微分方程式とラプラス変換(2), 化学反応論Ⅰ(1), 化学反応論Ⅱ(1) から 2単位以上
      機械電気:
        フーリエ解析と偏微分方程式(2), エネルギー変換工学(2) から 2単位以上

    requirements.yaml 側で:
      req_course["select_required"]["min_credits"]=2
      req_course["select_required"]["materials"]= [...]
      req_course["select_required"]["mech_elec"]= [...]
    を持つ想定（なければデフォルトで判定）
    """
    cfg = req_course.get("select_required", {})
    need = float(cfg.get("min_credits", 2.0))

    if selected_course_key == "material":
        items = cfg.get("material", ["常微分方程式とラプラス変換", "化学反応論Ⅰ", "化学反応論Ⅱ"])
        label = "選択必修（物質材料）"
    else:
        items = cfg.get("mech_elec", ["フーリエ解析と偏微分方程式", "エネルギー変換工学"])
        label = "選択必修（機械電気）"

    got = sum(_val(values, x, default=0.0) for x in items)
    # '*' が付いている科目が含まれる場合は未達扱いを強める（保守的）
    has_star = any(_star(values, x) for x in items)
    ok = (got >= need) and (not has_star)

    return {
        "label": label,
        "need": need,
        "got": got,
        "ok": ok,
        "items": [{"name": x, "value": _val(values, x, 0.0), "star": _star(values, x)} for x in items],
    }