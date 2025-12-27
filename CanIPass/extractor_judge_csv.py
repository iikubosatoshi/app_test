import csv
import unicodedata
from typing import Dict, Any, List, Optional, Tuple


def _norm(s: Any) -> str:
    s = unicodedata.normalize("NFKC", "" if s is None else str(s))
    return s.strip()


def _parse_cell(x: Any) -> Tuple[Optional[float], bool, str]:
    """
    Returns:
      value: float or None
      is_star: bool  # '*' 付き＝不足/未達
      raw: str
    """
    raw = _norm(x)
    if raw == "":
        return None, False, ""
    is_star = raw.startswith("*")
    s = raw[1:] if is_star else raw
    try:
        return float(s), is_star, raw
    except Exception:
        return None, is_star, raw


def load_judge_csv(path: str) -> Dict[str, Any]:
    """
    Excel出力の判定表CSV（横持ち）を読み込んで、次の形にする:
      {
        "meta": {...},
        "labels": {"001": "修得単位数計", ...},
        "values": {"基幹教育科目": {"value": 28.5, "star": True, "raw": "*28.5"}, ...}
      }

    前提:
      - ヘッダFLG=1 の行が "001..xxx" のラベル定義行
      - 同じ判定基準番号のデータ行が複数あることがある（あなたのサンプルもそう）
        → その場合は "修得単位数計" が最大の行を採用（より集計が進んだ行を優先）
    """
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # 1) ラベル定義行（ヘッダFLG=1）を探す
    header_row = None
    for r in rows:
        if _norm(r.get("ヘッダFLG")) == "1":
            header_row = r
            break
    if header_row is None:
        raise ValueError("ヘッダFLG=1 の行（項目ラベル定義）が見つかりません")

    # 2) "001".."196" の列を抽出
    code_cols = [k for k in header_row.keys() if k and k.isdigit()]
    code_to_label = {}
    for c in code_cols:
        label = _norm(header_row.get(c))
        if label:
            code_to_label[c] = label

    # 3) データ行を候補として集める
    candidates = []
    for r in rows:
        if _norm(r.get("ヘッダFLG")) == "1":
            continue

        values = {}
        for c, label in code_to_label.items():
            v, star, raw = _parse_cell(r.get(c))
            if v is None and raw == "":
                continue
            values[label] = {"value": v, "star": star, "raw": raw}

        meta = {
            "判定基準番号": _norm(r.get("判定基準番号")),
            "学年": _norm(r.get("学年")),
            "本判定・見込区分名": _norm(r.get("本判定・見込区分名")),
            "判定条件区分名": _norm(r.get("判定条件区分名")),
            "所属略称": _norm(r.get("所属略称")),
            "判定結果区分名": _norm(r.get("判定結果区分名")),
        }

        candidates.append({"meta": meta, "values": values})

    # 4) 同一判定基準番号で複数行ある場合は「修得単位数計」が最大の行を採用
    #    なければ先頭
    def score(item):
        v = item["values"].get("修得単位数計", {}).get("value")
        return v if isinstance(v, (int, float)) else -1

    best = None
    for it in candidates:
        if best is None or score(it) > score(best):
            best = it
    if best is None:
        raise ValueError("データ行が見つかりません")

    return {
        "meta": best["meta"],
        "labels": code_to_label,
        "values": best["values"],
    }