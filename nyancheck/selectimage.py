# selectimage.py  — Streamlit/汎用向けの軽量版
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import random

def randomselect(
    root_dir: str,
    k: int = 10,
    extensions: Sequence[str] = (".png", ".jpg", ".jpeg", ".webp"),
    recursive: bool = True,
    seed: int | None = None,
) -> list[str]:
    """
    root_dir 以下から画像ファイルをランダムに k 枚選んで「フルパスの文字列リスト」を返す。
    - extensions: 拾う拡張子（大文字小文字は自動で無視）
    - recursive: 下位ディレクトリも探索するか
    - seed: 乱数シード（テストや再現性が必要なときに指定）
    """
    root = Path(root_dir)
    if not root.exists():
        return []

    if seed is not None:
        random.seed(seed)

    # 拡張子を小文字化して比較
    exts = tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions)

    it = root.rglob("*") if recursive else root.glob("*")
    files = [
        p for p in it
        if p.is_file()
        and p.suffix.lower() in exts
        and not p.name.startswith(".")            # 隠しファイル除外
    ]

    if not files:
        return []

    # 必要枚数が総数以上なら、シャッフルして先頭 k
    if k >= len(files):
        random.shuffle(files)
        return [str(p) for p in files[:k]]

    # 重複なしでランダム抽出
    return [str(p) for p in random.sample(files, k)]


# （任意）DataFrame が必要な旧コード互換のヘルパー
def randomselect_df(root_dir: str, k: int = 10, **kwargs):
    """選んだ画像の filename / path を DataFrame にして返す（旧Flaskテンプレ互換）"""
    import pandas as pd
    paths = randomselect(root_dir, k, **kwargs)
    rows = [{"filename": Path(p).name, "path": p} for p in paths]
    return pd.DataFrame(rows)
