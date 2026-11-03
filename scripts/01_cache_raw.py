"""
2단계: 한국 DataGuide 엑셀(한국데이터 폴더 전체)을 '날것 그대로' parquet로 캐시하고 구조를 요약한다.

- 원본 엑셀은 읽기만 한다(수정 X).
- 머리글 없이(header=None) 셀을 통째로 읽고, 모든 셀을 문자열로 저장한다.
  (한 열에 머리글 글자와 숫자가 섞여 있어 그대로는 parquet 저장이 안 되기 때문. 숫자 변환은 3단계 파서에서)
- 이미 캐시된 파일은 건너뛴다(--force 로 다시 읽기).
- 구조 요약은 processed/raw_cache/_structure.txt 에 저장한다.

실행: python scripts/01_cache_raw.py [--force]
"""
import sys
import time
from pathlib import Path

import pandas as pd
from python_calamine import CalamineWorkbook

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "한국데이터"
OUT = ROOT / "processed" / "raw_cache"
FORCE = "--force" in sys.argv


def to_str(v):
    """셀 값을 문자열로. 빈칸은 None으로 둔다."""
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return None
    return str(v)


def read_raw(path):
    """엑셀의 모든 시트를 {시트명: 문자열 DataFrame}으로 읽는다."""
    wb = CalamineWorkbook.from_path(str(path))
    sheets = {}
    for name in wb.sheet_names:
        rows = wb.get_sheet_by_name(name).to_python(skip_empty_area=False)
        width = max((len(r) for r in rows), default=0)
        data = [[to_str(v) for v in r] + [None] * (width - len(r)) for r in rows]
        sheets[name] = pd.DataFrame(data, columns=[f"c{i}" for i in range(width)], dtype="object")
    return sheets


def summarize(fname, sheet, df):
    """구조 확인용 요약: 크기, 앞 30행(앞 5열), 마지막 3행, 마지막 열의 머리글."""
    lines = [f"\n{'=' * 100}\n{fname}  [시트: {sheet}]  shape={df.shape}\n{'=' * 100}"]
    with pd.option_context("display.width", 250, "display.max_colwidth", 40):
        lines.append("-- 앞 30행 x 앞 5열 --")
        lines.append(df.iloc[:30, :5].to_string())
        lines.append("-- 마지막 3행 x 앞 5열 --")
        lines.append(df.iloc[-3:, :5].to_string())
        lines.append("-- 마지막 열(앞 15행 + 마지막 3행) --")
        last = df.iloc[:, -1]
        lines.append(pd.concat([last.iloc[:15], last.iloc[-3:]]).to_string())
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = []
    for path in sorted(SRC.glob("*.xlsx")):
        if path.name.startswith("~$"):  # 엑셀이 파일을 열고 있을 때 생기는 임시 파일
            continue
        stem = path.stem
        cached = sorted(OUT.glob(f"{stem}__*.parquet"))
        t0 = time.time()
        if cached and not FORCE:
            sheets = {p.stem.split("__", 1)[1]: pd.read_parquet(p) for p in cached}
            how = "캐시"
        else:
            sheets = read_raw(path)
            for name, df in sheets.items():
                df.to_parquet(OUT / f"{stem}__{name}.parquet", index=False)
            how = "엑셀"
        shapes = ", ".join(f"{n}{df.shape}" for n, df in sheets.items())
        print(f"{stem:<40} {how} {time.time() - t0:6.1f}s  {shapes}", flush=True)
        for name, df in sheets.items():
            report.append(summarize(path.name, name, df))

    (OUT / "_structure.txt").write_text("\n".join(report), encoding="utf-8")
    print(f"\n구조 요약 저장: {OUT / '_structure.txt'}")


if __name__ == "__main__":
    main()
