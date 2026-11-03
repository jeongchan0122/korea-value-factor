"""
한국 DataGuide 원본 캐시(processed/raw_cache)를 깔끔한 표로 바꾸는 파서 모음.

행 번호는 하드코딩하지 않고, A열(c0)에서 '코드', '아이템명' 같은 머리글 셀을 찾아 위치를 정한다.
형태별 함수:
  parse_calendar   달력기준 종목 파일(넓은 표)  -> (연월, 종목코드) x 아이템 열
  parse_timeseries 달력기준 지수·금리·경제지표  -> 연월 x 시리즈 열
  parse_accounting 회계기준(결산·분기)          -> (종목코드, 회계연도[, 분기]) x 아이템 열
  parse_snapshot   특정시점(29번)               -> 종목코드 x 아이템 열
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "processed" / "raw_cache"

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def load_raw(stem):
    return pd.read_parquet(RAW / f"{stem}__Sheet1.parquet")


def find_row(df, label):
    """A열에서 label과 정확히 같은 셀의 행 위치. 없으면 에러."""
    hits = df.index[df["c0"] == label]
    if len(hits) == 0:
        raise ValueError(f"A열에서 '{label}' 머리글을 찾지 못함")
    return hits[0]


def to_number_if_possible(s):
    """전부 숫자로 바뀌면 숫자로, 글자가 섞여 있으면(예: KS/KQ, 업종명) 그대로 둔다."""
    num = pd.to_numeric(s, errors="coerce")
    return num if num.notna().sum() == s.notna().sum() else s


def _calendar_block(df):
    """달력기준 공통: 머리글(코드·코드명·아이템명)과 데이터 영역(날짜 행)을 분리."""
    r_code, r_name, r_item = find_row(df, "코드"), find_row(df, "코드명"), find_row(df, "아이템명")
    is_date = df["c0"].fillna("").str.match(DATE_RE)
    body = df[is_date]
    dates = pd.to_datetime(body["c0"].str[:10])
    header = pd.DataFrame({
        "종목코드": df.loc[r_code].iloc[1:].values,
        "종목명": df.loc[r_name].iloc[1:].values,
        "아이템": df.loc[r_item].iloc[1:].values,
    }, index=df.columns[1:])
    header = header[header["종목코드"].notna()]
    return header, body[header.index], dates


def parse_calendar(stem):
    """
    달력기준 종목 파일 -> 긴 표 (연월, 종목코드, 종목명, <아이템1>, <아이템2> ...).
    아이템이 2개인 파일은 열이 종목마다 번갈아 나오므로 아이템명으로 구분해 열로 펼친다.
    모든 아이템이 빈칸인 (월, 종목) 행은 버린다(미상장·상폐 이후).
    """
    df = load_raw(stem)
    header, body, dates = _calendar_block(df)
    body = body.copy()
    body.index = dates.dt.strftime("%Y-%m").values
    body.index.name = "연월"

    long = body.melt(ignore_index=False, var_name="열", value_name="값").reset_index()
    long = long[long["값"].notna()]
    long = long.join(header, on="열").drop(columns="열")

    wide = long.pivot_table(index=["연월", "종목코드", "종목명"], columns="아이템", values="값", aggfunc="first")
    wide.columns.name = None
    wide = wide.reset_index()
    for c in wide.columns[3:]:
        wide[c] = to_number_if_possible(wide[c])
    if wide.duplicated(["연월", "종목코드"]).any():
        raise ValueError(f"{stem}: (연월, 종목코드) 중복 발생")
    return wide.sort_values(["종목코드", "연월"]).reset_index(drop=True)


def parse_daily(stem):
    """
    달력기준 일간 출력 종목 파일(40~43번 상폐종목 일간 수정주가) -> 날짜 x 종목코드 넓은 표(숫자).
    한 달에 여러 줄이라 parse_calendar(연월 기준)를 쓸 수 없다.
    """
    df = load_raw(stem)
    header, body, dates = _calendar_block(df)
    out = body.apply(pd.to_numeric, errors="coerce")
    out.index = pd.DatetimeIndex(dates.values, name="날짜")
    out.columns = header["종목코드"].values
    return out


def parse_timeseries(stem):
    """
    달력기준 시계열(지수·금리·경제지표) -> 연월 x 시리즈.
    시리즈 이름: 지수처럼 코드가 시리즈별로 다르면 '코드'(I.001, 3FM.2B3.* 등),
    경제지표처럼 코드가 'Economy'로 같으면 '아이템명'.
    반환: (시계열 표, 시리즈 설명표[열이름, 코드, 코드명, 아이템명])
    """
    df = load_raw(stem)
    header, body, dates = _calendar_block(df)
    use_item = header["종목코드"].nunique() < len(header)
    names = header["아이템"] if use_item else header["종목코드"]
    out = body.copy()
    out.columns = names.values
    out = out.apply(pd.to_numeric, errors="coerce")
    out.insert(0, "날짜", dates.values)
    out.insert(0, "연월", dates.dt.strftime("%Y-%m").values)
    info = pd.DataFrame({"열이름": names.values, "코드": header["종목코드"].values,
                         "코드명": header["종목명"].values, "아이템명": header["아이템"].values})
    return out.reset_index(drop=True), info


def parse_accounting(stem):
    """
    회계기준 파일 -> (종목코드, 종목명, 결산월, 회계연도, 분기, <아이템...>).
    머리글 행('코드'로 시작)의 바로 위 행이 아이템명이다.
    """
    df = load_raw(stem)
    r_head = find_row(df, "코드")
    head = df.loc[r_head]
    items = df.loc[r_head - 1]
    item_codes = df.loc[r_head - 2]
    base = set(head.dropna())
    # 아이템명이 머리글과 겹치면(14번의 '결산월') 아이템 코드를 붙여 구분
    cols = [head[c] if pd.isna(items[c])
            else f"{items[c]}_{item_codes[c]}" if items[c] in base
            else items[c] for c in df.columns]
    out = df.loc[r_head + 1:].copy()
    out.columns = cols
    out = out.rename(columns={"코드": "종목코드", "코드명": "종목명"})
    out = out[out["종목코드"].notna()]
    for c in out.columns:
        if c not in ("종목코드", "종목명", "분기"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["결산월"] = out["결산월"].astype("Int64")
    out["회계연도"] = out["회계연도"].astype("Int64")
    return out.reset_index(drop=True)


def parse_snapshot(stem):
    """
    특정시점 파일(29번) -> 종목코드 x 아이템.
    '상장일'이 두 개(CP10001550, S411006500)라 아이템 코드(머리글 바로 위 행)를 이름에 붙여 구분한다.
    YYYYMMDD 숫자는 날짜로 바꾼다.
    """
    df = load_raw(stem)
    r_head = find_row(df, "코드")
    head = df.loc[r_head]
    codes = df.loc[r_head - 1]
    cols = ["종목코드", "종목명"] + [f"{head[c]}_{codes[c]}" for c in df.columns[2:]]
    out = df.loc[r_head + 1:].copy()
    out.columns = cols
    for c in cols[2:]:
        out[c] = pd.to_datetime(pd.to_numeric(out[c], errors="coerce").astype("Int64").astype("string"),
                                format="%Y%m%d", errors="coerce")
    return out.reset_index(drop=True)
