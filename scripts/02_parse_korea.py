"""
3단계: raw_cache를 형태별 파서로 정리해 processed/tidy/<파일명>.parquet 로 저장한다.
(여러 파일을 합치는 건 4단계)

실행: python scripts/02_parse_korea.py
"""
import time

from korea_parsers import (ROOT, RAW, parse_accounting, parse_calendar, parse_daily, parse_snapshot,
                           parse_timeseries)

OUT = ROOT / "processed" / "tidy"
TIMESERIES = {"07", "08", "09", "16"}
ACCOUNTING = {"12", "13", "14", "15", "17", "18", "19", "20", "21", "37", "38", "39"}
SNAPSHOT = {"29"}
DAILY = {"40", "41", "42", "43"}  # 상폐종목 일간 수정주가: 날짜 x 종목코드 넓은 표로 저장


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for p in sorted(RAW.glob("*__Sheet1.parquet")):
        stem = p.stem.split("__")[0]
        num = stem[:2]
        t0 = time.time()
        if num in TIMESERIES:
            df, info = parse_timeseries(stem)
            info.to_parquet(OUT / f"{stem}__info.parquet", index=False)
        elif num in ACCOUNTING:
            df = parse_accounting(stem)
        elif num in SNAPSHOT:
            df = parse_snapshot(stem)
        elif num in DAILY:
            df = parse_daily(stem).reset_index()
        else:
            df = parse_calendar(stem)
        df.to_parquet(OUT / f"{stem}.parquet", index=False)
        extra = [c for c in df.columns if c not in ("연월", "날짜", "종목코드", "종목명")]
        print(f"{stem:<36} {time.time() - t0:5.1f}s  {str(df.shape):<16} {extra[:4]}{' ...' if len(extra) > 4 else ''}",
              flush=True)


if __name__ == "__main__":
    main()
