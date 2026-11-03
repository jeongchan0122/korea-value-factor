"""
상장폐지 수익률 계산: 월별 수익률(03번)에 빠져 있는 '마지막 달 이후 ~ 정리매매 마지막 날' 수익률을
상폐종목 일간 수정주가(40~43번)로 구해 processed/상폐수익률.parquet 로 저장한다.
04_merge_korea.py 가 이 파일을 읽어 월별패널에 상폐한 달 줄을 추가한다.

왜 필요한가: 월말 가격 기준이라 월 중간에 상폐되면 그달 수익률이 없다. 빈칸이면 계산에서 빠져
"전달 말 가격에 손해 없이 팔고 나간 것"처럼 처리돼, 실제로는 피할 수 없는 손실이 결과에서 빠진다.
(확인 결과: 2×3 HML에는 영향이 거의 없었다 1.17% → 1.18%. 망한 회사가 가치주와 소형 성장주 양쪽에
 있어 상쇄됨. 개별 바구니, 특히 소형주 바구니 수익률은 조금 낮아짐)

규칙
- 기준가 = 월별 수익률이 있는 마지막 달의 말일 이하 마지막 일간 가격
- 마지막가 = 상장폐지일 전날까지의 마지막 일간 가격(정리매매 마지막 날). 43번(K-OTC 이전 종목)은
  상폐 뒤 K-OTC 가격이 이어지므로 상장폐지일 이전으로 자른다
- 기준가 이후 거래가 없으면 0%
- +300% 초과(감자 등이 수정주가에 반영 안 된 오류로 보임)나 기준가 없음 → 가정값
  가정값 = 실제 값의 그룹 평균(마지막 달에 관리종목·거래정지였는지로 구분)
- 반영연월 = 월별 수익률이 있는 마지막 달의 다음 달

실행: python scripts/02b_상폐수익률.py   (02_parse_korea.py 다음, 04_merge_korea.py 전에)
"""
import numpy as np
import pandas as pd

from korea_parsers import ROOT

TIDY = ROOT / "processed" / "tidy"
OUT = ROOT / "processed"
DAILY_FILES = ["40_상폐종목_일간수정주가_1999_2009", "41_상폐종목_일간수정주가_2010_2019",
               "42_상폐종목_일간수정주가_2020_2026", "43_누락상폐종목_일간수정주가"]
ERROR_THRESHOLD = 300  # % 초과면 오류로 본다


def load_daily_prices():
    parts = [pd.read_parquet(TIDY / f"{f}.parquet").set_index("날짜") for f in DAILY_FILES]
    prices = parts[0]
    for p in parts[1:]:
        prices = prices.combine_first(p)  # 같은 종목·날짜가 겹치면 앞 파일 값 유지
    return prices.sort_index()


def main():
    prices = load_daily_prices()

    listing = pd.read_parquet(TIDY / "29_상장일_상폐일.parquet").set_index("종목코드")
    delist = listing["상장폐지일_CP10001600"]
    relisted = delist < listing["상장일_CP10001550"]  # 상폐 후 재상장(예: 우양에이치씨)은 제외
    delist = delist[(delist >= "1999-01-01") & ~relisted]

    ret = pd.read_parquet(TIDY / "03_월수익률_배당포함.parquet")
    ret = ret[ret["수정주가수익률(현금배당포함)(%)"].notna()]
    last_month = ret.groupby("종목코드")["연월"].max()

    halt = pd.read_parquet(TIDY / "04_거래정지.parquet").set_index(["종목코드", "연월"])["거래정지구분"]
    admin = pd.read_parquet(TIDY / "05_관리종목.parquet").set_index(["종목코드", "연월"])["관리감리구분"]

    rows = []
    for code, dl in delist.items():
        if code not in last_month.index:
            continue  # 월별 수익률이 한 번도 없음(1999년 1월 이전 상폐 등)
        lm = last_month[code]
        month_end = pd.Period(lm, "M").end_time.normalize()
        x = prices[code].dropna() if code in prices.columns else pd.Series(dtype=float)
        x = x[x.index < dl]
        base, after = x[x.index <= month_end], x[x.index > month_end]

        distress = halt.get((code, lm)) == "거래정지" or admin.get((code, lm)) == "관리"
        if len(base) == 0:
            r, src, last_day = np.nan, "가정(기준가 없음)", pd.NaT
        elif len(after) == 0:
            r, src, last_day = 0.0, "거래없음(0%)", base.index[-1]
        else:
            r = (after.iloc[-1] / base.iloc[-1] - 1) * 100
            src, last_day = "실제", after.index[-1]
            if r > ERROR_THRESHOLD:
                r, src = np.nan, "가정(오류값)"
        rows.append({"종목코드": code, "종목명": listing.at[code, "종목명"], "상장폐지일": dl,
                     "마지막수익률월": lm, "반영연월": str(pd.Period(lm, "M") + 1),
                     "마지막거래일": last_day, "상폐수익률_%": r, "부실신호": distress, "출처": src})
    d = pd.DataFrame(rows)

    actual = d[d["출처"] == "실제"]
    fill = actual.groupby("부실신호")["상폐수익률_%"].mean()
    need = d["상폐수익률_%"].isna()
    d.loc[need, "상폐수익률_%"] = d.loc[need, "부실신호"].map(fill)

    d.to_parquet(OUT / "상폐수익률.parquet", index=False)
    print(f"상폐 종목 {len(d)}개 → {OUT / '상폐수익률.parquet'}")
    print("출처별:", d["출처"].value_counts().to_dict())
    print(f"가정값: 부실신호 {fill.get(True, np.nan):.1f}%, 그 외 {fill.get(False, np.nan):.1f}%")
    s = d["상폐수익률_%"]
    print(f"상폐수익률: 평균 {s.mean():.1f}%, 중앙값 {s.median():.1f}%, "
          f"−90% 이하 {(s <= -90).mean():.1%}, ±1% 이내 {s.abs().le(1).mean():.1%}")


if __name__ == "__main__":
    main()
