"""
3단계 검증: processed/tidy 결과가 한국데이터_설명.md 7장 검증값과 맞는지 확인한다.

실행: python scripts/03_verify_korea.py
"""
import pandas as pd

from korea_parsers import ROOT

TIDY = ROOT / "processed" / "tidy"
SAMSUNG, CELLTRION, SK, KAKAO, MIRAE = "A005930", "A068270", "A034730", "A035720", "A006800"
results = []


def load(stem):
    return pd.read_parquet(TIDY / f"{stem}.parquet")


def col(df, keyword):
    """열 이름에 keyword가 들어간 열 하나를 찾는다."""
    hits = [c for c in df.columns if keyword in c]
    assert len(hits) == 1, f"'{keyword}' 열 후보: {hits}"
    return hits[0]


def check(label, actual, expected, tol=0):
    ok = (abs(actual - expected) <= tol) if isinstance(expected, (int, float)) and pd.notna(actual) else actual == expected
    results.append(ok)
    print(f"[{'OK' if ok else '틀림'}] {label}: 실제={actual!r}  기대={expected!r}")


def val(df, code, ym, keyword):
    row = df[(df["종목코드"] == code) & (df["연월"] == ym)]
    return row[col(df, keyword)].iloc[0] if len(row) else None


def fin(df, code, year, keyword, quarter=None):
    m = (df["종목코드"] == code) & (df["회계연도"] == year)
    if quarter:
        m &= df["분기"] == quarter
    return df.loc[m, col(df, keyword)].iloc[0]


# 01 / 02: 첫 데이터 날짜
for stem in ["01_월수익률", "02_시가총액"]:
    check(f"{stem} 첫 연월", load(stem)["연월"].min(), "1999-01")

# 02: 삼성전자 시가총액으로 단위 판정
mc = load("02_시가총액")
s = val(mc, SAMSUNG, "2026-09", "시가총액")
print(f"[참고] 02 삼성전자 2026-09 시가총액 = {s:,.0f} (열 이름 '{col(mc, '시가총액')}') -> 백만원이면 약 {s / 1e6:,.0f}조원")

# 06: 셀트리온 시장구분 변화
mk = load("06_시장구분")
c = mk[mk["종목코드"] == CELLTRION].set_index("연월")["시장구분"]
check("06 셀트리온 2005-06", c["2005-06"], "EX")
check("06 셀트리온 2005-07", c["2005-07"], "KQ")
check("06 셀트리온 2018-01", c["2018-01"], "KQ")
check("06 셀트리온 2018-02", c["2018-02"], "KS")

# 10: 업종
sec = load("10_업종_Fnguide")
check("10 첫 값 연월", sec["연월"].min(), "2000-01")
sk = sec[sec["종목코드"] == SK].set_index("연월")["FnGuide Sector"]
print(f"[참고] 10 SK 업종 변화: {sk[sk != sk.shift()].to_dict()}")
kk = sec[sec["종목코드"] == KAKAO]["FnGuide Sector"]
print(f"[참고] 10 카카오 업종 종류: {kk.unique().tolist()}")

# 11: 거래대금
check("11 삼성전자 2026-09 거래대금", val(load("11_거래대금"), SAMSUNG, "2026-09", "거래대금"), 89_569_507_007_221)

# 12~15: 연간 재무
bs = load("12_재무_재무상태표")
check("12 삼성전자 2025 자산총계", fin(bs, SAMSUNG, 2025, "자산총계"), 566_942_110_000)
check("12 첫 회계연도(값 있는 행)", int(bs.loc[bs[col(bs, "자산총계")].notna(), "회계연도"].min()), 1998)
check("13 삼성전자 2024 유형자산순취득액",
      fin(load("13_재무_손익현금흐름"), SAMSUNG, 2024, "유형자산순취득액"), 51_250_164_000)
dv = load("14_재무_배당주식")
mm = dv[dv["종목코드"] == MIRAE].set_index("회계연도")[col(dv, "결산월_")]
check("14 미래에셋증권 2012 결산월", mm[2012], 3)
check("14 미래에셋증권 2013 결산월", mm[2013], 12)
ad = load("15_재무_추가")
check("15 삼성전자 2024 연구개발비", fin(ad, SAMSUNG, 2024, "연구개발비"), 34_998_142_000)
check("15 삼성전자 2024 차입금의존도", fin(ad, SAMSUNG, 2024, "차입금의존도"), 3.76, tol=0.005)

# 17: 어닝서프라이즈
es = load("17_어닝서프라이즈_2000_2004")
check("17 삼성전자 2000 1Q 매출액", fin(es, SAMSUNG, 2000, "매출액(최초", "1Q"), 7_873_303_753)
check("17 삼성전자 2001 1Q 매출액 서프라이즈", fin(es, SAMSUNG, 2001, "매출액(어닝", "1Q"), -1.84, tol=0.005)
check("17 삼성전자 2001 1Q 영업이익 서프라이즈", fin(es, SAMSUNG, 2001, "영업이익(어닝", "1Q"), 20.15, tol=0.005)
check("17 삼성전자 2000 4Q 실적보고서발표일", fin(es, SAMSUNG, 2000, "실적보고서발표일", "4Q"), 20010321)

# 22: 추정기관수
na = load("22_컨센_추정기관수")
s22 = na[na["종목코드"] == SAMSUNG]
check("22 삼성전자 첫 값 연월", s22["연월"].min(), "2000-09")
print(f"[참고] 22 삼성전자 최근 추정기관수: {s22.tail(3)[['연월', '추정기관수']].values.tolist()}")

# 30: 공매도잔고비율
sh = load("30_공매도_잔고")
s30 = sh[(sh["종목코드"] == SAMSUNG) & sh[col(sh, "잔고비율")].notna()]
check("30 삼성전자 공매도잔고비율 첫 연월", s30["연월"].iloc[0], "2016-06")
check("30 삼성전자 공매도잔고비율 첫 값", s30[col(sh, "잔고비율")].iloc[0], 0.05, tol=0.005)

# 31: 상장주식수
check("31 삼성전자 2026-10 상장주식수", val(load("31_공매도_거래"), SAMSUNG, "2026-10", "상장주식수"), 5_846_278_608)

# 32: 베타·변동성
bv = load("32_베타_변동성")
check("32 삼성전자 2026-10 베타", round(val(bv, SAMSUNG, "2026-10", "로그베타"), 2), 1.29)
check("32 삼성전자 2026-10 변동성", round(val(bv, SAMSUNG, "2026-10", "변동성"), 2), 0.77)
check("32 삼성전자 2015-01 변동성", round(val(bv, SAMSUNG, "2015-01", "변동성"), 2), 0.24)

# 33: 원주가(액면분할)
px = load("33_원주가")
for ym, exp in [("2018-03", 2_461_000), ("2018-04", 2_650_000), ("2018-05", 50_700)]:
    check(f"33 삼성전자 {ym} 원주가", val(px, SAMSUNG, ym, "종가"), exp)

# 36: 베타 보조
b2 = load("36_베타_보조")
check("36 삼성전자 2026-10 R²", val(b2, SAMSUNG, "2026-10", "R-Square"), 0.887504, tol=1e-6)
check("36 삼성전자 2026-10 36개월 베타", val(b2, SAMSUNG, "2026-10", "36개월"), 1.238692, tol=1e-6)

# 37~39: 잠정치발표일
pa = load("39_잠정치발표일_2020_2026")
check("39 삼성전자 2024 4Q 잠정치발표일", fin(pa, SAMSUNG, 2024, "잠정치발표일", "4Q"), 20250108)

print(f"\n결과: {sum(results)}/{len(results)} 통과")
