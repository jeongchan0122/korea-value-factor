"""
4단계 검증: processed/ 의 합친 표 5개가 tidy 원본·검증값과 맞는지, 날짜 규칙이 말이 되는지 확인한다.

실행: python scripts/05_verify_merge.py
"""
import pandas as pd

from korea_parsers import ROOT

OUT = ROOT / "processed"
SAMSUNG, MIRAE = "A005930", "A006800"
results = []


def check(label, actual, expected, tol=0):
    if isinstance(expected, (int, float)) and not isinstance(expected, bool) and pd.notna(actual):
        ok = abs(actual - expected) <= tol
    else:
        ok = actual == expected
    results.append(bool(ok))
    print(f"[{'OK' if ok else '틀림'}] {label}: 실제={actual!r}  기대={expected!r}")


def load(name):
    return pd.read_parquet(OUT / f"{name}.parquet")


# ---- 월별패널
p = load("월별패널")
check("월별패널 (연월, 종목코드) 중복 없음", int(p.duplicated(["연월", "종목코드"]).sum()), 0)
# 1999년 이전 상폐 종목(약 180개)은 값이 없어 빠지는 게 정상
listing = pd.read_parquet(OUT / "tidy" / "29_상장일_상폐일.parquet")
alive = listing[~(listing["상장폐지일_CP10001600"] < "1999-01-01")]
print(f"[참고] 월별패널 종목 {p['종목코드'].nunique()}개 / 1999년 이후에도 상장돼 있던 종목 {len(alive)}개")
check("1999년 이후 상장 종목 대부분 포함(누락 ≤ 20)", len(set(alive["종목코드"]) - set(p["종목코드"])) <= 20, True)
s = p[(p["종목코드"] == SAMSUNG) & (p["연월"] == "2015-01")].iloc[0]
check("삼성전자 2015-01 수익률", s["수정주가수익률_%"], 2.86)
check("삼성전자 2015-01 시가총액", s["시가총액_백만원"], 201063595)
check("삼성전자 2015-01 투자의견 최고", s["컨센_투자의견_최고"], 5)
check("삼성전자 2015-01 상장폐지일 없음", pd.isna(s["상장폐지일"]), True)
s = p[(p["종목코드"] == SAMSUNG) & (p["연월"] == "2026-09")].iloc[0]
check("삼성전자 2026-09 거래대금", s["거래대금_월합계_원"], 89_569_507_007_221)
print(f"[참고] 시장구분별 행 수: {p['시장구분'].value_counts(dropna=False).to_dict()}")
print(f"[참고] KS/KQ인데 수익률 없는 행: {int((p['시장구분'].isin(['KS', 'KQ']) & p['수정주가수익률_%'].isna()).sum())}")

# 상폐수익률 반영(02b): 상폐한 달 줄에는 기존 수익률이 없어야 하고, 1999년 이후 상폐 대부분이 들어가야 함
check("상폐월 줄 수(1999년 이후 상폐 대부분)", int(p["상폐월"].sum()) >= 1200, True)
check("상폐월인데 기존 수익률도 있는 줄 없음", int((p["상폐월"] & p["수정주가수익률_배당포함_%"].notna()).sum()), 0)
s = p[(p["종목코드"] == "A043090") & (p["연월"] == "2026-08")].iloc[0]
check("더테크놀로지 2026-08 상폐수익률(7월 말 → 8월 정리매매 마지막 날)",
      round(s["수정주가수익률_배당포함_상폐반영_%"], 1), -62.5)

# 투자의견 방향: 목표주가 괴리율(상승여력)이 클수록 점수가 높으면 '높을수록 매수'
c = p[["컨센_투자의견_평균", "컨센_목표주가괴리율_%"]].dropna()
print(f"[참고] 투자의견 평균 범위 {c.iloc[:, 0].min()}~{c.iloc[:, 0].max()}, 평균 {c.iloc[:, 0].mean():.2f}, "
      f"목표주가괴리율과 상관 {c.corr(method='spearman').iloc[0, 1]:.2f}")

# ---- 연간재무
a = load("연간재무")
check("연간재무 (종목코드, 회계연도) 중복 없음", int(a.duplicated(["종목코드", "회계연도"]).sum()), 0)
r = a[(a["종목코드"] == SAMSUNG) & (a["회계연도"] == 2025)].iloc[0]
check("삼성전자 2025 자산총계", r["자산총계_천원"], 566_942_110_000)
r = a[(a["종목코드"] == SAMSUNG) & (a["회계연도"] == 2024)].iloc[0]
check("삼성전자 2024 회계연도말", r["회계연도말"], "2024-12")
check("삼성전자 2024 사용가능연월", r["사용가능연월"], "2025-06")
r = a[(a["종목코드"] == MIRAE) & (a["회계연도"] == 2011)].iloc[0]
check("미래에셋증권 2011(3월 결산) 회계연도말", r["회계연도말"], "2012-03")
check("미래에셋증권 2011 사용가능연월", r["사용가능연월"], "2012-09")
r = a[(a["종목코드"] == MIRAE) & (a["회계연도"] == 2013)].iloc[0]
check("미래에셋증권 2013(12월로 변경) 회계연도말", r["회계연도말"], "2013-12")

# ---- 분기실적
q = load("분기실적")
check("분기실적 (종목코드, 회계연도, 분기) 중복 없음", int(q.duplicated(["종목코드", "회계연도", "분기"]).sum()), 0)
r = q[(q["종목코드"] == SAMSUNG) & (q["회계연도"] == 2024) & (q["분기"] == "4Q")].iloc[0]
check("삼성전자 2024 4Q 분기말", str(r["분기말"].date()), "2024-12-31")
check("삼성전자 2024 4Q 사용가능일", str(r["사용가능일"].date()), "2025-01-08")
r = q[(q["종목코드"] == SAMSUNG) & (q["회계연도"] == 2001) & (q["분기"] == "1Q")].iloc[0]
check("삼성전자 2001 1Q 영업이익 서프라이즈", r["영업이익_서프라이즈_%"], 20.15, tol=0.005)
r = q[(q["종목코드"] == MIRAE) & (q["회계연도"] == 2011) & (q["분기"] == "4Q")].iloc[0]
check("미래에셋증권 2011 4Q 분기말", str(r["분기말"].date()), "2012-03-31")
r = q[(q["종목코드"] == "A050860") & (q["회계연도"] == 2019) & (q["분기"] == "2Q")].iloc[0]
check("아세아텍(6월 결산) 2019 2Q 분기말", str(r["분기말"].date()), "2018-12-31")
# 회계연도 규칙이 맞다면 공시일은 분기말 이후 몇 달 안이어야 한다
gap = (q["실적보고서발표일"] - q["분기말"]).dt.days.dropna()
print(f"[참고] 실적보고서발표일 − 분기말(일): 중앙값 {gap.median():.0f}, 0~150일 비율 {gap.between(0, 150).mean():.3f}, "
      f"음수(분기 끝나기 전 공시) {int((gap < 0).sum())}건")
check("사용가능일이 분기말 이전인 줄 없음", int((q["사용가능일"] <= q["분기말"]).sum()), 0)
print(f"[참고] 사용가능일 출처: {q['사용가능일_출처'].value_counts().to_dict()}")

# ---- 시장시계열·팩터지수
m = load("시장시계열")
check("시장시계열 행 수", len(m), 334)
r = m[m["연월"] == "2026-10"].iloc[0]
check("2026-10 CD91", r["CD91_연%"], 3.27)
print(f"[참고] 코스피 월수익률 평균 {m['코스피_수익률_%'].mean():.2f}%, 무위험 월 평균 {m['무위험수익률_월_%'].mean():.3f}%")
f = load("FnGuide팩터지수")
check("FnGuide팩터지수 행 수(70개 × 334개월)", len(f), 70 * 334)
hml = f[f["코드"] == "3FM.2B3.HML"]["수익률_%"]
print(f"[참고] HML 월수익률 평균 {hml.mean():.2f}%, 표준편차 {hml.std():.2f}%")

# ---- 설명표
d = pd.read_csv(OUT / "열이름설명.csv", encoding="utf-8-sig")
for name in ["월별패널", "연간재무", "분기실적", "시장시계열", "FnGuide팩터지수"]:
    missing = set(load(name).columns) - set(d.loc[d["표"] == name, "열이름"])
    check(f"열이름설명에 {name} 열 모두 있음", sorted(missing), [])

print(f"\n결과: {sum(results)}/{len(results)} 통과")
