"""
가치 팩터(B/M) 재현 — Fama-French 2×3 방식으로 HML을 만들고 FnGuide 3FM.2B3.HML과 비교한다.

과제 단계
  1단계 노출 행렬   매달 종목별 B/M → winsorize(1%) → z-score  (결과/…/노출행렬.parquet)
  2단계 포트폴리오  (가) 2×3: 매년 6월 말 사이즈 2 × B/M 3 = 6바구니 → HML, SMB
                    (나) 단순 롱숏: B/M 상위 30% − 하위 30% (시가총액가중, 동일가중)
  3단계 백테스트    t월 말 구성 → t+1월 수익률(배당 포함, 상장폐지 수익률 반영), 성적표, FnGuide 비교
                    + 시장 벤치마크(분석 대상 전체 시가총액가중, 코스피TR):
                      HML의 시장 베타·알파(CAPM), 롱온리 가치(가치 바구니만 매수) vs 시장

B/M = 자본총계_지배(천원) ÷ (전년 12월 말 시가총액(백만원) × 1000)
  - 장부가는 6월 말에 '쓸 수 있는' 가장 최근 연간재무(사용가능연월 ≤ 6월). 12월 결산이면 전년도 재무
  - 시가총액을 전년 12월로 맞추는 건 FF 방식(장부가 시점과 맞추기 위해)

실행: python "scripts/가치팩터(B-M) 재현.py"   →  결과/가치팩터(B-M)/
"""
import numpy as np
import pandas as pd

from 팩터공통 import (ME, RESULTS, RET, attach_annual, capm, load_tables, market_return, performance,
                  portfolio_returns, setup_korean_font, shift_ym, universe_mask, winsorize, zscore)

# ---------------------------------------------------------------- 설정 (바꿔서 다른 버전을 돌려볼 수 있음)
첫구성연월 = "2000-06"     # 첫 6월(전년 12월 시가총액이 필요해서 1999-12가 있는 2000년부터)
끝연월 = "2026-09"         # 2026-10은 파일마다 기준일이 달라 제외
가중방식 = "시가총액"       # "시가총액" 또는 "동일"
사이즈기준시장 = "전체"     # 사이즈 중앙값·B/M 30/70% 기준선을 정할 종목: "전체" 또는 "KS"(코스피만)
금융업제외 = False         # FF 원래 방식은 포함
BM기준선 = (0.3, 0.7)
FNGUIDE_HML, FNGUIDE_SMB = "3FM.2B3.HML", "3FM.2B3.SMB"

OUT = RESULTS / "가치팩터(B-M)"
BE = "자본총계_지배_천원"


def form_portfolios(panel, annual):
    """매년 6월 말 바구니 구성. 반환: 종목별 구성 정보(구성연월, 종목코드, 사이즈, B/M, 바구니들)."""
    junes = [f"{y}-06" for y in range(int(첫구성연월[:4]), int(끝연월[:4]) + 1) if f"{y}-06" < 끝연월]
    uni = panel[universe_mask(panel, 금융업제외) & panel["연월"].isin(junes)][["연월", "종목코드", "시장구분", ME]]
    uni = uni.rename(columns={ME: "사이즈"}).reset_index(drop=True)

    uni[BE] = attach_annual(uni, annual, [BE])[BE].values
    dec = panel[["연월", "종목코드", ME]].rename(columns={ME: "12월시가총액"})
    dec["연월"] = shift_ym(dec["연월"], 6)  # 전년 12월 → 올해 6월 줄에 붙임
    uni = uni.merge(dec, on=["연월", "종목코드"], how="left")
    uni["BM"] = uni[BE] / (uni["12월시가총액"] * 1000)
    uni = uni[(uni[BE] > 0) & (uni["12월시가총액"] > 0)].copy()  # 자본잠식·12월 시가총액 없는 종목 제외

    parts = []
    for ym, g in uni.groupby("연월"):
        base = g if 사이즈기준시장 == "전체" else g[g["시장구분"] == "KS"]
        size_med = base["사이즈"].median()
        lo, hi = base["BM"].quantile(BM기준선[0]), base["BM"].quantile(BM기준선[1])
        g = g.copy()
        g["사이즈그룹"] = np.where(g["사이즈"] <= size_med, "S", "B")
        g["BM그룹"] = np.select([g["BM"] <= lo, g["BM"] >= hi], ["L", "H"], "M")
        g["바구니2x3"] = g["사이즈그룹"] + g["BM그룹"]
        g["BM10분위"] = pd.qcut(g["BM"], 10, labels=[f"D{i:02d}" for i in range(1, 11)]).astype(str)
        parts.append(g)
    return pd.concat(parts).rename(columns={"연월": "구성연월"})


def exposure_matrix(panel, annual):
    """1단계: 매달 종목별 B/M(그 달 시가총액 기준) → winsorize → z-score. 긴 표로 반환."""
    m = panel[universe_mask(panel, 금융업제외) & (panel["연월"] > 첫구성연월) & (panel["연월"] <= 끝연월)]
    e = m[["연월", "종목코드", ME]].reset_index(drop=True)
    e[BE] = attach_annual(e, annual, [BE])[BE].values
    e = e[e[BE] > 0].copy()
    e["BM"] = e[BE] / (e[ME] * 1000)
    e["BM_winsor"] = e.groupby("연월")["BM"].transform(winsorize)
    e["BM_z"] = e.groupby("연월")["BM_winsor"].transform(zscore)
    return e[["연월", "종목코드", "BM", "BM_winsor", "BM_z"]]


def turnover(form, col, groups):
    """바구니별 연간 교체율: 올해 그 바구니 종목 중 작년에는 그 바구니가 아니었던 비율(종목 수 기준)."""
    rates = {}
    for grp in groups:
        sets = form[form[col] == grp].groupby("구성연월")["종목코드"].apply(set).sort_index()
        r = [1 - len(sets.iloc[i] & sets.iloc[i - 1]) / len(sets.iloc[i]) for i in range(1, len(sets))]
        rates[grp] = np.mean(r)
    return pd.Series(rates)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = load_tables()
    panel, annual, fac, mkt_ts = t["월별패널"], t["연간재무"], t["FnGuide팩터지수"], t["시장시계열"].set_index("연월")

    # 1단계
    expo = exposure_matrix(panel, annual)
    expo.to_parquet(OUT / "노출행렬.parquet", index=False)

    # 2단계
    form = form_portfolios(panel, annual)
    form.to_csv(OUT / "바구니구성.csv", index=False, encoding="utf-8-sig")

    def rets(col, weighting=가중방식):
        h = form[["구성연월", "종목코드", col]].rename(columns={col: "바구니"})
        r = portfolio_returns(h, panel, weighting)
        return r[r.index <= 끝연월]

    p6 = rets("바구니2x3")
    p3_vw, p3_ew = rets("BM그룹", "시가총액"), rets("BM그룹", "동일")
    p10 = rets("BM10분위")

    monthly = pd.DataFrame({
        "HML_2x3": (p6["SH"] + p6["BH"]) / 2 - (p6["SL"] + p6["BL"]) / 2,
        "SMB_2x3": p6[["SL", "SM", "SH"]].mean(axis=1) - p6[["BL", "BM", "BH"]].mean(axis=1),
        "HML_단순롱숏_시총가중": p3_vw["H"] - p3_vw["L"],
        "HML_단순롱숏_동일가중": p3_ew["H"] - p3_ew["L"],
    })
    fg = fac.pivot(index="연월", columns="코드", values="수익률_%")
    monthly["FnGuide_HML"] = fg[FNGUIDE_HML].reindex(monthly.index)
    monthly["FnGuide_SMB"] = fg[FNGUIDE_SMB].reindex(monthly.index)
    # 롱온리 가치(공매도 없이 가치 바구니만 매수)와 시장 벤치마크
    monthly["롱온리_가치_2x3"] = (p6["SH"] + p6["BH"]) / 2
    monthly["롱온리_가치_시총가중"] = p3_vw["H"]
    monthly["시장_분석대상전체"] = market_return(panel, 금융업제외).reindex(monthly.index)
    monthly["코스피TR"] = mkt_ts["코스피TR_수익률_%"].reindex(monthly.index)
    monthly["무위험"] = mkt_ts["무위험수익률_월_%"].reindex(monthly.index)
    monthly = monthly.join(p6.add_prefix("바구니_"))
    monthly.index.name = "연월"
    monthly.to_csv(OUT / "월별수익률.csv", encoding="utf-8-sig")

    # 3단계: 성적표 (롱온리·시장은 샤프지수를 무위험 초과수익률로 계산)
    long_only = ["롱온리_가치_2x3", "롱온리_가치_시총가중", "시장_분석대상전체", "코스피TR"] + \
                [c for c in monthly.columns if c.startswith("바구니_")]
    report = pd.DataFrame({c: performance(monthly[c], monthly["무위험"] if c in long_only else None)
                           for c in monthly.columns if c != "무위험"}).T
    report.index.name = "시계열"
    report.to_csv(OUT / "성적표.csv", encoding="utf-8-sig")

    # 시장 벤치마크 비교
    mkt_ex = monthly["시장_분석대상전체"] - monthly["무위험"]
    bench = []
    for c in ["HML_2x3", "FnGuide_HML", "HML_단순롱숏_시총가중"]:  # 롱숏: 수익률 그대로 시장초과수익률에 회귀
        bench.append(capm(monthly[c], mkt_ex).rename(c))
    for c in ["롱온리_가치_2x3", "롱온리_가치_시총가중", "코스피TR"]:  # 롱온리: 무위험 초과수익률로 회귀
        s = capm(monthly[c] - monthly["무위험"], mkt_ex)
        diff = (monthly[c] - monthly["시장_분석대상전체"]).dropna()
        s["시장대비_초과_월%"] = diff.mean()
        s["추적오차_연%"] = diff.std() * np.sqrt(12)
        s["정보비율"] = diff.mean() / diff.std() * np.sqrt(12)
        bench.append(s.rename(c))
    bench = pd.DataFrame(bench)
    bench.index.name = "시계열"
    bench.to_csv(OUT / "시장벤치마크비교.csv", encoding="utf-8-sig")

    # FnGuide 비교
    comp = []
    for mine, theirs in [("HML_2x3", "FnGuide_HML"), ("HML_단순롱숏_시총가중", "FnGuide_HML"),
                         ("HML_단순롱숏_동일가중", "FnGuide_HML"), ("SMB_2x3", "FnGuide_SMB")]:
        d = monthly[[mine, theirs]].dropna()
        slope, intercept = np.polyfit(d[mine], d[theirs], 1)
        comp.append({"내 시계열": mine, "비교": theirs, "개월수": len(d), "상관계수": d.corr().iloc[0, 1],
                     "내 월평균_%": d[mine].mean(), "FnGuide 월평균_%": d[theirs].mean(),
                     "회귀기울기(FnGuide~내것)": slope, "추적오차_월%": (d[mine] - d[theirs]).std()})
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT / "FnGuide비교.csv", index=False, encoding="utf-8-sig")

    to = turnover(form, "바구니2x3", ["SL", "SH", "BL", "BH"])

    make_charts(monthly, p6, p10)

    # 요약 출력
    n = form.groupby("구성연월").size()
    print(f"구성: {form['구성연월'].min()} ~ {form['구성연월'].max()} 매년 6월, 바구니당 평균 종목 "
          f"{form.groupby(['구성연월', '바구니2x3']).size().mean():.0f}개 (해마다 {n.min()}~{n.max()}개)")
    print(f"노출행렬: {len(expo):,}줄, 월평균 {expo.groupby('연월').size().mean():.0f}종목")
    cols = ["개월수", "월평균_%", "월표준편차_%", "t통계량", "NW_t통계량", "연환산샤프", "최대낙폭_%"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        show = ["HML_2x3", "SMB_2x3", "HML_단순롱숏_시총가중", "HML_단순롱숏_동일가중", "FnGuide_HML", "FnGuide_SMB",
                "롱온리_가치_2x3", "롱온리_가치_시총가중", "시장_분석대상전체", "코스피TR"]
        print("\n[성적표] (롱온리·시장의 샤프는 무위험 초과수익률 기준)"); print(report.loc[show, cols].to_string())
        print("\n[FnGuide 비교]"); print(comp.to_string(index=False))
        print("\n[시장 벤치마크 비교] (시장 = 분석 대상 전체 시가총액가중)"); print(bench.to_string())
        print("\n[6바구니 월평균 %]"); print(p6.mean().round(2).to_string())
        print("\n[연간 교체율]"); print(to.round(2).to_string())
    print(f"\n결과 저장: {OUT}")


def make_charts(monthly, p6, p10):
    plt = setup_korean_font()

    # 1. 누적수익률
    fig, ax = plt.subplots(figsize=(10, 5))
    for c, style in [("HML_2x3", "-"), ("FnGuide_HML", "--"), ("HML_단순롱숏_시총가중", ":"),
                     ("시장_분석대상전체", "-")]:
        w = (1 + monthly[c].dropna() / 100).cumprod()
        kw = {"color": "gray", "alpha": .7} if c.startswith("시장") else {}
        ax.plot(pd.to_datetime(w.index, format="%Y-%m"), w.values, style, label=c + (" (참고)" if c.startswith("시장") else ""), **kw)
    ax.set_yscale("log"); ax.set_title("가치 팩터(롱숏) 누적수익률 (1원 투자, 로그 눈금)"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "1_누적수익률.png", dpi=150); plt.close(fig)

    # 6. 롱온리 가치 vs 시장
    fig, ax = plt.subplots(figsize=(10, 5))
    mkt_w = (1 + monthly["시장_분석대상전체"].dropna() / 100).cumprod()
    for c, style in [("롱온리_가치_2x3", "-"), ("롱온리_가치_시총가중", "--"), ("시장_분석대상전체", "-"), ("코스피TR", ":")]:
        w = (1 + monthly[c].dropna() / 100).cumprod()
        label, kw = c, ({"color": "gray"} if c == "시장_분석대상전체" else {})
        if c == "코스피TR":  # 2011년부터만 있어서 시작점을 그 전달 시장 선 높이에 맞춤
            w = w * mkt_w.shift(1).get(w.index[0], 1.0)
            label = f"코스피TR ({w.index[0][:4]}년부터, 시작점을 시장 선에 맞춤)"
        ax.plot(pd.to_datetime(w.index, format="%Y-%m"), w.values, style, label=label, **kw)
    ax.set_yscale("log"); ax.set_title("롱온리 가치 포트폴리오 vs 시장 (1원 투자, 로그 눈금)"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "6_롱온리가치_vs_시장.png", dpi=150); plt.close(fig)

    # 2. 월수익률 산점도
    d = monthly[["HML_2x3", "FnGuide_HML"]].dropna()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(d["HML_2x3"], d["FnGuide_HML"], s=10, alpha=.6)
    lim = [min(d.min()), max(d.max())]
    ax.plot(lim, lim, "k--", lw=1, label="45도선(완전 일치)")
    ax.set_xlabel("내 HML 월수익률 (%)"); ax.set_ylabel("FnGuide HML 월수익률 (%)")
    ax.set_title(f"월수익률 비교 (상관계수 {d.corr().iloc[0, 1]:.2f})"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "2_월수익률_산점도.png", dpi=150); plt.close(fig)

    # 3. 6바구니 평균수익률
    m = p6.mean()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(3)
    ax.bar(x - .2, [m["SL"], m["SM"], m["SH"]], .4, label="소형(S)")
    ax.bar(x + .2, [m["BL"], m["BM"], m["BH"]], .4, label="대형(B)")
    ax.set_xticks(x, ["Low B/M(성장)", "Medium", "High B/M(가치)"])
    ax.set_ylabel("월평균 수익률 (%)"); ax.set_title("2×3 바구니 월평균 수익률"); ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "3_6바구니_평균수익률.png", dpi=150); plt.close(fig)

    # 4. 연도별 HML
    yr = monthly[["HML_2x3", "FnGuide_HML"]].copy()
    yr["연도"] = yr.index.str[:4]
    yearly = yr.groupby("연도").apply(lambda g: ((1 + g[["HML_2x3", "FnGuide_HML"]] / 100).prod() - 1) * 100,
                                      include_groups=False)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(yearly))
    ax.bar(x - .2, yearly["HML_2x3"], .4, label="내 HML")
    ax.bar(x + .2, yearly["FnGuide_HML"], .4, label="FnGuide HML")
    ax.set_xticks(x, yearly.index, rotation=90); ax.axhline(0, color="k", lw=.8)
    ax.set_ylabel("연간 수익률 (%)"); ax.set_title("연도별 HML 수익률 (2000년은 7~12월)"); ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "4_연도별_HML.png", dpi=150); plt.close(fig)

    # 5. B/M 10분위
    m10 = p10.mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(m10.index, m10.values)
    ax.set_xlabel("B/M 10분위 (D01 = 가장 비쌈, D10 = 가장 쌈)"); ax.set_ylabel("월평균 수익률 (%)")
    ax.set_title(f"B/M 10분위 월평균 수익률 ({가중방식}가중)"); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "5_BM_10분위.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
