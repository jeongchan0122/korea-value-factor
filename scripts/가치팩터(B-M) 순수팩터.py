"""
가치 팩터(B/M) 순수 팩터 포트폴리오 (과제 2단계 '욕심 버전').

2×3 HML은 사이즈만 통제했다. 여기서는 매달 횡단면 가중회귀로 사이즈·베타·모멘텀·업종을 한꺼번에 통제해
'가치 노출 = 1, 다른 노출 = 0'인 포트폴리오의 수익률을 구한다.

  매달 t (구성 월):
    다음 달 수익률ᵢ = Σ 업종효과 × 업종더미ᵢ + f가치·가치ᵢ + f사이즈·사이즈ᵢ + f베타·베타ᵢ + f모멘텀·모멘텀ᵢ + 오차
    가중치 = √시가총액 (WLS, MSCI Barra 방식: 소형주가 결과를 휘두르는 걸 줄임)
  노출(점수)은 매달 1% winsorize → z-score
    가치   = B/M (장부가 ÷ 그 달 시가총액)            ← 노출행렬과 같은 정의
    사이즈 = log(시가총액)
    베타   = 52주 베타(일간)
    모멘텀 = t−11 ~ t−1월 누적수익률 (최근 1개월 제외, 12-1 모멘텀)
  업종 더미는 FnGuide Sector 10개(절편 대신 전부 넣음) → 스타일 팩터 포트폴리오는 업종마다 순투자 0

f가치 = (XᵀWX)⁻¹XᵀW r 의 가치 행 = 수익률의 가중합 → 그 자체가 하나의 포트폴리오.
그 비중 w는 w·가치 = 1, w·(사이즈·베타·모멘텀) = 0, w·업종더미 = 0 을 만족한다(코드에서 매달 확인).

실행: python "scripts/가치팩터(B-M) 순수팩터.py"   →  결과/가치팩터(B-M)/순수팩터/
(HML과 비교하려면 "가치팩터(B-M) 재현.py"를 먼저 실행해 월별수익률.csv가 있어야 함)
"""
import numpy as np
import pandas as pd

from 팩터공통 import (ME, RESULTS, RET, attach_annual, load_tables, performance, setup_korean_font, shift_ym,
                  universe_mask, winsorize, zscore)

# ---------------------------------------------------------------- 설정
첫구성연월 = "2000-06"     # 이 달 말에 처음 구성 → 첫 수익률은 2000-07
끝연월 = "2026-09"         # 마지막 수익률 월
금융업제외 = False
스타일 = ["가치", "사이즈", "베타", "모멘텀"]

OUT = RESULTS / "가치팩터(B-M)" / "순수팩터"
HML_CSV = RESULTS / "가치팩터(B-M)" / "월별수익률.csv"
BE, BETA, SECTOR = "자본총계_지배_천원", "베타_52주_일간", "업종_FnGuide"
RAW_RET = "수정주가수익률_배당포함_%"


def momentum_12_1(panel):
    """t−11 ~ t−1월 누적수익률(%). 11개월이 모두 있어야 계산."""
    w = panel.pivot(index="연월", columns="종목코드", values=RAW_RET).sort_index()
    logr = np.log1p(w / 100)
    mom = np.expm1(logr.shift(1).rolling(11, min_periods=11).sum()) * 100
    return mom.stack().rename("모멘텀원값").reset_index()


def build_exposures(panel, annual):
    """매달 분석 대상 종목의 노출(z-score), 업종, √시가총액 가중치, 다음 달 수익률."""
    m = universe_mask(panel, 금융업제외) & (panel["연월"] >= 첫구성연월) & (panel["연월"] < 끝연월)
    e = panel.loc[m, ["연월", "종목코드", ME, BETA, SECTOR]].reset_index(drop=True)
    e[BE] = attach_annual(e, annual, [BE])[BE].values
    e = e.merge(momentum_12_1(panel), on=["연월", "종목코드"], how="left")

    nxt = panel[["연월", "종목코드", RET]].dropna().copy()
    nxt["연월"] = shift_ym(nxt["연월"], -1)  # 다음 달 수익률을 이번 달 줄에 붙임
    e = e.merge(nxt.rename(columns={RET: "다음달수익률"}), on=["연월", "종목코드"], how="left")

    e = e[(e[BE] > 0)].dropna(subset=[BETA, SECTOR, "모멘텀원값", "다음달수익률"]).copy()
    raw = {"가치": e[BE] / (e[ME] * 1000), "사이즈": np.log(e[ME]), "베타": e[BETA], "모멘텀": e["모멘텀원값"]}
    for k, v in raw.items():
        e[k] = v.groupby(e["연월"]).transform(lambda s: zscore(winsorize(s)))
    e["가중치"] = np.sqrt(e[ME])
    return e[["연월", "종목코드", SECTOR, *스타일, "가중치", "다음달수익률"]]


def cross_section(g):
    """한 달치 가중회귀. 반환: 팩터 수익률, 가치 포트폴리오 비중 점검값."""
    dummies = pd.get_dummies(g[SECTOR], dtype=float)
    X = np.column_stack([dummies.values, g[스타일].values])
    W = g["가중치"].values
    XtW = X.T * W
    H = np.linalg.solve(XtW @ X, XtW)             # 팩터 포트폴리오 비중: 행마다 하나의 포트폴리오
    f = H @ g["다음달수익률"].values
    k = dummies.shape[1]
    w_val = H[k + 0]                                # 가치 포트폴리오 비중
    exposure = X.T @ w_val                          # 업종·스타일 노출(가치만 1이어야 함)
    out = {s: f[k + i] for i, s in enumerate(스타일)}
    out.update({"종목수": len(g), "가치노출": exposure[k], "다른노출_최대절대값": np.abs(np.delete(exposure, k)).max(),
                "가치포트_총비중합": np.abs(w_val).sum(), "가치포트_롱비중합": w_val[w_val > 0].sum()})
    return pd.Series(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = load_tables()
    panel, annual = t["월별패널"], t["연간재무"]

    e = build_exposures(panel, annual)
    res = e.groupby("연월").apply(cross_section, include_groups=False)
    res.index = shift_ym(pd.Series(res.index), 1)   # 수익률이 실현된 달(t+1)로 표시
    res.index.name = "연월"

    # 수학적으로 맞게 됐는지 확인: 가치 노출 1, 다른 노출 0
    assert np.allclose(res["가치노출"], 1, atol=1e-6) and (res["다른노출_최대절대값"] < 1e-6).all(), "노출 제약이 안 맞음"

    pure = res[스타일].add_prefix("순수_")
    report = pd.DataFrame({c: performance(pure[c]) for c in pure.columns}).T

    comp = None
    if HML_CSV.exists():
        hml = pd.read_csv(HML_CSV, encoding="utf-8-sig", index_col=0)
        both = pure.join(hml[["HML_2x3", "FnGuide_HML", "SMB_2x3"]], how="inner")
        comp = pd.DataFrame({
            "상관계수": [both["순수_가치"].corr(both["HML_2x3"]), both["순수_가치"].corr(both["FnGuide_HML"]),
                     both["순수_사이즈"].corr(both["SMB_2x3"])],
        }, index=["순수_가치 vs HML_2x3", "순수_가치 vs FnGuide_HML", "순수_사이즈 vs SMB_2x3"])
        report = pd.concat([report, pd.DataFrame({c: performance(both[c]) for c in ["HML_2x3", "FnGuide_HML"]}).T])
        make_charts(both)
    else:
        print(f"[주의] {HML_CSV} 가 없어 HML 비교를 건너뜀")

    pure.join(res[["종목수", "가치포트_총비중합", "가치포트_롱비중합"]]).to_csv(OUT / "순수팩터_월별수익률.csv",
                                                                     encoding="utf-8-sig")
    report.index.name = "시계열"
    report.to_csv(OUT / "순수팩터_성적표.csv", encoding="utf-8-sig")

    cols = ["개월수", "월평균_%", "월표준편차_%", "t통계량", "NW_t통계량", "연환산샤프", "최대낙폭_%"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(f"매달 회귀 종목 수: 평균 {res['종목수'].mean():.0f} (최소 {res['종목수'].min():.0f}, 최대 {res['종목수'].max():.0f})")
        print(f"노출 확인: 가치 노출 = 1, 다른 노출 최대 {res['다른노출_최대절대값'].max():.1e} (0이면 정상)")
        print(f"순수 가치 포트폴리오 비중: 롱 합 평균 {res['가치포트_롱비중합'].mean():.3f} (숏도 같은 크기, 업종마다 순투자 0)")
        print("\n[성적표]  순수 팩터 수익률 = 노출 1(=1 표준편차)짜리 포트폴리오의 월수익률")
        print(report[cols].to_string())
        if comp is not None:
            print("\n[상관계수]"); print(comp.to_string())
    print(f"\n결과 저장: {OUT}")


def make_charts(both):
    plt = setup_korean_font()

    # 1. 누적수익률: 순수 가치는 HML과 변동성을 맞춰서 비교(크기 단위가 달라서)
    scale = both["HML_2x3"].std() / both["순수_가치"].std()
    fig, ax = plt.subplots(figsize=(10, 5))
    for c, s, style in [("순수_가치 (HML과 변동성 맞춤)", both["순수_가치"] * scale, "-"),
                        ("HML_2x3", both["HML_2x3"], "--"), ("FnGuide_HML", both["FnGuide_HML"], ":")]:
        w = (1 + s / 100).cumprod()
        ax.plot(pd.to_datetime(w.index, format="%Y-%m"), w.values, style, label=c)
    ax.set_yscale("log"); ax.set_title("순수 가치 팩터 vs HML 누적수익률 (1원 투자, 로그 눈금)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "1_순수가치_vs_HML_누적.png", dpi=150); plt.close(fig)

    # 2. 순수 팩터 4개 평균 수익률과 t-통계량
    cols = [f"순수_{s}" for s in 스타일]
    mean = both[cols].mean()
    tstat = mean / (both[cols].std() / np.sqrt(len(both)))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(스타일, mean.values, color=["tab:blue" if v > 0 else "tab:red" for v in mean.values])
    for b, tv in zip(bars, tstat.values):
        ax.annotate(f"t={tv:.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom" if b.get_height() >= 0 else "top")
    ax.axhline(0, color="k", lw=.8)
    ax.set_ylabel("월평균 수익률 (%)"); ax.set_title("순수 팩터 월평균 수익률 (업종·다른 팩터 통제, 노출 1 기준)")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "2_순수팩터_평균수익률.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
