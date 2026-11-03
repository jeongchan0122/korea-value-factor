"""
가치 팩터를 B/M 말고 다른 가치 지표로도 만들어 비교한다 (+ 6개를 합친 종합 가치 점수).

모두 B/M 재현과 똑같은 2×3 틀: 매년 6월 말 사이즈 2(중앙값) × 지표 3(30/70%) → 시가총액가중 → HML
  - 분자(재무)는 6월 말에 쓸 수 있는 가장 최근 연간재무(사용가능연월 ≤ 6월), 분모는 전년 12월 시가총액(FF 방식)
  - 적자·마이너스(E/P, CF/P, EBITDA/EV)와 무배당(D/P = 0)은 그 지표의 바구니 나누기에서 뺌(FF 관행)
    → 지표마다 들어가는 종목 수가 다름. 사이즈 중앙값·기준선도 그 지표에 들어간 종목들로 정함
  - 종합 가치: 종목마다 값이 있는 지표들의 백분위 순위를 평균(최소 3개 지표 있어야 포함)

지표 (모두 '클수록 싸다' 방향)
  B/M       자본총계_지배 ÷ 12월 시가총액
  E/P       당기순이익_지배 ÷ 12월 시가총액
  CF/P      영업현금흐름 ÷ 12월 시가총액
  S/P       매출액 ÷ 12월 시가총액
  D/P       현금배당액_보통주 ÷ 12월 시가총액
  EBITDA/EV (영업이익 + 감가상각비) ÷ (12월 시가총액 + 차입금 − 현금)
            무형자산상각비가 데이터에 없어 EBITDA는 근사. 차입금 = 단기차입금 + 유동성장기부채 + 사채 + 장기차입금(빈칸 = 0),
            현금 = 현금및현금성자산 + 단기금융자산(빈칸 = 0)

실행: python "scripts/가치팩터(여러지표) 비교.py"   →  결과/가치팩터(여러지표)/
"""
import numpy as np
import pandas as pd

from 팩터공통 import ME, RESULTS, attach_annual, load_tables, performance, portfolio_returns, setup_korean_font, \
    shift_ym, universe_mask

첫구성연월, 끝연월 = "2000-06", "2026-09"
기준선 = (0.3, 0.7)
최소지표수 = 3
OUT = RESULTS / "가치팩터(여러지표)"

차입금 = ["단기차입금_천원", "유동성장기부채_천원", "사채_천원", "장기차입금_천원"]
현금 = ["현금및현금성자산_천원", "단기금융자산_천원"]
재무열 = ["자본총계_지배_천원", "당기순이익_지배_천원", "영업현금흐름_천원", "매출액_천원", "현금배당액_보통주_천원",
       "영업이익_천원", "감가상각비_천원", *차입금, *현금]
지표 = ["B/M", "E/P", "CF/P", "S/P", "D/P", "EBITDA/EV"]


def june_universe(panel, annual):
    """매년 6월 말 분석 대상 + 재무 + 전년 12월 시가총액 → 지표 6개(쓸 수 없는 값은 빈칸)."""
    junes = [f"{y}-06" for y in range(int(첫구성연월[:4]), int(끝연월[:4]) + 1) if f"{y}-06" < 끝연월]
    u = panel[universe_mask(panel) & panel["연월"].isin(junes)][["연월", "종목코드", ME]]
    u = u.rename(columns={ME: "사이즈"}).reset_index(drop=True)
    u = pd.concat([u, attach_annual(u, annual, 재무열)[재무열].reset_index(drop=True)], axis=1)
    dec = panel[["연월", "종목코드", ME]].rename(columns={ME: "12월시가총액"})
    dec["연월"] = shift_ym(dec["연월"], 6)
    u = u.merge(dec, on=["연월", "종목코드"], how="left")
    u = u[(u["12월시가총액"] > 0) & u["자본총계_지배_천원"].notna()].copy()  # 재무가 아예 없는 종목 제외

    me = u["12월시가총액"] * 1000  # 백만원 → 천원
    pos = lambda x: x.where(x > 0)  # 적자·마이너스·0은 빈칸(바구니 나누기에서 빠짐)
    u["B/M"] = pos(u["자본총계_지배_천원"]) / me
    u["E/P"] = pos(u["당기순이익_지배_천원"]) / me
    u["CF/P"] = pos(u["영업현금흐름_천원"]) / me
    u["S/P"] = pos(u["매출액_천원"]) / me
    u["D/P"] = pos(u["현금배당액_보통주_천원"]) / me
    ev = me + u[차입금].fillna(0).sum(axis=1) - u[현금].fillna(0).sum(axis=1)
    u["EBITDA/EV"] = pos(u["영업이익_천원"] + u["감가상각비_천원"].fillna(0)) / ev.where(ev > 0)

    # 종합 가치: 그해 값이 있는 종목 안에서의 백분위 순위를 평균
    ranks = u.groupby("연월")[지표].rank(pct=True)
    n = ranks.notna().sum(axis=1)
    u["종합"] = ranks.mean(axis=1).where(n >= 최소지표수)
    return u


def form(u, col):
    """한 지표로 2×3 바구니 구성(값이 있는 종목만, 기준선도 그 종목들로)."""
    parts = []
    for ym, g in u[u[col].notna()].groupby("연월"):
        med = g["사이즈"].median()
        lo, hi = g[col].quantile(기준선[0]), g[col].quantile(기준선[1])
        g = g[["연월", "종목코드"]].assign(
            바구니=np.where(g["사이즈"] <= med, "S", "B") + np.select([g[col] <= lo, g[col] >= hi], ["L", "H"], "M"))
        parts.append(g)
    return pd.concat(parts).rename(columns={"연월": "구성연월"})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = load_tables()
    panel, annual = t["월별패널"], t["연간재무"]
    u = june_universe(panel, annual)

    hml, 종목수 = {}, {}
    for col in 지표 + ["종합"]:
        h = form(u, col)
        p = portfolio_returns(h, panel, "시가총액")
        p = p[p.index <= 끝연월]
        name = "종합 가치" if col == "종합" else col
        hml[name] = (p["SH"] + p["BH"]) / 2 - (p["SL"] + p["BL"]) / 2
        종목수[name] = h.groupby("구성연월").size().mean()
    monthly = pd.DataFrame(hml).dropna()
    monthly.to_csv(OUT / "월별수익률.csv", encoding="utf-8-sig")

    report = pd.DataFrame({c: performance(monthly[c]) for c in monthly.columns}).T
    report = report[["개월수", "월평균_%", "월표준편차_%", "t통계량", "연환산샤프", "최대낙폭_%"]]
    report["BM과_상관"] = monthly.corr()["B/M"]
    report["평균_종목수"] = pd.Series(종목수)
    report.index.name = "지표"
    report.to_csv(OUT / "성적표.csv", encoding="utf-8-sig")
    monthly.corr().to_csv(OUT / "상관행렬.csv", encoding="utf-8-sig")

    make_chart(report)
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(report.to_string())
        print("\n[상관행렬]"); print(monthly.corr().to_string())
    print(f"\n결과 저장: {OUT}")


def make_chart(report):
    plt = setup_korean_font()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["#004C86" if i == "B/M" else ("#C00000" if i == "종합 가치" else "#8EA9DB") for i in report.index]
    for ax, col, title in [(axes[0], "월평균_%", "월평균 수익률 (%)"), (axes[1], "연환산샤프", "연환산 샤프")]:
        ax.bar(report.index, report[col], color=colors)
        for i, v in enumerate(report[col]):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title); ax.grid(axis="y", alpha=.3)
    fig.suptitle("가치 지표별 2×3 HML (2000.7~2026.9)")
    fig.tight_layout(); fig.savefig(OUT / "1_지표별_비교.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
