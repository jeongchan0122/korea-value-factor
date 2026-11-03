"""
가치 팩터 욕심 버전 (나): 지수(벤치마크)를 따라가면서 가치 노출만 늘리기 — 롱온리 '기울이기(tilt)'.

실제 스마트 베타 펀드 방식. 공매도 없이, 벤치마크 비중을 조금씩만 바꿔서
추적오차(벤치마크와 수익률 차이의 변동성)는 작게, 가치 노출은 크게 만든다.

  벤치마크 = 매년 6월 말 분석 대상(2×3과 같은 종목)을 시가총액가중으로 보유
  기울인 포트폴리오 비중 ∝ 시가총액 × max(0, 1 + k × 가치점수)
    - 가치점수 = 6월 말 B/M(전년 12월 시가총액 기준)의 1% winsorize → z-score (±3에서 자름)
    - k = 기울이는 강도. 0이면 벤치마크 그대로, 클수록 싼 종목 비중↑ 비싼 종목 비중↓(0 아래로는 안 감 = 롱온리)
  리밸런싱: 매년 6월 말(2×3과 같음), 보유 중 비중은 주가 따라 변함(전달 말 시가총액 × 승수)

결과: k마다 추적오차, 벤치마크 대비 초과수익, 정보비율, 가치 노출(포트폴리오 가중평균 가치점수 − 벤치마크)

실행: python "scripts/가치팩터(B-M) 벤치마크기울이기.py"   →  결과/가치팩터(B-M)/벤치마크기울이기/
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from 팩터공통 import RESULTS, load_tables, newey_west_t, portfolio_returns, setup_korean_font, winsorize, zscore

강도목록 = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
대표추적오차 = 3.0  # 연 %, 발표에서 대표로 보여줄 수준
OUT = RESULTS / "가치팩터(B-M)" / "벤치마크기울이기"
SCRIPTS = Path(__file__).resolve().parent


def load_value_module():
    spec = importlib.util.spec_from_file_location("vf", SCRIPTS / "가치팩터(B-M) 재현.py")
    vf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vf)
    return vf


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    vf = load_value_module()
    t = load_tables()
    panel, annual, mkt = t["월별패널"], t["연간재무"], t["시장시계열"].set_index("연월")

    form = vf.form_portfolios(panel, annual)[["구성연월", "종목코드", "사이즈", "BM"]].copy()
    form["가치점수"] = form.groupby("구성연월")["BM"].transform(lambda s: zscore(winsorize(s))).clip(-3, 3)

    rets, rows = {}, []
    for k in 강도목록:
        h = form.assign(바구니="P", 승수=(1 + k * form["가치점수"]).clip(lower=0))
        r = portfolio_returns(h[["구성연월", "종목코드", "바구니", "승수"]], panel, "시가총액")["P"]
        rets[k] = r[r.index <= vf.끝연월]
        # 구성 시점 가치 노출·종목 수·액티브 쉐어(벤치마크와 비중이 다른 정도)
        w = h["사이즈"] * h["승수"]
        h["w"] = w / w.groupby(h["구성연월"]).transform("sum")
        h["wb"] = h["사이즈"] / h["사이즈"].groupby(h["구성연월"]).transform("sum")
        g = h.groupby("구성연월")
        rows.append({"강도k": k,
                     "가치노출_차이": g.apply(lambda x: (x["w"] * x["가치점수"]).sum() - (x["wb"] * x["가치점수"]).sum(),
                                        include_groups=False).mean(),
                     "보유종목비율": g.apply(lambda x: (x["승수"] > 0).mean(), include_groups=False).mean(),
                     "액티브쉐어_%": g.apply(lambda x: 0.5 * (x["w"] - x["wb"]).abs().sum(), include_groups=False).mean() * 100})

    rets = pd.DataFrame(rets)
    bench = rets[0.0]
    rf = mkt["무위험수익률_월_%"].reindex(rets.index)
    for row in rows:
        k = row["강도k"]
        active = rets[k] - bench
        ex = rets[k] - rf
        row.update({"월평균_%": rets[k].mean(), "연환산샤프": ex.mean() / ex.std() * np.sqrt(12),
                    "초과수익_연%": active.mean() * 12, "추적오차_연%": active.std() * np.sqrt(12),
                    "정보비율": active.mean() / active.std() * np.sqrt(12) if k else np.nan,
                    "초과수익_NW_t": newey_west_t(active) if k else np.nan})
    table = pd.DataFrame(rows).set_index("강도k")
    table.to_csv(OUT / "강도별_결과.csv", encoding="utf-8-sig")
    rets.columns = [f"k={k:g}" for k in rets.columns]
    rets.to_csv(OUT / "월별수익률.csv", encoding="utf-8-sig")

    # 대표: 추적오차가 연 3%에 가장 가까운 강도
    rep_k = (table.loc[table.index > 0, "추적오차_연%"] - 대표추적오차).abs().idxmin()
    make_charts(table, rets, rep_k)

    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(f"벤치마크와 코스피TR 상관: {bench.corr(mkt['코스피TR_수익률_%'].reindex(bench.index)):.3f} (2011년~)")
        print("\n[강도별 결과]  (초과수익·추적오차는 벤치마크 대비, 연 단위)")
        print(table[["추적오차_연%", "초과수익_연%", "정보비율", "초과수익_NW_t", "가치노출_차이", "액티브쉐어_%",
                     "보유종목비율", "연환산샤프"]].to_string())
        print(f"\n대표(추적오차 ≈ 연 {대표추적오차}%): k = {rep_k:g}")
    print(f"\n결과 저장: {OUT}")


def make_charts(table, rets, rep_k):
    plt = setup_korean_font()
    tb = table[table.index > 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tb["추적오차_연%"], tb["초과수익_연%"], "o-", color="#004C86")
    for k, r in tb.iterrows():
        ax.annotate(f"k={k:g}\nIR {r['정보비율']:.2f}", (r["추적오차_연%"], r["초과수익_연%"]),
                    textcoords="offset points", xytext=(6, -14), fontsize=9)
    ax.set_xlabel("추적오차 (연 %)"); ax.set_ylabel("벤치마크 대비 초과수익 (연 %)")
    ax.set_title("가치 기울이기: 추적오차를 얼마나 감수하면 얼마나 더 버나"); ax.grid(alpha=.3)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(OUT / "1_추적오차_vs_초과수익.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for k in [0.0, rep_k, max(강도목록)]:
        w = (1 + rets[f"k={k:g}"] / 100).cumprod()
        ax.plot(pd.to_datetime(w.index, format="%Y-%m"), w.values,
                label="벤치마크" if k == 0 else f"기울이기 k={k:g}", color="gray" if k == 0 else None)
    ax.set_yscale("log"); ax.set_title("벤치마크 vs 가치 기울이기 누적수익률 (로그 눈금)"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "2_누적수익률.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
