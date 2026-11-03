"""
무형자산 조정 B/M 민감도 분석: 미국 연구에서 가져온 가정(감가율·조직자본 비율)을 바꿔도
"무형자산을 반영하면 가치 팩터가 좋아진다"는 결론이 유지되는지 확인한다.

  순수 가치(욕심 버전): R&D 감가율 {10,15,20,30%} × 조직 감가율 {10,20,30%} × 조직 비율 {0,20,30,40%} = 48개
    - 사이즈·베타·모멘텀·업종 점수와 회귀 표본은 한 번만 만들고, 가치 점수만 조합마다 다시 계산(빠르게)
    - 조직 비율 0% = R&D만 반영 → 개선이 R&D 덕인지 판관비(브랜드·조직) 덕인지 구분
  HML(2×3): 계산이 무거워 기본값에서 한 가지씩만 바꾼 7개
  각 조합: 성과와 '조정 − 기존' 차이의 t-통계량

실행: python "scripts/가치팩터(무형자산조정) 민감도.py"   →  결과/가치팩터(무형자산조정)/민감도_*.csv, 3_민감도.png
(확장 스크립트와 같은 폴더에 저장)
"""
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from 팩터공통 import RESULTS, attach_annual, load_tables, setup_korean_font, winsorize, zscore

OUT = RESULTS / "가치팩터(무형자산조정)"
SCRIPTS = Path(__file__).resolve().parent
기본 = {"감가율_지식": 0.15, "감가율_조직": 0.20, "조직비율": 0.30}
격자 = {"감가율_지식": [0.10, 0.15, 0.20, 0.30], "감가율_조직": [0.10, 0.20, 0.30], "조직비율": [0.0, 0.2, 0.3, 0.4]}


def load_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def t_of(x):
    x = pd.Series(x).dropna()
    return x.mean() / (x.std() / np.sqrt(len(x)))


def set_params(ext, p):
    for k, v in p.items():
        setattr(ext, k, v)


def main():
    ext = load_module("가치팩터(무형자산조정) 확장.py", "ext")
    pf = load_module("가치팩터(B-M) 순수팩터.py", "pf")
    t = load_tables()
    panel, annual0 = t["월별패널"], t["연간재무"]

    # ---- 순수 가치: 기본 표본(기존 B/M 기준)을 한 번 만들고 가치 점수만 바꿔 끼움
    pf.BE = "자본총계_지배_천원"
    e = pf.build_exposures(panel, annual0)
    me = e["가중치"] ** 2  # 가중치 = √시가총액

    def pure_value(value_raw):
        g = e.copy()
        g["가치"] = value_raw.groupby(g["연월"]).transform(lambda s: zscore(winsorize(s)))
        return g.groupby("연월").apply(pf.cross_section, include_groups=False)["가치"]

    base_pure = e.groupby("연월").apply(pf.cross_section, include_groups=False)["가치"]

    rows = []
    for dk, do, share in itertools.product(*격자.values()):
        p = {"감가율_지식": dk, "감가율_조직": do, "조직비율": share}
        set_params(ext, p)
        a = ext.add_intangibles(annual0)
        be = attach_annual(e, a, [ext.ADJ_BE])[ext.ADJ_BE].values
        r = pure_value(pd.Series(be, index=e.index) / (me * 1000))
        d = r - base_pure
        rows.append({**p, "월평균_%": r.mean(), "t통계량": t_of(r), "샤프": r.mean() / r.std() * np.sqrt(12),
                     "개선_월%": d.mean(), "개선_t": t_of(d), "개선_t_2013이후": t_of(d.loc["2012-12":])})
    pure = pd.DataFrame(rows)
    pure.to_csv(OUT / "민감도_순수가치.csv", index=False, encoding="utf-8-sig")

    # ---- HML(2×3): 기본값에서 한 가지씩만 바꾼 조합
    vf = load_module("가치팩터(B-M) 재현.py", "vf")
    base_hml, *_ = ext.hml_from(vf, panel, annual0, "자본총계_지배_천원")
    combos = [dict(기본)] + [{**기본, k: v} for k, vals in 격자.items() for v in vals if v != 기본[k]]
    combos = [c for c in combos if (c["감가율_조직"] == 기본["감가율_조직"])]  # 조직 감가율은 순수 가치로 충분히 봄
    hrows = []
    for p in combos:
        set_params(ext, p)
        a = ext.add_intangibles(annual0)
        hml, *_ = ext.hml_from(vf, panel, a, ext.ADJ_BE)
        d = hml - base_hml
        hrows.append({**p, "월평균_%": hml.mean(), "t통계량": t_of(hml), "개선_월%": d.mean(), "개선_t": t_of(d),
                      "개선_t_2013이후": t_of(d.loc["2013-01":])})
    hml_tab = pd.DataFrame(hrows)
    hml_tab.to_csv(OUT / "민감도_HML.csv", index=False, encoding="utf-8-sig")

    make_chart(pure)

    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(f"[순수 가치] 기존: 월 {base_pure.mean():.2f}%, t {t_of(base_pure):.2f}")
        print(f"  48개 조합 중 개선(차이 > 0): {(pure['개선_월%'] > 0).sum()}개, "
              f"개선이 유의(t > 2): {(pure['개선_t'] > 2).sum()}개, 나빠짐이 유의(t < −2): {(pure['개선_t'] < -2).sum()}개")
        print(f"  t통계량 범위 {pure['t통계량'].min():.2f} ~ {pure['t통계량'].max():.2f}")
        print("\n  조직비율별 평균 (0% = R&D만 반영)")
        print(pure.groupby("조직비율")[["t통계량", "개선_월%", "개선_t"]].mean().to_string())
        print("\n  R&D 감가율별 평균")
        print(pure.groupby("감가율_지식")[["t통계량", "개선_월%", "개선_t"]].mean().to_string())
        print("\n  조직 감가율별 평균")
        print(pure.groupby("감가율_조직")[["t통계량", "개선_월%", "개선_t"]].mean().to_string())
        print(f"\n[HML 2×3] 기존: 월 {base_hml.mean():.2f}%, t {t_of(base_hml):.2f}")
        print(hml_tab.to_string(index=False))
    print(f"\n결과 저장: {OUT}")


def make_chart(pure):
    plt = setup_korean_font()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, do in zip(axes, 격자["감가율_조직"]):
        piv = pure[pure["감가율_조직"] == do].pivot(index="조직비율", columns="감가율_지식", values="개선_t")
        im = ax.imshow(piv.values, cmap="RdBu", vmin=-4, vmax=4, origin="lower", aspect="auto")
        ax.set_xticks(range(piv.shape[1]), [f"{c:.0%}" for c in piv.columns])
        ax.set_yticks(range(piv.shape[0]), [f"{i:.0%}" for i in piv.index])
        for (i, j), v in np.ndenumerate(piv.values):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9)
        ax.set_xlabel("R&D 감가율"); ax.set_title(f"조직자본 감가율 {do:.0%}")
    axes[0].set_ylabel("조직자본 비율 (판관비 중)")
    fig.suptitle("순수 가치: 무형자산 조정의 개선 t-통계량 (2 이상이면 유의한 개선, 파란색)")
    fig.colorbar(im, ax=axes, shrink=.8)
    fig.savefig(OUT / "3_민감도.png", dpi=150, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
