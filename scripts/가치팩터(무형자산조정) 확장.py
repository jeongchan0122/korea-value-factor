"""
가치 팩터 확장: 무형자산 조정 B/M (Eisfeldt·Kim·Papanikolaou 2022 방식을 한국 데이터에 적용).

회계상 R&D·브랜드/조직 투자는 대부분 비용 처리돼 장부가에 안 남는다 → 무형자산이 많은 회사(IT·바이오 등)가
실제보다 '비싸게(성장주)' 보인다. 과거 투자를 쌓아 무형자산을 추정하고 장부가에 더해 B/M을 다시 계산한다.

  지식자본(t) = 지식자본(t−1) × (1 − 15%) + 연구개발비(t)                      (연구개발비 빈칸 = 0)
  조직자본(t) = 조직자본(t−1) × (1 − 20%) + 30% × max(판관비(t) − 연구개발비(t), 0)
                (한국 기업은 R&D 일부를 판관비에 넣어 보고하므로 겹치지 않게 뺀다)
  첫해 값    = 첫해 투자 ÷ (감가율 + 성장률 10%)    (그 전 투자를 모르니 꾸준히 성장해 왔다고 가정하는 표준 방법)
  조정 장부가 = 자본총계_지배 + 지식자본 + 조직자본

비교는 공정하게: 기존 재현·순수팩터 코드의 함수를 그대로 불러와 장부가 열만 바꾼다(기간·필터·바구니·수익률 동일).

실행: python "scripts/가치팩터(무형자산조정) 확장.py"   →  결과/가치팩터(무형자산조정)/
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from 팩터공통 import RESULTS, load_tables, performance, portfolio_returns, setup_korean_font

# ---------------------------------------------------------------- 설정
감가율_지식, 감가율_조직, 조직비율, 초기성장률 = 0.15, 0.20, 0.30, 0.10
OUT = RESULTS / "가치팩터(무형자산조정)"
ADJ_BE = "조정장부가_천원"
SCRIPTS = Path(__file__).resolve().parent


def load_module(filename, name):
    """파일 이름에 괄호·공백이 있어 import 문을 못 써서 경로로 불러온다."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def add_intangibles(annual):
    """연간재무에 지식자본·조직자본·조정장부가 열을 추가한다(종목별로 회계연도 순서대로 쌓음)."""
    a = annual.sort_values(["종목코드", "회계연도"]).copy()
    rd = a["연구개발비_천원"].fillna(0).clip(lower=0)
    org_inv = 조직비율 * (a["판매비와관리비_천원"].fillna(0) - rd).clip(lower=0)

    def accumulate(inv, delta):
        out = np.empty(len(inv))
        for g_idx in a.groupby("종목코드").indices.values():
            x = inv.values[g_idx]
            k = x[0] / (delta + 초기성장률)
            for j, v in enumerate(x):
                k = k * (1 - delta) + v if j else k
                out[g_idx[j]] = k
        return out

    a["지식자본_천원"] = accumulate(rd, 감가율_지식)
    a["조직자본_천원"] = accumulate(org_inv, 감가율_조직)
    a[ADJ_BE] = a["자본총계_지배_천원"] + a["지식자본_천원"] + a["조직자본_천원"]
    return a


def hml_from(vf, panel, annual, be_col):
    """재현 스크립트의 바구니 함수로 HML·SMB와 구성표를 만든다(장부가 열만 바꿈)."""
    vf.BE = be_col
    form = vf.form_portfolios(panel, annual)
    h = form[["구성연월", "종목코드", "바구니2x3"]].rename(columns={"바구니2x3": "바구니"})
    p6 = portfolio_returns(h, panel, vf.가중방식)
    p6 = p6[p6.index <= vf.끝연월]
    hml = (p6["SH"] + p6["BH"]) / 2 - (p6["SL"] + p6["BL"]) / 2
    smb = p6[["SL", "SM", "SH"]].mean(axis=1) - p6[["BL", "BM", "BH"]].mean(axis=1)
    return hml, smb, p6, form


def pure_value_from(pf, panel, annual, be_col):
    pf.BE = be_col
    e = pf.build_exposures(panel, annual)
    res = e.groupby("연월").apply(pf.cross_section, include_groups=False)
    res.index = pf.shift_ym(pd.Series(res.index), 1)
    assert np.allclose(res["가치노출"], 1, atol=1e-6) and (res["다른노출_최대절대값"] < 1e-6).all()
    return res["가치"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    vf = load_module("가치팩터(B-M) 재현.py", "vf")
    pf = load_module("가치팩터(B-M) 순수팩터.py", "pf")
    t = load_tables()
    panel, fac = t["월별패널"], t["FnGuide팩터지수"]
    annual = add_intangibles(t["연간재무"])

    base_be = "자본총계_지배_천원"
    hml_b, smb_b, p6_b, form_b = hml_from(vf, panel, annual, base_be)
    hml_a, smb_a, p6_a, form_a = hml_from(vf, panel, annual, ADJ_BE)
    pure_b = pure_value_from(pf, panel, annual, base_be)
    pure_a = pure_value_from(pf, panel, annual, ADJ_BE)

    fg = fac.pivot(index="연월", columns="코드", values="수익률_%")
    monthly = pd.DataFrame({"HML_기존": hml_b, "HML_무형조정": hml_a,
                            "순수가치_기존": pure_b, "순수가치_무형조정": pure_a})
    monthly["FnGuide_HML"] = fg["3FM.2B3.HML"].reindex(monthly.index)
    monthly = monthly.join(p6_a.add_prefix("조정바구니_"))
    monthly.index.name = "연월"
    monthly.to_csv(OUT / "월별수익률.csv", encoding="utf-8-sig")

    cols = ["HML_기존", "HML_무형조정", "순수가치_기존", "순수가치_무형조정", "FnGuide_HML"]
    report = pd.DataFrame({c: performance(monthly[c]) for c in cols}).T
    # 기간별: 앞 절반 / 뒤 절반 (무형자산이 중요해진 최근에 차이가 커지는지)
    for lab, (s, e) in {"2000~2012": ("2000-07", "2012-12"), "2013~2026": ("2013-01", "2026-09")}.items():
        for c in cols[:4]:
            p = performance(monthly[c].loc[s:e])
            report.loc[c, f"월평균_{lab}"], report.loc[c, f"t_{lab}"] = p["월평균_%"], p["t통계량"]
    report["FnGuide상관"] = [monthly[c].corr(monthly["FnGuide_HML"]) for c in cols]
    report.index.name = "시계열"
    report.to_csv(OUT / "성적표.csv", encoding="utf-8-sig")

    # 분류가 어떻게 바뀌었나: 업종별로 성장(L)·가치(H) 바구니에 들어간 비율
    sector = panel[["연월", "종목코드", "업종_FnGuide"]].rename(columns={"연월": "구성연월"})
    moves = []
    for name, form in [("기존", form_b), ("무형조정", form_a)]:
        f = form.merge(sector, on=["구성연월", "종목코드"], how="left")
        share = f.groupby("업종_FnGuide")["BM그룹"].value_counts(normalize=True).unstack().fillna(0)
        moves.append(share[["L", "H"]].add_prefix(f"{name}_"))
    moves = pd.concat(moves, axis=1)
    moves["성장비율_변화"] = moves["무형조정_L"] - moves["기존_L"]
    moves = moves.sort_values("성장비율_변화")
    moves.to_csv(OUT / "업종별_분류변화.csv", encoding="utf-8-sig")

    # 무형자산 규모: 조정 장부가 중 무형자산 비중(2024년)
    a24 = annual[(annual["회계연도"] == 2024) & (annual["자본총계_지배_천원"] > 0)]
    ratio = ((a24["지식자본_천원"] + a24["조직자본_천원"]) / a24[ADJ_BE])
    swaps = form_b.merge(form_a[["구성연월", "종목코드", "BM그룹"]], on=["구성연월", "종목코드"],
                         suffixes=("_기존", "_조정"))
    changed = (swaps["BM그룹_기존"] != swaps["BM그룹_조정"]).mean()

    make_charts(monthly, moves)

    show = ["개월수", "월평균_%", "t통계량", "연환산샤프", "최대낙폭_%", "FnGuide상관",
            "월평균_2000~2012", "t_2000~2012", "월평균_2013~2026", "t_2013~2026"]
    with pd.option_context("display.width", 220, "display.float_format", "{:.2f}".format):
        print(f"무형자산 비중(2024, 조정 장부가 중): 중앙값 {ratio.median():.1%}, 상위 10% {ratio.quantile(.9):.1%}")
        print(f"B/M 3그룹(L/M/H) 분류가 바뀐 종목 비율: {changed:.1%}")
        print(f"HML 기존 vs 조정 상관 {monthly['HML_기존'].corr(monthly['HML_무형조정']):.2f}, "
              f"순수가치 기존 vs 조정 상관 {monthly['순수가치_기존'].corr(monthly['순수가치_무형조정']):.2f}")
        print("\n[성적표]"); print(report[show].to_string())
        print("\n[업종별 성장(L) 바구니 비율 변화]"); print(moves[["기존_L", "무형조정_L", "성장비율_변화"]].to_string())
    print(f"\n결과 저장: {OUT}")


def make_charts(monthly, moves):
    plt = setup_korean_font()

    fig, ax = plt.subplots(figsize=(10, 5))
    for c, style in [("HML_무형조정", "-"), ("HML_기존", "--"), ("FnGuide_HML", ":")]:
        w = (1 + monthly[c].dropna() / 100).cumprod()
        ax.plot(pd.to_datetime(w.index, format="%Y-%m"), w.values, style, label=c)
    ax.set_yscale("log"); ax.set_title("무형자산 조정 HML vs 기존 HML 누적수익률 (1원 투자, 로그 눈금)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "1_누적수익률_비교.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(moves))
    ax.bar(x - .2, moves["기존_L"] * 100, .4, label="기존 B/M")
    ax.bar(x + .2, moves["무형조정_L"] * 100, .4, label="무형자산 조정 B/M")
    ax.set_xticks(x, moves.index, rotation=30)
    ax.set_ylabel("성장(L) 바구니에 들어간 비율 (%)")
    ax.set_title("업종별 '성장주'로 분류된 비율 (매년 6월 평균)"); ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "2_업종별_성장분류_변화.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
