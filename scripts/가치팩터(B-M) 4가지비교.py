"""
가치 팩터 4가지 방식 비교: 2×3 HML, 단순 롱숏(시총가중), 순수 팩터, 벤치마크 기울이기(시장 대비 초과수익).
기준선으로 시장 초과수익(시장 − 무위험)도 같이 넣는다.

그냥 겹치면 공정하지 않다:
  - 순수 팩터는 '가치 노출 1' 기준이라 크기(흔들림)가 임의로 정해짐
  - 기울이기는 롱온리 포트폴리오 그대로(무위험 초과수익) 넣음 → 시장 선에 붙어 가는 모양이 그대로 보임
    (시장 몫이 통째로 들어 있어 샤프는 낮게 나옴. 가치로 기울인 부분만의 효율은 정보비율로 따로 봄)
그래서 모든 시계열을 같은 변동성(연 10%)으로 맞춘 뒤 누적수익률을 비교한다(= 샤프지수를 그림으로 본 것).
맞추는 건 숫자 하나를 곱하는 것이라 샤프·t값은 바뀌지 않는다. 변동성을 키운 쪽은 레버리지를 쓴 셈(차입 비용 무시).

실행: python "scripts/가치팩터(B-M) 4가지비교.py"   →  결과/가치팩터(B-M)/4가지비교/
(재현·순수팩터·벤치마크기울이기 스크립트를 먼저 실행해 두어야 함)
"""
import numpy as np
import pandas as pd

from 팩터공통 import RESULTS, performance, setup_korean_font

목표변동성 = 10.0  # 연 %
기울이기강도 = "k=0.5"
BASE = RESULTS / "가치팩터(B-M)"
OUT = BASE / "4가지비교"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(BASE / "월별수익률.csv", encoding="utf-8-sig", index_col=0)
    pure = pd.read_csv(BASE / "순수팩터" / "순수팩터_월별수익률.csv", encoding="utf-8-sig", index_col=0)
    tilt = pd.read_csv(BASE / "벤치마크기울이기" / "월별수익률.csv", encoding="utf-8-sig", index_col=0)

    raw = pd.DataFrame({
        "① 2×3 HML": m["HML_2x3"],
        "② 단순 롱숏(시총가중)": m["HML_단순롱숏_시총가중"],
        "③ 순수 팩터": pure["순수_가치"],
        "④ 기울이기 (롱온리)": tilt[기울이기강도] - m["무위험"],
        "⑤ 시장 초과수익 (참고)": m["시장_분석대상전체"] - m["무위험"],  # 기준선: 그냥 시장에 투자
    }).dropna()

    vol = raw.std() * np.sqrt(12)
    scaled = raw * (목표변동성 / vol)

    table = pd.DataFrame({c: performance(raw[c]) for c in raw.columns}).T[
        ["개월수", "월평균_%", "월표준편차_%", "t통계량", "연환산샤프", "최대낙폭_%"]]
    table["원래_연변동성_%"] = vol
    table["맞춘_배수"] = 목표변동성 / vol
    table["맞춘뒤_누적_배"] = (1 + scaled / 100).prod()
    table["맞춘뒤_최대낙폭_%"] = [performance(scaled[c])["최대낙폭_%"] for c in scaled.columns]
    table.index.name = "방식"
    table.to_csv(OUT / "4가지비교.csv", encoding="utf-8-sig")
    scaled.to_csv(OUT / "변동성맞춘_월별수익률.csv", encoding="utf-8-sig")

    make_charts(raw, scaled)
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(table.to_string())
    print(f"\n결과 저장: {OUT}")


def make_charts(raw, scaled):
    plt = setup_korean_font()
    colors = ["#004C86", "#4F81BD", "#C00000", "#8DB33A", "#A6A6A6"]
    for name, df, title in [("1_원래크기_누적.png", raw, "4가지 방식 누적수익률 (원래 크기, 참고용)"),
                            ("2_같은변동성_누적.png", scaled, f"4가지 방식 누적수익률 (모두 연 변동성 {목표변동성:.0f}%로 맞춤)")]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for c, col in zip(df.columns, colors):
            w = (1 + df[c] / 100).cumprod()
            ax.plot(pd.to_datetime(w.index, format="%Y-%m"), w.values, label=c, color=col, lw=2)
        ax.set_yscale("log"); ax.set_title(title); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(OUT / name, dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
