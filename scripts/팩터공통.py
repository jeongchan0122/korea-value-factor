"""
팩터 분석에서 공통으로 쓰는 함수 모음. 가치 팩터 말고 다른 팩터를 할 때도 그대로 쓴다.

  불러오기     load_tables
  분석 대상    universe_mask
  재무 붙이기  attach_annual
  극단값·표준화 winsorize, zscore
  포트폴리오   portfolio_returns
  성과 통계    performance, newey_west_t
  그래프       setup_korean_font
"""
import matplotlib
import numpy as np
import pandas as pd

from korea_parsers import ROOT

PROCESSED = ROOT / "processed"
RESULTS = ROOT / "결과"
RET = "수정주가수익률_배당포함_상폐반영_%"  # 백테스트 수익률: 배당 포함 + 상장폐지 수익률 반영
ME = "시가총액_백만원"


def load_tables():
    names = ["월별패널", "연간재무", "시장시계열", "FnGuide팩터지수"]
    return {n: pd.read_parquet(PROCESSED / f"{n}.parquet") for n in names}


def ym_to_int(ym):
    """'2015-06' → 연*12 + (월-1). 개월 수 계산과 정렬용."""
    s = pd.Series(ym)
    return (s.str[:4].astype(int) * 12 + s.str[5:7].astype(int) - 1).values


def int_to_ym(i):
    i = pd.Series(i)
    return (i // 12).astype(str) + "-" + (i % 12 + 1).astype(str).str.zfill(2)


def shift_ym(ym, months):
    return int_to_ym(ym_to_int(ym) + months).values


def universe_mask(panel, exclude_financials=False):
    """
    그 달 말에 포트폴리오에 넣을 수 있는 종목인지(참/거짓).
    코스피·코스닥, 거래정지·관리종목 아님, 스팩 아님, 시가총액 있음, (선택) 금융업 아님.
    """
    m = (panel["시장구분"].isin(["KS", "KQ"])
         & (panel["거래정지구분"] != "거래정지")
         & (panel["관리감리구분"] != "관리")
         & ~panel["종목명"].fillna("").str.contains("스팩")
         & (panel[ME] > 0))
    if exclude_financials:
        m &= panel["업종_FnGuide"] != "금융"
    return m


def attach_annual(left, annual, cols, max_age_months=24):
    """
    left(연월, 종목코드)의 각 줄에, 그 연월에 쓸 수 있는 가장 최근 연간재무를 붙인다.
    '쓸 수 있다' = 사용가능연월 ≤ 연월(미래 정보 방지). 회계연도말이 max_age_months보다 오래되면 버린다.
    """
    l = left[["연월", "종목코드"]].copy()
    l["_t"] = ym_to_int(l["연월"])
    r = annual[["종목코드", "회계연도", "회계연도말", "사용가능연월", *cols]].copy()
    r["_t"] = ym_to_int(r["사용가능연월"])
    r = r.sort_values("_t")
    out = pd.merge_asof(l.reset_index().sort_values("_t"), r, on="_t", by="종목코드", direction="backward")
    out = out.set_index("index").sort_index()
    too_old = (l["_t"] - ym_to_int(out["회계연도말"].fillna("1900-01"))) > max_age_months
    out.loc[too_old, cols] = np.nan
    return out[["회계연도", "회계연도말", *cols]]


def winsorize(s, lower=0.01, upper=0.99):
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def zscore(s):
    return (s - s.mean()) / s.std()


def portfolio_returns(holdings, panel, weighting="시가총액"):
    """
    holdings: (구성연월, 종목코드, 바구니[, 승수]) — 구성연월 말에 만든 바구니. 다음 구성 때까지 보유.
    매달 수익률 = 그 달 수익률(RET)의 가중평균. 시가총액가중은 전달 말 시가총액으로 가중(FF 방식).
    '승수' 열이 있으면 가중치에 곱한다(벤치마크 비중을 기울이는 용도).
    반환: 연월 x 바구니 월수익률(%)
    """
    h = holdings.copy()
    extra = ["승수"] if "승수" in h.columns else []
    starts = np.sort(h["구성연월"].unique())
    ret = panel[["연월", "종목코드", RET]].dropna(subset=[RET])
    me_prev = panel[["연월", "종목코드", ME]].dropna(subset=[ME]).copy()
    me_prev["연월"] = shift_ym(me_prev["연월"], 1)  # 전달 말 시가총액을 이번 달에 붙임
    me_prev = me_prev.rename(columns={ME: "가중치"})

    out = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else None
        months = ret["연월"][(ret["연월"] > start) & ((ret["연월"] <= end) if end else True)].unique()
        hs = h[h["구성연월"] == start][["종목코드", "바구니", *extra]]
        r = ret[ret["연월"].isin(months)].merge(hs, on="종목코드")
        r = r.merge(me_prev, on=["연월", "종목코드"], how="left")
        if weighting == "시가총액":
            r = r.dropna(subset=["가중치"])
        else:
            r["가중치"] = 1.0
        if extra:
            r["가중치"] = r["가중치"] * r["승수"]
            r = r[r["가중치"] > 0]
        g = r.groupby(["연월", "바구니"])
        out.append((g.apply(lambda x: np.average(x[RET], weights=x["가중치"]), include_groups=False)).unstack())
    return pd.concat(out).sort_index()


def newey_west_t(x, lags=None):
    """평균이 0인지 검정하는 Newey-West t-통계량(자기상관 보정). lags 기본 = floor(4·(T/100)^(2/9))."""
    x = pd.Series(x).dropna().values
    T = len(x)
    if lags is None:
        lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
    e = x - x.mean()
    var = e @ e / T
    for L in range(1, lags + 1):
        var += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / T
    return x.mean() / np.sqrt(var / T)


def performance(r, rf=None):
    """
    월수익률(%) 시계열의 성적표.
    rf(무위험수익률, %)를 주면 샤프지수를 초과수익률(r − rf)로 계산한다. 롱온리 포트폴리오·시장용.
    롱숏(HML 등)은 자기 돈이 거의 안 드는 전략이라 rf를 빼지 않는다.
    """
    r = pd.Series(r).dropna()
    ex = r - pd.Series(rf).reindex(r.index) if rf is not None else r
    wealth = (1 + r / 100).cumprod()
    return pd.Series({
        "시작": r.index.min(), "끝": r.index.max(), "개월수": len(r),
        "월평균_%": r.mean(), "월표준편차_%": r.std(),
        "t통계량": r.mean() / (r.std() / np.sqrt(len(r))),
        "NW_t통계량": newey_west_t(r),
        "연환산샤프": ex.mean() / ex.std() * np.sqrt(12),
        "최대낙폭_%": ((wealth / wealth.cummax()) - 1).min() * 100,
        "누적수익률_%": (wealth.iloc[-1] - 1) * 100,
    })


def market_return(panel, exclude_financials=False):
    """
    시장 수익률(Fama-French의 시장 팩터 방식): 전달 말 분석 대상 전체를 전달 말 시가총액으로 가중한 이번 달 수익률(%).
    코스피 지수와 달리 코스닥까지 포함하고, 우리 분석 대상 필터(스팩·관리종목 제외 등)를 똑같이 적용한다.
    """
    prev = panel[universe_mask(panel, exclude_financials)][["연월", "종목코드", ME]].copy()
    prev["연월"] = shift_ym(prev["연월"], 1)
    r = panel[["연월", "종목코드", RET]].dropna().merge(prev, on=["연월", "종목코드"])
    return r.groupby("연월").apply(lambda x: np.average(x[RET], weights=x[ME]), include_groups=False)


def capm(y, mkt_excess):
    """
    y = 알파 + 베타 × 시장초과수익률 + 오차 (월, %). OLS.
    반환: 알파(월%), 알파 t(Newey-West), 베타, 베타 t, R²
    """
    d = pd.concat([pd.Series(y, name="y"), pd.Series(mkt_excess, name="x")], axis=1).dropna()
    X = np.column_stack([np.ones(len(d)), d["x"]])
    b, *_ = np.linalg.lstsq(X, d["y"].values, rcond=None)
    e = d["y"].values - X @ b
    # Newey-West 공분산(자기상관 보정)
    T, lags = len(d), int(np.floor(4 * (len(d) / 100) ** (2 / 9)))
    S = (X * e[:, None]).T @ (X * e[:, None]) / T
    for L in range(1, lags + 1):
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None]) / T
        S += (1 - L / (lags + 1)) * (G + G.T)
    XtX_inv = np.linalg.inv(X.T @ X / T)
    se = np.sqrt(np.diag(XtX_inv @ S @ XtX_inv) / T)
    r2 = 1 - (e @ e) / ((d["y"] - d["y"].mean()) @ (d["y"] - d["y"].mean()))
    return pd.Series({"개월수": T, "알파_월%": b[0], "알파_t": b[0] / se[0], "베타": b[1], "베타_t": b[1] / se[1],
                      "R제곱": r2})


def setup_korean_font():
    matplotlib.use("Agg")  # 창을 띄우지 않고 파일로만 저장
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지
    return plt
