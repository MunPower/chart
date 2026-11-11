import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import FinanceDataReader as fdr

st.set_page_config(page_title="추세전환/사이클 차트", layout="wide")
st.title("추세전환 / 추세사이클 차트")

c1, c2, c3 = st.columns(3)
code = c1.text_input("종목코드", "278530")
disp = c2.number_input("화면 표시 기간(거래일)", 60, 600, 120, 10)
calc = c3.number_input("계산 기간(거래일) - 직선/곡선 계산에 사용", 120, 2000, 720, 50,
                       help="클수록 긴 과거로 추세를 계산합니다. 직선추세 기울기를 맞출 때 조절하세요.")

c4, c5, c6 = st.columns(3)
lam = c4.select_slider("HP필터 λ (클수록 곡선이 완만)",
                       options=[5e3, 6e3, 7e3, 8e3, 9e3, 1e4, 2e4, 5e4, 1e5, 3e5, 1e6], value=8e3,
                       format_func=lambda v: f"{v:,.0f}")
kmul = c5.slider("상·하한 폭 (σ 배수, 90% 신뢰구간 = 1.645)", 0.5, 3.0, 1.645, 0.005)
band_pct = c6.number_input("상·하한 폭 직접 지정(%) · 0이면 σ방식", 0.0, 30.0, 0.0, 0.1,
                           help="증권사 차트는 곡선의 약 ±5.4%로 보입니다. 5.4를 입력해 보세요.")


@st.cache_data(ttl=600)
def load(code: str, days: int) -> pd.DataFrame:
    start = (pd.Timestamp.today() - pd.Timedelta(days=int(days * 1.6) + 10)).strftime("%Y-%m-%d")
    return fdr.DataReader(code, start).tail(int(days))


def hp_trend(z: np.ndarray, lam: float) -> np.ndarray:
    """Hodrick-Prescott 필터의 추세 성분: (I + λ D'D) τ = z"""
    n = len(z)
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i:i + 3] = [1, -2, 1]
    return np.linalg.solve(np.eye(n) + lam * D.T @ D, z)


def fit_cycle(r: np.ndarray):
    """잔차에 sin/cos 적합, 설명력(R²)이 가장 높은 주기 탐색"""
    n = len(r)
    t = np.arange(n)
    r0 = r - r.mean()
    best = (None, -1, None)
    for P in np.arange(5, min(n, 250) + 0.01, 0.5):
        X = np.column_stack([np.sin(2 * np.pi * t / P), np.cos(2 * np.pi * t / P), np.ones(n)])
        beta, *_ = np.linalg.lstsq(X, r, rcond=None)
        fit = X @ beta
        r2 = 1 - np.sum((r - fit) ** 2) / np.sum(r0 ** 2)
        if r2 > best[1]:
            best = (P, r2, fit)
    return best


calc = max(int(calc), int(disp))
try:
    df = load(code.strip(), calc)
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()
if df.empty or len(df) < 60:
    st.warning("데이터가 부족합니다(60개 이상 필요). 종목코드를 확인하세요.")
    st.stop()

y = df["Close"].to_numpy(dtype=float)
n = len(y)
t = np.arange(n)

# 직선추세: 계산기간 전체에 대한 1차 선형회귀
b1, b0 = np.polyfit(t, y, 1)
line = b0 + b1 * t
# 곡선추세: 직선추세 제거 → HP필터 평활화 → 직선추세 복원
curve = line + hp_trend(y - line, lam)
# 상·하한: 곡선 대비 비율 변동성 기준 (곡선 × (1 ± kσ))
rel = y / curve - 1
sd_rel = rel.std()
width = band_pct / 100 if band_pct > 0 else kmul * sd_rel
upper, lower = curve * (1 + width), curve * (1 - width)
resid = y - curve

# 화면에는 최근 disp일만 표시
D = int(min(disp, n))
sl = slice(n - D, n)
dates = df.index.strftime("%Y-%m-%d")[sl]
yv, linev, curvev, upv, lov = y[sl], line[sl], curve[sl], upper[sl], lower[sl]

last, prev = y[-1], y[-2]
m = st.columns(5)
m[0].metric("현재가", f"{last:,.0f}", f"{last - prev:+,.0f}", delta_color="inverse")
m[1].metric("전일대비", f"{(last / prev - 1) * 100:+.2f}%")
m[2].metric("거래량", f"{df['Volume'].iloc[-1]:,.0f}")
m[3].metric("상한 대비율", f"{last / upper[-1] * 100:.1f}%", help="100% 초과 시 90% 범위 위로 이탈")
m[4].metric("하한 대비율", f"{last / lower[-1] * 100:.1f}%", help="100% 미만 시 90% 범위 아래로 이탈")
st.caption(f"직선추세: 화면 시작 {linev[0]:,.0f} → 끝 {linev[-1]:,.0f} · "
           f"상·하한 폭: 곡선의 ±{width * 100:.1f}%")


with st.expander("🔧 계산 기간 찾기 (증권사 직선추세 값에 맞추기)"):
    t1, t2 = st.columns(2)
    tgt0 = t1.number_input("증권사 직선추세 화면 시작값", value=30800, step=100)
    tgt1 = t2.number_input("증권사 직선추세 화면 끝값", value=36000, step=100)
    if st.button("가장 가까운 계산 기간 찾기"):
        full = load(code.strip(), 2000)["Close"].to_numpy(dtype=float)
        rows = []
        for L in range(int(disp) + 20, len(full) + 1, 10):
            yy = full[-L:]
            sl_, c0_ = np.polyfit(np.arange(L), yy, 1)
            a = c0_ + sl_ * (L - int(disp))
            b = c0_ + sl_ * (L - 1)
            rows.append((L, a, b, abs(a - tgt0) + abs(b - tgt1)))
        res = pd.DataFrame(rows, columns=["계산기간", "직선 시작값", "직선 끝값", "오차"]).sort_values("오차").head(5)
        st.dataframe(res.round(0).astype(int), hide_index=True)
        st.caption("오차가 가장 작은 '계산기간'을 위의 계산 기간 칸에 넣어 보세요.")


def direction_colors(v):
    d = np.diff(v, prepend=v[1])
    d[0] = d[1]
    return np.where(d >= 0, "red", "blue")  # 상승=빨강, 하락=파랑


def base_traces(fig):
    fig.add_trace(go.Scatter(x=dates, y=yv, name="수정종가", line=dict(color="red", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=linev, name="직선추세", line=dict(color="magenta", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=curvev, name="곡선추세", line=dict(color="royalblue", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=upv, name="곡선추세상한", line=dict(color="navy", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=lov, name="곡선추세하한", line=dict(color="navy", width=1)), row=1, col=1)


def finish(fig):
    fig.update_xaxes(type="category", nticks=8)
    # 십자선(마우스 위치를 따라다니는 가로·세로선) + 날짜별 값 한꺼번에 표시
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor",
                     spikethickness=1, spikedash="dot", spikecolor="gray")
    fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor",
                     spikethickness=1, spikedash="dot", spikecolor="gray")
    # 값 표시는 수정종가(가격)만
    for tr in fig.data:
        if tr.name == "수정종가":
            tr.hovertemplate = "%{y:,.0f}"
        else:
            tr.hoverinfo = "skip"
    fig.update_layout(height=700, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.06), bargap=0.05,
                      hovermode="x unified")
    return fig


tab1, tab2 = st.tabs(["추세전환차트", "추세사이클차트"])

with tab1:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.03)
    base_traces(fig)
    cv = curvev - curvev.min()
    fig.add_trace(go.Bar(x=dates, y=cv, marker_color=direction_colors(curvev),
                         name="곡선추세 방향", showlegend=False), row=2, col=1)
    st.plotly_chart(finish(fig), use_container_width=True)
    st.caption("하단 봉: 곡선추세가 상승 중이면 빨강, 하락 중이면 파랑 (추세 전환 시점 확인용)")

with tab2:
    P, r2, cyc = fit_cycle(resid)
    cyc = (cyc - cyc.mean())[sl]
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.03)
    base_traces(fig2)
    fig2.add_trace(go.Bar(x=dates, y=cyc, marker_color=direction_colors(cyc),
                          name="싸이클", showlegend=False), row=2, col=1)
    st.plotly_chart(finish(fig2), use_container_width=True)
    st.caption(f"추정 최적 주기 약 {P:.1f}거래일 · 잔차 설명력(R²) {r2 * 100:.0f}%. "
               "싸이클의 상승/하락은 추세의 상승/하락과 무관하며, 추세선보다 설명력이 낮은 보조 지표입니다.")

st.info("HP필터 특성상 새 데이터가 추가되면 과거 곡선도 다시 계산되어 모양이 바뀝니다(끝점 효과). "
        "미래에셋 차트는 스펙트럴 분석 등 비공개 방식을 쓰므로 이 앱은 근사 재현입니다. 투자 판단은 본인 책임입니다.")
