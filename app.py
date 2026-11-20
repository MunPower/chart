import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import FinanceDataReader as fdr
import json
import html
from concurrent.futures import ThreadPoolExecutor
import re
import streamlit.components.v1 as components

st.set_page_config(page_title="추세전환/사이클 차트", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
.block-container{padding:0.8rem 0.4rem 1rem 0.4rem;max-width:100%;}
header,[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],footer,
[data-testid="manage-app-button"],[class*="_profileContainer_"],[class*="viewerBadge"],[class*="_terminalButton_"],[class*="_gzau3_"]{display:none !important;}
</style>""", unsafe_allow_html=True)

# ───────────── 차트 계산 설정(고정값) ─────────────
disp, calc, lam, kmul, band_pct = 120, 720, 8e3, 1.6, 0.0
DEFAULT_CODE = "278530"   # 처음 열 때 보여줄 종목: KODEX 200TR


# ───────────── 종목 검색 (한글·영문 혼용) ─────────────
LETTERS = {"A": "에이", "B": "비", "C": "씨", "D": "디", "E": "이", "F": "에프", "G": "지", "H": "에이치",
           "I": "아이", "J": "제이", "K": "케이", "L": "엘", "M": "엠", "N": "엔", "O": "오", "P": "피",
           "Q": "큐", "R": "알", "S": "에스", "T": "티", "U": "유", "V": "브이", "W": "더블유",
           "X": "엑스", "Y": "와이", "Z": "제트"}
# 영문 단어 → 한글 발음 (자주 쓰는 영문 사명·ETF 브랜드)
WORDS = {
    "naver": "네이버", "posco": "포스코", "kakao": "카카오", "kakaobank": "카카오뱅크", "kakaopay": "카카오페이",
    "kodex": "코덱스", "tiger": "타이거", "ace": "에이스", "sol": "쏠", "kbstar": "케이비스타",
    "hanaro": "하나로", "rise": "라이즈", "plus": "플러스", "arirang": "아리랑", "kosef": "코세프",
    "kindex": "킨덱스", "timefolio": "타임폴리오", "woori": "우리", "hana": "하나", "shinhan": "신한",
    "samsung": "삼성", "hyundai": "현대", "lotte": "롯데", "hanwha": "한화", "doosan": "두산",
    "celltrion": "셀트리온", "kia": "기아", "amorepacific": "아모레퍼시픽", "krafton": "크래프톤",
    "netmarble": "넷마블", "ncsoft": "엔씨소프트", "oil": "오일", "lg": "엘지", "sk": "에스케이",
    "kt": "케이티", "cj": "씨제이", "gs": "지에스", "kb": "케이비", "db": "디비", "ls": "엘에스",
}
PHRASES = {r"s-oil": "에쓰오일"}

# ───────────── 미국 주식: 한글 이름으로도 찾을 수 있게 (티커: (한글이름, 거래소)) ─────────────
# 거래소가 NASDAQ이면 나스닥 지수, 그 밖(NYSE 등)이면 S&P500 지수와 비교
US_KO = {
    "AAPL": ("애플", "NASDAQ"), "MSFT": ("마이크로소프트", "NASDAQ"), "NVDA": ("엔비디아", "NASDAQ"),
    "AMZN": ("아마존", "NASDAQ"), "GOOGL": ("알파벳A 구글", "NASDAQ"), "GOOG": ("알파벳C 구글", "NASDAQ"),
    "META": ("메타 페이스북", "NASDAQ"), "TSLA": ("테슬라", "NASDAQ"), "AVGO": ("브로드컴", "NASDAQ"),
    "AMD": ("에이엠디", "NASDAQ"), "NFLX": ("넷플릭스", "NASDAQ"), "INTC": ("인텔", "NASDAQ"),
    "QCOM": ("퀄컴", "NASDAQ"), "MU": ("마이크론", "NASDAQ"), "ASML": ("에이에스엠엘", "NASDAQ"),
    "ADBE": ("어도비", "NASDAQ"), "CSCO": ("시스코", "NASDAQ"), "PEP": ("펩시코", "NASDAQ"),
    "COST": ("코스트코", "NASDAQ"), "SBUX": ("스타벅스", "NASDAQ"), "PLTR": ("팔란티어", "NASDAQ"),
    "PYPL": ("페이팔", "NASDAQ"), "ARM": ("암홀딩스", "NASDAQ"), "SMCI": ("슈퍼마이크로컴퓨터", "NASDAQ"),
    "MSTR": ("마이크로스트래티지", "NASDAQ"), "COIN": ("코인베이스", "NASDAQ"), "ABNB": ("에어비앤비", "NASDAQ"),
    "RIVN": ("리비안", "NASDAQ"), "LCID": ("루시드", "NASDAQ"), "QQQ": ("나스닥100 ETF", "NASDAQ"),
    "TQQQ": ("나스닥100 3배 ETF", "NASDAQ"), "SOXL": ("반도체 3배 ETF", "NASDAQ"),
    "IONQ": ("아이온큐", "NYSE"), "TSM": ("TSMC 대만반도체", "NYSE"), "BRK-B": ("버크셔해서웨이", "NYSE"),
    "JPM": ("제이피모건", "NYSE"), "V": ("비자", "NYSE"), "MA": ("마스터카드", "NYSE"),
    "KO": ("코카콜라", "NYSE"), "DIS": ("디즈니", "NYSE"), "NKE": ("나이키", "NYSE"),
    "MCD": ("맥도날드", "NYSE"), "WMT": ("월마트", "NYSE"), "JNJ": ("존슨앤드존슨", "NYSE"),
    "PFE": ("화이자", "NYSE"), "LLY": ("일라이릴리", "NYSE"), "NVO": ("노보노디스크", "NYSE"),
    "XOM": ("엑슨모빌", "NYSE"), "BA": ("보잉", "NYSE"), "ORCL": ("오라클", "NYSE"),
    "CRM": ("세일즈포스", "NYSE"), "IBM": ("아이비엠", "NYSE"), "UBER": ("우버", "NYSE"),
    "BABA": ("알리바바", "NYSE"), "SPY": ("S&P500 ETF", "NYSE"), "VOO": ("뱅가드 S&P500 ETF", "NYSE"),
    "SCHD": ("슈왑 배당 ETF", "NYSE"),
}
US_MKTS = ("NASDAQ", "NYSE", "AMEX", "US")


def is_us(code: str) -> bool:
    """한국 종목코드는 숫자로 시작(005930, 0091C0), 미국 티커는 영문으로 시작(AAPL)"""
    return bool(code) and code[0].isalpha()


def norm(t) -> str:
    return re.sub(r"[\s\-_.·&/()]", "", str(t)).lower()


def to_ko(name: str) -> str:
    """영문이 섞인 이름을 한글 발음으로 변환 (NAVER→네이버, LG전자→엘지전자, SK하이닉스→에스케이하이닉스)"""
    t = str(name)
    for k, v in PHRASES.items():
        t = re.sub(k, v, t, flags=re.I)

    def rep(m):
        w = m.group(0)
        if w.lower() in WORDS:
            return WORDS[w.lower()]
        if len(w) <= 6 and (w.isupper() or len(w) <= 3):      # 약어는 알파벳 읽는 소리로 (예: GS→지에스)
            return "".join(LETTERS.get(c.upper(), c) for c in w)
        return w
    return re.sub(r"[A-Za-z]+", rep, t)


@st.cache_data(ttl=86400, show_spinner=False)
def listing() -> pd.DataFrame:
    """종목명 검색용 목록 (국내 주식 + 국내 ETF + 미국 나스닥·뉴욕·아멕스 + 미국 ETF)"""
    # (목록 이름, 거래소 표시) — 국내는 목록 안의 Market 값으로 코스피/코스닥을 나눔
    SOURCES = (("KRX", ""), ("ETF/KR", "ETF"), ("NASDAQ", "NASDAQ"), ("NYSE", "NYSE"),
               ("AMEX", "AMEX"), ("ETF/US", "US"))

    def one(src):
        market, label = src
        try:
            d = fdr.StockListing(market)
            cc = "Code" if "Code" in d.columns else "Symbol"
            d = d.rename(columns={cc: "Code"})
            if label in US_MKTS or "Market" not in d.columns:
                d["Market"] = label
            return d[["Code", "Name", "Market"]]
        except Exception:
            return None
    ex = ThreadPoolExecutor(max_workers=len(SOURCES))             # 목록을 동시에 받아 대기 시간 단축
    futs = [ex.submit(one, s) for s in SOURCES]
    frames = []
    for f in futs:
        try:
            r = f.result(timeout=25)                               # 한 목록이 늦어도 나머지는 사용
            if r is not None:
                frames.append(r)
        except Exception:
            pass
    ex.shutdown(wait=False)
    # 자주 찾는 미국 종목은 목록을 못 받아도 항상 검색되게 추가
    frames.append(pd.DataFrame([(k, k, v[1]) for k, v in US_KO.items()], columns=["Code", "Name", "Market"]))
    out = pd.concat(frames).dropna().reset_index(drop=True)
    out["Code"] = out["Code"].astype(str).str.strip().str.replace(".", "-", regex=False)   # BRK.B → BRK-B
    out = out.drop_duplicates("Code").reset_index(drop=True)      # 먼저 받은 목록(실제 회사명)을 우선 사용

    def mk(x):
        x = str(x).upper()
        if x in US_MKTS:
            return x
        return "KOSDAQ" if "KOSDAQ" in x else "KOSPI"
    out["mkt"] = out["Market"].map(mk)
    for k, (ko, m) in US_KO.items():                               # 거래소를 확실히 아는 종목은 고정
        out.loc[out["Code"] == k, "mkt"] = m
    out["ko"] = out["Code"].map(lambda c: US_KO.get(c, ("",))[0])   # 미국 종목 한글 이름
    us = out["mkt"].isin(US_MKTS)
    out["nn"] = out["Name"].map(norm)                       # 원래 이름(정규화)
    out["nk"] = out["Name"].map(lambda x: norm(to_ko(x)))   # 한글 발음 이름(정규화)
    out.loc[us, "nk"] = [norm(k) if k else n for k, n in zip(out.loc[us, "ko"], out.loc[us, "nn"])]
    out["disp"] = [f"{k} {nm}" if k and k != nm else nm for k, nm in zip(out["ko"], out["Name"])]
    return out


def search(lst: pd.DataFrame, q: str):
    """('exact', 1행) 바로 표시 / ('list', 여러행) 연관 종목 선택 / ('none', 빈표)"""
    nq, nqk = norm(q), norm(to_ko(q))
    if not nq or lst.empty:
        return "none", lst.iloc[0:0]
    qc = q.strip().upper().replace(".", "-")
    exact = lst[(lst["nn"] == nq) | (lst["nk"] == nq) | (lst["nk"] == nqk) | (lst["Code"] == qc)]
    if len(exact) == 1:
        return "exact", exact
    m = exact
    if m.empty:
        m = lst[lst["nn"].str.contains(nq, regex=False) | lst["nk"].str.contains(nq, regex=False)
                | lst["nk"].str.contains(nqk, regex=False) | lst["Code"].str.startswith(q.upper())]
    if m.empty:
        return "none", m
    if len(m) == 1:
        return "exact", m
    starts = m["nn"].str.startswith(nq) | m["nk"].str.startswith(nq) | m["nk"].str.startswith(nqk)
    m = m.assign(_s=(~starts).astype(int), _l=m["Name"].str.len()).sort_values(["_s", "_l"])
    return "list", m


# ───────────── 종목 이름 + 검색 ─────────────
if "code" not in st.session_state:
    st.session_state.code = DEFAULT_CODE

name_ph = st.empty()
q = st.text_input("종목 검색", placeholder="🔍 종목코드·종목명 (예: 삼성전자, 278530, 애플, AAPL, 테슬라)",
                  label_visibility="collapsed").strip()
# 종목 목록은 검색할 때(또는 다른 종목을 보고 있을 때)만 불러온다 → 처음 화면이 바로 뜸
if q or st.session_state.code != DEFAULT_CODE:
    with st.spinner("종목 목록(국내·미국)을 불러오는 중입니다… 처음 한 번은 조금 걸릴 수 있어요"):
        lst = listing()
else:
    lst = pd.DataFrame(columns=["Code", "Name", "nn", "nk", "mkt", "disp"])
names = dict(zip(lst["Code"], lst["disp"]))
mkts = dict(zip(lst["Code"], lst["mkt"]))        # 코스피/코스닥/나스닥/뉴욕 구분 (국내 ETF는 코스피로 처리)
names.setdefault(DEFAULT_CODE, "KODEX 200TR")   # 종목 목록 없이도 기본 종목 이름 표시
for _k, (_ko, _m) in US_KO.items():             # 목록을 못 받아도 인기 미국 종목은 한글 이름 표시
    names.setdefault(_k, _ko)
    mkts.setdefault(_k, _m)
MKT_KO = {"NASDAQ": "나스닥", "NYSE": "뉴욕", "AMEX": "아멕스", "US": "미국"}
if q:
    qu = q.upper().replace(".", "-")
    if len(q) == 6 and q.isdigit():
        st.session_state.code = qu                                  # 국내 6자리 종목코드 → 바로 표시
    elif lst.empty:
        if re.fullmatch(r"[A-Z]{1,5}(-[A-Z])?", qu):               # 목록이 없어도 미국 티커는 바로 시도
            st.session_state.code = qu
        else:
            st.caption("종목 목록을 불러오지 못했습니다. 6자리 종목코드나 미국 티커(예: AAPL)로 검색해 주세요.")
    else:
        kind, m = search(lst, q)
        if kind == "none" and re.fullmatch(r"[A-Z]{1,5}(-[A-Z])?", qu):
            st.session_state.code = qu                              # 목록에 없는 미국 티커도 바로 시도
        elif kind == "exact":                                       # 이름·코드가 정확히 맞으면 바로 표시
            st.session_state.code = m.iloc[0]["Code"]
        elif kind == "list":                                        # 일부만 입력 → 연관 종목에서 선택
            opts = [f"{r.disp}{' · ' + MKT_KO[r.mkt] if r.mkt in MKT_KO else ''} ({r.Code})"
                    for r in m.head(30).itertuples()]
            sel = st.selectbox("연관 종목", opts, index=None, key=f"sel_{norm(q)}",
                               placeholder=f"연관 종목 {len(m)}개 — 여기서 선택", label_visibility="collapsed")
            if sel:
                st.session_state.code = sel.rsplit("(", 1)[1].rstrip(")")
        else:
            st.caption("검색 결과가 없습니다.")
code = st.session_state.code
name = names.get(code, code)
US = is_us(code)                                  # 미국 종목 여부
DEC = 2 if US else 0                              # 가격 소수 자릿수 (미국 $187.45 / 국내 71,500)
CUR = "$" if US else ""


def px(v) -> str:
    return f"{CUR}{v:,.{DEC}f}"


code_tag = html.escape(code) + (f" · {MKT_KO.get(mkts.get(code, 'US'), '미국')}" if US else "")
name_ph.markdown(
    f'<div style="font-size:20px;font-weight:700;line-height:1.3">{html.escape(str(name))} '
    f'<span style="font-size:13px;color:#888;font-weight:400">{code_tag}</span></div>',
    unsafe_allow_html=True)

# 시세를 받는 동안 "고장난 것처럼" 보이지 않도록 안내 표시 (차트가 나오면 사라짐)
status = st.empty()
status.markdown(
    '<div style="padding:10px 12px;border-radius:8px;background:#fff4dc;color:#8a5a00;font-size:14px;line-height:1.5">'
    '⏳ 시세를 불러오는 중입니다. 처음 한 번은 20~30초 걸릴 수 있어요. 잠시만 기다려 주세요.</div>',
    unsafe_allow_html=True)

@st.cache_data(ttl=30, show_spinner=False)
def live_index(sym: str):
    """네이버에서 지수 현재값(장중이면 실시간, 장 마감 후엔 종가)을 받는다. 실패하면 None (국내 지수만)"""
    if sym not in ("KS11", "KQ11"):
        return None
    import requests
    try:
        r = requests.get(f"https://m.stock.naver.com/api/index/{'KOSDAQ' if sym == 'KQ11' else 'KOSPI'}/basic",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=4)
        j = r.json()
        val = float(str(j["closePrice"]).replace(",", ""))
        day = pd.Timestamp(str(j.get("localTradedAt", ""))[:10]).normalize()
        return day, val
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def load_both(code: str, idx_sym: str, days: int):
    """종목과 지수 시세를 동시에 받아 온다. 지수는 8초 안에 못 받으면 None(차트는 지수선 없이 표시)"""
    start = (pd.Timestamp.today() - pd.Timedelta(days=int(days * 1.6) + 10)).strftime("%Y-%m-%d")
    ex = ThreadPoolExecutor(max_workers=2)
    f_stock = ex.submit(fdr.DataReader, code.replace(".", "-"), start)
    f_idx = ex.submit(fdr.DataReader, idx_sym, start)
    try:
        stock = f_stock.result(timeout=40).tail(int(days))
        try:
            idx = f_idx.result(timeout=8).tail(int(days))
        except Exception:
            idx = None
        return stock, idx
    finally:
        ex.shutdown(wait=False)


def hp_trend(z: np.ndarray, lam: float) -> np.ndarray:
    """Hodrick-Prescott 필터의 추세 성분: (I + λ D'D) τ = z"""
    n = len(z)
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i:i + 3] = [1, -2, 1]
    return np.linalg.solve(np.eye(n) + lam * D.T @ D, z)


calc = max(int(calc), 180)
mkt = mkts.get(code, "NASDAQ" if US else "KOSPI")
if US:      # 미국: 나스닥 상장 → 나스닥 종합지수, 그 밖(뉴욕 등) → S&P500
    idx_sym, idx_name = ("IXIC", "나스닥") if mkt == "NASDAQ" else ("US500", "S&P500")
else:
    idx_sym, idx_name = ("KQ11", "코스닥") if mkt == "KOSDAQ" else ("KS11", "코스피")
try:
    df, _idf = load_both(code.strip(), idx_sym, calc)
except Exception as e:
    status.empty()
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()
if df.empty or len(df) < 60:
    status.empty()
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

# 화면에는 최근 disp일만 표시
D = n
init_span = int(min(disp, D))
sl = slice(n - D, n)
dates = df.index.strftime("%Y-%m-%d")[sl]
yv, curvev, upv, lov = y[sl], curve[sl], upper[sl], lower[sl]
def _ohl(c):   # 시가·고가·저가가 비어 있으면 종가로 대신 채움
    a = df[c].to_numpy(dtype=float)[sl] if c in df.columns else yv.copy()
    return np.where(np.isfinite(a) & (a > 0), a, yv)
ov, hv, lv = _ohl("Open"), _ohl("High"), _ohl("Low")
hv, lv = np.maximum.reduce([hv, ov, yv]), np.minimum.reduce([lv, ov, yv])

# ───────────── 지수(코스피/코스닥) 비교선 ─────────────
idx_arr = None
_live_ok = False
try:
    _s = _idf["Close"].reindex(df.index).ffill().bfill()
    _live = live_index(idx_sym)                 # 종목처럼 접속할 때마다 최신 지수값으로 갱신
    if _live is not None and _live[0] >= df.index[-1].normalize():
        _s.iloc[-1] = _live[1]
        _live_ok = True
    if len(_s) == n and np.isfinite(_s.to_numpy(dtype=float)).all():
        idx_arr = _s.to_numpy(dtype=float)
except Exception:
    idx_arr = None
idx_note = ""
if idx_arr is not None:     # 지수 데이터가 종목보다 며칠 일찍 끝나면 알려줌 (마지막 값으로 채워졌다는 뜻)
    _gap = (df.index.max() - _idf.index.max()).days
    if _gap >= 1 and not _live_ok:
        idx_note = f"※ {idx_name} 데이터가 {_idf.index.max():%Y-%m-%d}까지만 있어 최근 {_gap}일은 마지막 값으로 채웠습니다."
idx_scaled = None
if idx_arr is not None:     # 처음 열었을 때 맨 왼쪽 종가에 한 번만 맞춘 뒤 고정 (밀거나 확대해도 변하지 않음)
    idx_scaled = idx_arr * (yv[n - init_span] / idx_arr[n - init_span])
PHONE_SPAN_PY = min(42, init_span)
idx_scaled_p = None
if idx_scaled is not None:  # 폰은 처음 2달치만 보이므로 그 맨 왼쪽 종가에 맞춘 지수선도 따로 준비
    idx_scaled_p = idx_arr * (yv[n - PHONE_SPAN_PY] / idx_arr[n - PHONE_SPAN_PY])

# ───────────── 상단 시세 헤더 ─────────────
last, prev = y[-1], y[-2]
chg = last - prev
pct = chg / prev * 100
col = "#e5484d" if chg > 0 else ("#3b6fd4" if chg < 0 else "#333333")
arrow = "▲" if chg > 0 else ("▼" if chg < 0 else "-")
vol = df["Volume"].iloc[-1]

# ───────────── 종목명 옆: 기준 날짜 + 지수 ─────────────
if idx_arr is not None:
    name_ph.markdown(
        f'<div style="display:flex;align-items:baseline;gap:18px;flex-wrap:wrap">'
        f'<div style="font-size:20px;font-weight:700;line-height:1.3">{html.escape(str(name))} '
        f'<span style="font-size:13px;color:#888;font-weight:400">{html.escape(code)}</span></div>'
        f'<div style="font-size:17px;font-weight:700;color:#0aa05a">'
        f'● {df.index[-1]:%Y-%m-%d} {idx_name} {idx_arr[-1]:,.2f}</div>'
        '</div>',
        unsafe_allow_html=True)

# ───────────── 매수·매도 표시 (현재가가 곡선추세 상·하한에 닿거나 넘었을 때) ─────────────
TOUCH_TOL = 0          # 0 = 상·하한에 정확히 닿거나 넘었을 때만 표시
_up, _lo = float(upper[-1]), float(lower[-1])
sig_html = ""
if last >= _up * (1 - TOUCH_TOL):          # 상한에 닿거나 위로 넘음 → 파란색 매도
    sig_html = '<div style="font-size:40px;font-weight:800;color:#1a56db;line-height:1">매도</div>'
elif last <= _lo * (1 + TOUCH_TOL):        # 하한에 닿거나 아래로 내려감 → 빨간색 매수
    sig_html = '<div style="font-size:40px;font-weight:800;color:#e5484d;line-height:1">매수</div>'

_sig = ""
if sig_html:
    _sig_txt, _sig_col = ("매도 가능합니다", "#1a56db") if "매도" in sig_html else ("매수 가능합니다", "#e5484d")
    _sig = f'<div class="hd-sig" style="color:{_sig_col}">{_sig_txt}</div>'
st.markdown(
    '<style>'
    '.hd-row{display:flex;align-items:center;gap:16px;flex-wrap:nowrap}'
    '.hd-px{display:flex;flex-direction:column;align-items:flex-start;gap:4px}'
    '.hd-big{font-size:42px;font-weight:700;line-height:1}'
    '.hd-chg{font-size:16px}'
    '.hd-lines{display:flex;flex-direction:column;gap:10px}'
    '.hd-lbl{font-size:15px;color:#111;line-height:1.4;text-align:center;white-space:nowrap}'
    '.hd-sig{font-size:24px;font-weight:800;line-height:1.2;white-space:nowrap}'
    '@media (max-width:600px){'
    '.hd-row{gap:10px}.hd-big{font-size:30px}.hd-chg{font-size:14px}'
    '.hd-lbl{font-size:12px}.hd-sig{font-size:17px}}'
    '</style>'
    '<div class="hd-row">'
    '<div>'
    f'<div class="hd-px"><span class="hd-big" style="color:{col}">{px(last)}</span>'
    f'<span class="hd-chg" style="color:{col}">{arrow} {abs(chg):,.{DEC}f} &nbsp; {pct:+.2f}%</span></div>'
    f'<div style="font-size:13px;color:#888;margin-top:6px">거래량 {vol:,.0f}</div>'
    '</div>'
    '<div class="hd-lines">'
    f'<div class="hd-lbl" data-ul="1" style="color:#1a56db;font-weight:600">매도선<br>(<span data-k="u">{_up:,.{DEC}f}</span>)</div>'
    f'<div class="hd-lbl" data-ul="1" style="color:#e5484d;font-weight:600">매수선<br>(<span data-k="l">{_lo:,.{DEC}f}</span>)</div>'
    '</div>'
    f'{_sig}'
    '</div>',
    unsafe_allow_html=True)

# ───────────── 차트 ─────────────
def direction_colors(v):
    d = np.diff(v, prepend=v[1])
    d[0] = d[1]
    return np.where(d >= 0, "#e5484d", "#3b6fd4")  # 상승=빨강, 하락=파랑


def base_traces(fig):
    fig.add_trace(go.Candlestick(x=dates, open=ov, high=hv, low=lv, close=yv, name="수정종가", showlegend=False,
                                 increasing=dict(line=dict(color="#e5484d", width=1), fillcolor="#e5484d"),
                                 decreasing=dict(line=dict(color="#3b6fd4", width=1), fillcolor="#3b6fd4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=curvev, name="곡선추세", line=dict(color="#111111", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=upv, name="매도선", line=dict(color="#1a56db", width=2.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=lov, name="매수선", line=dict(color="#e5484d", width=2.2)), row=1, col=1)
    if idx_scaled is not None:   # 4번 트레이스: 지수(녹색 점선)
        fig.add_trace(go.Scatter(x=dates, y=idx_scaled, name=idx_name,
                                 line=dict(color="#00a651", width=3, dash="dot")), row=1, col=1)
        if idx_scaled_p is not None:   # 5번 트레이스: 폰용 지수선(처음엔 숨김, 폰이면 JS가 이것으로 교체)
            fig.add_trace(go.Scatter(x=dates, y=idx_scaled_p, name=idx_name, visible=False, showlegend=False,
                                     line=dict(color="#00a651", width=3, dash="dot")), row=1, col=1)


CROSSHAIR_JS = """
function installCrosshair(gd, initSpan, S, B, IDX, IT, IDXNAME, IDXRAW, OPEN, LOW, ANCH, IDX2, IT2, HIGH, DEC){
  DEC=DEC||0;              // 가격 소수 자릿수 (국내 0, 미국 2)
  var LP_MS=500;           // 길게 눌러 진동이 오기까지의 시간(ms)
  var CROSS_DELAY_MS=150;  // 진동 후 십자선이 나타나기까지의 시간(ms) (진동 → 십자선 순서 맞춤)
  var PAN_GAIN=1.5;        // 손가락 이동 속도 배율
  var ZOOM_GAIN=1.5;       // 확대/축소 속도 배율
  var COMPARE_MODE='level'; // 'level'=그날 종목 가격과 그날 지수선(녹색 점선) 비교, 'daily'=그날 등락률끼리 비교
  var SAME_BAND_LEVEL=2;    // level: 가격 차이가 ±2% 이내면 '보합'
  var SAME_BAND_DAILY=0.3;  // daily: 등락률 차이가 ±0.3%p 이내면 '보합'
  var idxOn=true;          // 지수 비교선 표시 여부
  var RPAD_RATIO=0.08;     // 오른쪽 여백: 보이는 기간의 8%만큼 빈 칸 (120일 → 약 10칸, 폰 42일 → 약 3칸)
  var RPAD_MIN=3;          // 오른쪽 여백 최소 칸 수
  // 브라우저(앱 브라우저)가 두 손가락으로 화면 전체를 확대하지 못하게 막기
  try{
    var pd=window.parent.document, mv=pd.querySelector('meta[name=viewport]');
    if(mv)mv.setAttribute('content','width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no');
    if(!pd.getElementById('nozoom_css')){
      var st=pd.createElement('style'); st.id='nozoom_css';
      st.textContent='iframe{touch-action:pan-y !important;}'; pd.head.appendChild(st);
    }
  }catch(err){}
  gd.style.position='relative'; gd.style.touchAction='pan-y';
  gd.style.userSelect='none'; gd.style.webkitUserSelect='none';
  gd.style.webkitTouchCallout='none'; gd.style.webkitTapHighlightColor='transparent';
  var cats=gd.data[0].x, N=cats.length, MINS=Math.min(20,N), MAXS=Math.min(Math.max(120,initSpan),N);
  function clampSpan(v){return Math.round(Math.max(MINS,Math.min(MAXS,v)));}
  function clampI0(v,sp){return Math.round(Math.max(0,Math.min(N-sp,v)));}
  // 핸드폰(화면 폭 600px 이하)은 처음에 약 2달(42거래일)만 확대해서 보여줌. PC는 그대로.
  var PHONE_SPAN=42, ITa=IT;
  var isPhone=false;
  try{ isPhone=Math.min(window.screen&&window.screen.width||9999, window.innerWidth||9999)<=600; }catch(e0){}
  if(isPhone && initSpan>PHONE_SPAN && IDX && IDX2 && IDX2.length===N){
    try{   // 폰: 2달치 기준으로 미리 환산해 둔 지수선으로 교체(보이기/숨기기만 사용)
      IDX=IDX2; ITa=IT2; initSpan=PHONE_SPAN;
      var pr0=Plotly.restyle(gd,{visible:[false]},[IT]); if(pr0&&pr0.then)pr0.then(null,function(){});
      var pr1=Plotly.restyle(gd,{visible:[true]},[IT2]); if(pr1&&pr1.then)pr1.then(null,function(){});
    }catch(e1){}
  } else if(isPhone && initSpan>PHONE_SPAN){ initSpan=PHONE_SPAN; }
  var span=clampSpan(initSpan), i0=N-span;
  var pending=false, mode=null, timer=null, sx=0, sy=0, startI0=0, lastT=null, mdown=false, pinch=null, markersOn=null;
  var tStart=0, lastTap=0, lastTouch=0;

  function mk(css){var d=document.createElement('div');
    d.style.cssText='position:absolute;pointer-events:none;display:none;z-index:50;'+css;
    gd.appendChild(d);return d;}
  var lab='background:#222;color:#fff;font:11px sans-serif;padding:2px 4px;white-space:nowrap;';
  var vl=mk('width:0;border-left:1px solid #333;');
  var hl=mk('height:0;border-top:1px solid #333;');
  var dl=mk(lab+'transform:translateX(-50%);white-space:pre;text-align:center;line-height:1.3;');   // 날짜 + (가격 , 지수) 두 줄
  var pr=mk(lab+'width:50px;transform:translateY(-50%);');
  function f2(v){return Number(v).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});}
  var sm=mk('left:4px;top:6px;font:600 12px sans-serif;white-space:nowrap;');   // 지수 대비 요약
  var dt=mk('left:4px;top:24px;font:11px sans-serif;color:#666;white-space:nowrap;');   // 계산 근거(기준일부터의 수익률)
  var ig=mk('left:4px;top:40px;font:600 11px sans-serif;color:#00a651;white-space:nowrap;');   // 지수 값(녹색)
  if(IDX){sm.style.display='block'; dt.style.display='block'; ig.style.display='block';}
  var focus=N-1;           // 십자선이 가리키는 날짜(없으면 맨 오른쪽=최근)
  function sg(v){return (v>=0?'+':'')+v.toFixed(1)+'%';}
  function md(i){return cats[i].slice(5);}
  function pxs(v){return Number(v).toLocaleString('en-US',{minimumFractionDigits:DEC,maximumFractionDigits:DEC});}
  function updateSummary(){   // 선택한 날짜(없으면 최근 날짜)의 종목 가격과 지수로 강함/약함/보합 판단
    var D0=focus;
    try{
      var su=window.parent.document.querySelectorAll('[data-k=u]'), sl2=window.parent.document.querySelectorAll('[data-k=l]');
      for(var q=0;q<su.length;q++)su[q].textContent=pxs(S[2][D0]);
      for(var q2=0;q2<sl2.length;q2++)sl2[q2].textContent=pxs(S[3][D0]);}catch(e2){}
    if(!IDX)return;
    var R=IDXRAW||IDX, D=focus, d, band, row2, unit;
    if(COMPARE_MODE==='daily' && D>=1){
      var sr=(S[0][D]/S[0][D-1]-1)*100, ir=(R[D]/R[D-1]-1)*100;
      d=sr-ir; band=SAME_BAND_DAILY; unit='%p ';
      row2='그날 등락 종목 '+sg(sr)+' · '+IDXNAME+' '+sg(ir);
    } else {
      d=(S[0][D]/IDX[D]-1)*100; band=SAME_BAND_LEVEL; unit='% ';     // 종목 가격이 지수선보다 몇 % 위/아래인지
      // 시가·저가·종가는 전일 종가 대비, 지수선은 전일 지수선 대비 등락률
      var prevClose = D > 0 ? S[0][D-1] : NaN;
      var prevIndex = D > 0 ? IDX[D-1] : NaN;
      function coloredValue(label, value, base, pctSourceValue){
        // pctSourceValue가 있으면 실제 원본 지수값으로 등락률을 계산
        var compareValue = (pctSourceValue !== undefined) ? pctSourceValue : value;
        var pct = (isFinite(base) && base !== 0 && isFinite(compareValue))
                    ? (compareValue / base - 1) * 100 : NaN;
        var color = !isFinite(pct) || Math.abs(pct) < 0.0000001 ? '#000000' : (pct > 0 ? '#e5484d' : '#3b6fd4');
        var pctText = isFinite(pct) ? (pct > 0 ? '+' : '') + pct.toFixed(2) + '%' : '—';
        return '<span style="color:'+color+'">'+label+' '+pxs(value)+' ('+pctText+')</span>';
      }
      var prevRawIndex = D > 0 ? R[D-1] : NaN;
      row2 = '<span style="color:#000000">시가 '+pxs(OPEN[D])+'</span> · ' +
       '<span style="color:#000000">저가 '+pxs(LOW[D])+'</span> · ' +
       '<span style="color:#000000">종가 '+pxs(S[0][D])+'</span> · ' +
       coloredValue('지수', R[D], prevRawIndex);
    }
    var up=d>band, dn=d<-band;
    sm.textContent=IDXNAME+' 대비 '+(d>=0?'+':'')+d.toFixed(1)+unit+(up?'강함':(dn?'약함':'보합'));
    sm.style.color=up?'#e5484d':(dn?'#3b6fd4':'#666');
    if(COMPARE_MODE==='daily'){
      dt.textContent=row2;
    } else {
      dt.innerHTML=row2;
    }
    ig.textContent='● '+cats[D]+'  '+IDXNAME+' '+f2(R[D]);
  }
  var tg=null;
  var dbg=mk('left:4px;top:66px;font:11px sans-serif;color:#c00;background:rgba(255,255,255,.92);padding:2px 4px;max-width:92%;z-index:70;');
  function showErr(e){dbg.style.display='block'; dbg.textContent='차트 오류: '+(e&&e.message?e.message:e);}
  function hide(){[vl,hl,dl,pr].forEach(function(e){e.style.display='none';}); if(focus!==N-1){focus=N-1; updateSummary();}}

  function fmt(c){return c.slice(5).replace('-','.');}

  // 보이는 구간의 최고/최저에 맞춰 가격 범위를 즉시 변경 + 지수선을 맨 왼쪽 종가에 맞춰 환산
  function draw(){
    var a=Math.max(0,Math.floor(i0)), b=Math.min(N-1,Math.ceil(i0+span)-1), i, k, v;
    var lo=Infinity, hi=-Infinity, bl=0, bh=0;
    for(k=0;k<S.length;k++){for(i=a;i<=b;i++){v=S[k][i]; if(v<lo)lo=v; if(v>hi)hi=v;}}
    if(IDX&&idxOn){for(i=a;i<=b;i++){v=IDX[i]; if(v<lo)lo=v; if(v>hi)hi=v;}}
    if(HIGH&&LOW){for(i=a;i<=b;i++){if(HIGH[i]>hi)hi=HIGH[i]; if(LOW[i]<lo)lo=LOW[i];}}   // 캔들 고가·저가
    for(i=a;i<=b;i++){v=B[i]; if(v<bl)bl=v; if(v>bh)bh=v;}
    if(!isFinite(lo)||!isFinite(hi)||!isFinite(bl)||!isFinite(bh))throw new Error('가격 범위 계산 오류');
    var pad=(hi-lo)*0.06||1;
    var step=Math.max(3,Math.round(span/4)), tv=[], tt=[], idx;
    for(idx=N-1;idx>=a;idx-=step){tv.push(cats[idx]); tt.push(fmt(cats[idx]));}
    var rp=Math.max(RPAD_MIN,Math.round(span*RPAD_RATIO));   // 오늘 캔들이 오른쪽 끝에 붙지 않게 빈 칸 추가
    var xr=[i0-0.5,i0+span-0.5+rp];
    var lay={'xaxis.range':xr,'xaxis2.range':xr,
      'yaxis.range':[lo-pad,hi+pad],'yaxis2.range':[bl*1.1,(bh*1.1)||1],
      'xaxis.tickvals':tv,'xaxis.ticktext':tt,'xaxis2.tickvals':tv,'xaxis2.ticktext':tt};
    var p=Plotly.relayout(gd,lay);
    if(p&&p.then)p.then(null,showErr);
  }
  function render(){
    pending=false;
    try{ draw(); }
    catch(e){                                // 지수 부분에서 문제가 생겨도 차트는 계속 그리도록 지수 기능만 끔
      showErr(e);
      if(IDX){
        IDX=null; sm.style.display='none'; if(tg)tg.style.display='none';
        try{ Plotly.restyle(gd,{visible:[false]},[ITa]); }catch(e1){}
        try{ draw(); }catch(e2){ showErr(e2); }
      }
    }
  }
  function schedule(){if(!pending){pending=true;requestAnimationFrame(render);}}

  function move(cx,cy){
    var r=gd.getBoundingClientRect(), x=cx-r.left, y=cy-r.top;
    var fl=gd._fullLayout, s=fl._size, xa=fl.xaxis, ya=fl.yaxis, yb=fl.yaxis2;
    if(x<s.l||x>s.l+s.w||y<s.t||y>s.t+s.h){hide();return;}
    var lo=xa.range[0], hi=xa.range[1];
    var idx=Math.round((x-xa._offset)/xa._length*(hi-lo)+lo);
    idx=Math.max(0,Math.min(N-1,idx));
    focus=idx; updateSummary();
    var xs=xa._offset+(idx-lo)/(hi-lo)*xa._length;
    if(!isFinite(xs))return;
    var ax=(y>=yb._offset)?yb:ya;
    var v=ax.range[1]-(y-ax._offset)/ax._length*(ax.range[1]-ax.range[0]);
    var txt=pxs(v);
    vl.style.cssText+=';display:block;left:'+xs+'px;top:'+s.t+'px;height:'+s.h+'px;';
    hl.style.cssText+=';display:block;left:'+s.l+'px;top:'+y+'px;width:'+s.w+'px;';
    var showI=!!(IDX&&IDXRAW&&idxOn);
    // 예) 2026-10-20 (줄바꿈) (11,700 , 5,600.00)  → 날짜 / (그날 종목 가격 , 그날 지수)
    dl.textContent=cats[idx]+'\\n('+pxs(S[0][idx])+(showI?' , '+f2(IDXRAW[idx]):'')+')';
    dl.style.display='block'; dl.style.top=(ya._offset+ya._length+2)+'px';
    var half=(dl.offsetWidth||0)/2, tot=fl.width||gd.clientWidth||0;
    dl.style.left=(tot?Math.max(half,Math.min(tot-half,xs)):xs)+'px';        // 화면 밖으로 나가지 않게
    pr.style.display='block'; pr.style.left=(s.l+s.w+2)+'px'; pr.style.top=y+'px'; pr.textContent=txt;
  }
  function pxToBars(dx){return dx/gd._fullLayout._size.w*span;}
  function fracOf(clientX){
    var r=gd.getBoundingClientRect(), s=gd._fullLayout._size;
    return Math.max(0,Math.min(1,(clientX-r.left-s.l)/s.w));
  }
  function dist(t){var dx=t[0].clientX-t[1].clientX, dy=t[0].clientY-t[1].clientY; return Math.sqrt(dx*dx+dy*dy);}
  function resetView(){                       // 더블클릭/더블탭: 처음 상태로
    span=clampSpan(initSpan); i0=N-span; pinch=null; if(mode==='cross')mode=null; hide(); schedule();
  }
  function zoomBy(factor){                    // 화면 가운데 기준 확대/축소
    var c=i0+span/2, ns=clampSpan(span*factor);
    if(ns===span)ns=clampSpan(span+(factor<1?-1:1));
    span=ns; i0=clampI0(c-ns/2,ns); hide(); schedule();
  }

  // ── 확대/축소 버튼 (두 손가락 확대가 막히는 폰용) ──
  function btn(txt,right,fn,w,fs){
    var b=document.createElement('div'); b.textContent=txt;
    b.style.cssText='position:absolute;top:2px;right:'+right+'px;width:'+(w||28)+'px;height:22px;line-height:22px;text-align:center;'
      +'font:600 '+(fs||15)+'px sans-serif;background:#f3f3f3;border:1px solid #ccc;border-radius:4px;z-index:60;cursor:pointer;'
      +'touch-action:manipulation;user-select:none;-webkit-user-select:none;';
    ['touchstart','touchmove','touchend','mousedown','mousemove','dblclick'].forEach(function(n){
      b.addEventListener(n,function(e){e.stopPropagation();},{passive:true});});
    b.addEventListener('click',function(e){e.stopPropagation();fn();});
    gd.appendChild(b);
    return b;
  }
  btn('＋',92,function(){zoomBy(0.6);});
  btn('－',62,function(){zoomBy(1/0.6);});
  if(IDX){                                  // 지수 비교선 켜기/끄기 버튼 (버튼 글자 = 지수 이름)
    tg=btn(IDXNAME,124,function(){idxOn=!idxOn; paintTg(); try{Plotly.restyle(gd,{visible:[idxOn]},[ITa]);}catch(e){showErr(e);} schedule();},46,12);
    var paintTg=function(){tg.style.color=idxOn?'#007a3d':'#999'; tg.style.background=idxOn?'#e8f8ef':'#f3f3f3'; tg.style.borderColor=idxOn?'#00a651':'#ccc';};
    paintTg();
  }

  // ── 손가락 ──
  // 바로 좌우로 밀기=기간 이동(한 칸씩 딱딱하게) / 길게 눌러 진동=십자선 / 두 손가락=확대·축소
  gd.addEventListener('touchstart',function(e){
    clearTimeout(timer); lastTouch=Date.now();
    if(e.touches.length>=2){
      hide(); mode='pinch';
      pinch={d:dist(e.touches),span:span,anchor:i0+fracOf((e.touches[0].clientX+e.touches[1].clientX)/2)*span};
      return;
    }
    hide();
    var t=e.touches[0]; lastT=t; sx=t.clientX; sy=t.clientY; startI0=i0; mode=null; tStart=Date.now();
    timer=setTimeout(function(){
      if(mode!==null)return;
      mode='arming'; if(navigator.vibrate)navigator.vibrate(40);       // ① 진동
      timer=setTimeout(function(){                                      // ② 잠깐 뒤 십자선 표시 후 따라다님
        if(mode==='arming'){mode='cross'; move(lastT.clientX,lastT.clientY);}
      },CROSS_DELAY_MS);
    },LP_MS);
  },{passive:true});
  gd.addEventListener('touchmove',function(e){
    if(mode==='pinch'){
      if(e.touches.length===2&&pinch){
        e.preventDefault();
        var f=fracOf((e.touches[0].clientX+e.touches[1].clientX)/2);
        var ns=clampSpan(pinch.span*Math.pow(pinch.d/dist(e.touches),ZOOM_GAIN));
        span=ns; i0=clampI0(pinch.anchor-f*ns,ns); schedule();
      }
      return;
    }
    if(e.touches.length!==1)return;
    var t=e.touches[0]; lastT=t;
    if(mode==='arming'){e.preventDefault(); return;}
    if(mode==='cross'){e.preventDefault(); move(t.clientX,t.clientY); return;}
    if(mode==='pan'){e.preventDefault(); i0=clampI0(startI0-pxToBars(t.clientX-sx)*PAN_GAIN,span); schedule(); return;}
    if(mode==='scroll')return;
    var dx=t.clientX-sx, dy=t.clientY-sy;
    if(Math.abs(dx)>6||Math.abs(dy)>6){
      clearTimeout(timer);
      if(Math.abs(dx)>Math.abs(dy)){mode='pan'; e.preventDefault(); i0=clampI0(startI0-pxToBars(dx)*PAN_GAIN,span); schedule();}
      else mode='scroll';
    }
  },{passive:false});
  function touchEnd(e){
    clearTimeout(timer); lastTouch=Date.now();
    if(e.touches&&e.touches.length>0)return;      // 아직 손가락이 남아 있음
    pinch=null;
    if(mode===null&&e.type==='touchend'&&Date.now()-tStart<300){   // 짧게 톡 → 두 번이면 기본값
      var now=Date.now();
      if(now-lastTap<350){resetView(); lastTap=0;} else lastTap=now;
    }
    if(mode!=='cross')mode=null;                   // 십자선은 손을 떼도 남고, 다음 터치 때 사라짐
  }
  gd.addEventListener('touchend',touchEnd);
  gd.addEventListener('touchcancel',touchEnd);
  gd.addEventListener('contextmenu',function(e){e.preventDefault();});

  // ── PC: 마우스를 누른 채 끌면 이동, 그냥 움직이면 십자선, Ctrl+휠=확대/축소 ──
  function emu(e){return (Date.now()-lastTouch<1000)||(e.sourceCapabilities&&e.sourceCapabilities.firesTouchEvents);}
  gd.addEventListener('mousedown',function(e){if(emu(e))return; mdown=true; sx=e.clientX; startI0=i0; e.preventDefault();});
  window.addEventListener('mousemove',function(e){
    if(!mdown)return;
    i0=clampI0(startI0-pxToBars(e.clientX-sx),span); schedule();
  },true);
  window.addEventListener('mouseup',function(){mdown=false;},true);
  gd.addEventListener('mousemove',function(e){if(emu(e))return; move(e.clientX,e.clientY);});
  gd.addEventListener('mouseleave',function(e){if(emu(e))return; hide();});
  gd.addEventListener('dblclick',function(e){resetView();});
  gd.addEventListener('wheel',function(e){
    if(!e.ctrlKey)return; e.preventDefault();
    var f=fracOf(e.clientX), anchor=i0+f*span;
    var ns=clampSpan(span*(e.deltaY>0?1.23:0.81));
    if(ns===span)ns=clampSpan(span+(e.deltaY>0?1:-1));
    span=ns; i0=clampI0(anchor-f*ns,ns); schedule();
  },{passive:false});
  updateSummary();
  render();
}
"""


def finish(fig, low_label):
    m = len(dates)
    pos = sorted({m - 1 - int(round(i * (m - 1) / 3.6)) for i in range(4)})
    fig.update_xaxes(type="category", tickmode="array",
                     tickvals=[dates[i] for i in pos],
                     ticktext=[pd.to_datetime(dates[i]).strftime("%m.%d") for i in pos],
                     tickangle=0, tickfont=dict(size=11, color="#555"),
                     showgrid=True, gridcolor="#f0b8b8", gridwidth=1,
                     showline=False, zeroline=False)
    fig.update_xaxes(showticklabels=True, row=1, col=1)    # 날짜는 위 차트 아래(두 차트 사이)에 표시
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_yaxes(side="right", tickformat=(",.2f" if US and last < 100 else ",.0f"), tickfont=dict(size=11, color="#555"),
                     showgrid=True, gridcolor="#eeeeee", zeroline=False, showline=False, ticks="")
    fig.update_yaxes(nticks=4, row=2, col=1)
    tagc = "#e5484d" if chg > 0 else "#3b6fd4"
    # 현재가 태그(오른쪽 축)
    fig.add_annotation(xref="paper", x=1.0, xanchor="left", yref="y", y=yv[-1], showarrow=False,
                       text=f"{last:,.{DEC}f}<br>{pct:+.2f}%", align="left", borderpad=2,
                       font=dict(color="white", size=11), bgcolor=tagc)
    # 범례(차트 위 한 줄)
    fig.add_annotation(xref="paper", yref="paper", x=0.0, y=1.0, xanchor="left", yanchor="bottom",
                       showarrow=False, font=dict(size=11),
                       text=('<span style="color:#e5484d">▮</span><span style="color:#3b6fd4">▮</span> <span style="color:#111111">수정종가 캔들</span>  '
                             '<span style="color:#111111">━ 곡선</span>  '
                             '<span style="color:#1a56db">━ 상한</span>  '
                             '<span style="color:#e5484d">━ 하한</span>'))
    # 아래 차트 이름
    fig.add_annotation(xref="paper", yref="paper", x=0.01, y=fig.layout.yaxis2.domain[1],
                       xanchor="left", yanchor="top", showarrow=False,
                       text=low_label, font=dict(size=11, color="#555"))
    fig.update_layout(height=620, margin=dict(l=4, r=58, t=(78 if idx_arr is not None else 26), b=22),
                      showlegend=False, bargap=0.05, hovermode=False, dragmode=False,
                      xaxis_rangeslider_visible=False, xaxis2_rangeslider_visible=False,
                      plot_bgcolor="white", paper_bgcolor="white")
    return fig


def show(fig, div_id, bars):
    """좌우 이동 + 길게 눌러 십자선 기능을 얹어 차트를 표시"""
    def pack(v):
        return np.round(np.asarray(v, dtype=float), 2).tolist()
    series = [pack(v) for v in (yv, curvev, upv, lov)]
    dump = lambda o: json.dumps(o, separators=(",", ":"))
    idx_json = dump(pack(idx_scaled)) if idx_scaled is not None else "null"
    raw_json = dump(pack(idx_arr)) if idx_arr is not None else "null"
    open_json = dump(pack(ov))
    low_json = dump(pack(lv))
    idx2_json = dump(pack(idx_scaled_p)) if idx_scaled_p is not None else "null"
    post = "installCrosshair(document.getElementById('{plot_id}'), %d, %s, %s, %s, %d, %s, %s, %s, %s, %d, %s, %d, %s, %d);" % (
        init_span, dump(series), dump(pack(bars)),
        idx_json, (4 if idx_scaled is not None else -1), json.dumps(idx_name),
        raw_json, open_json, low_json, n - init_span, idx2_json, 5, dump(pack(hv)), DEC)
    html = "<script>" + CROSSHAIR_JS + "</script>" + fig.to_html(
        full_html=False, include_plotlyjs="cdn", div_id=div_id,
        config={"staticPlot": True, "responsive": True},
        post_script=post)
    components.html(html, height=640)


fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.06)
base_traces(fig)
cv = curvev - curvev.min()
fig.add_trace(go.Bar(x=dates, y=cv, marker_color=direction_colors(curvev),
                     name="곡선추세 방향", showlegend=False), row=2, col=1)
show(finish(fig, "곡선추세 방향"), "chart_trend", cv)
status.empty()
if idx_arr is None:
    st.caption("※ 지수 데이터를 불러오지 못해 비교선 없이 표시 중입니다.")
elif idx_note:
    st.caption(idx_note)
