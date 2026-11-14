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
    """종목명 검색용 목록 (주식 + ETF)"""
    frames = []
    for market in ("KRX", "ETF/KR"):
        try:
            d = fdr.StockListing(market)
            cc = "Code" if "Code" in d.columns else "Symbol"
            d = d.rename(columns={cc: "Code"})
            d["Market"] = d["Market"] if "Market" in d.columns else ("ETF" if market == "ETF/KR" else "")
            frames.append(d[["Code", "Name", "Market"]])
        except Exception:
            pass
    if not frames:
        return pd.DataFrame(columns=["Code", "Name", "nn", "nk", "mkt"])
    out = pd.concat(frames).dropna().drop_duplicates("Code").reset_index(drop=True)
    out["Code"] = out["Code"].astype(str)
    out["mkt"] = out["Market"].astype(str).str.upper().map(lambda x: "KOSDAQ" if "KOSDAQ" in x else "KOSPI")
    out["nn"] = out["Name"].map(norm)                       # 원래 이름(정규화)
    out["nk"] = out["Name"].map(lambda x: norm(to_ko(x)))   # 한글 발음 이름(정규화)
    return out


def search(lst: pd.DataFrame, q: str):
    """('exact', 1행) 바로 표시 / ('list', 여러행) 연관 종목 선택 / ('none', 빈표)"""
    nq, nqk = norm(q), norm(to_ko(q))
    if not nq or lst.empty:
        return "none", lst.iloc[0:0]
    exact = lst[(lst["nn"] == nq) | (lst["nk"] == nq) | (lst["nk"] == nqk)]
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
lst = listing()
names = dict(zip(lst["Code"], lst["Name"]))
mkts = dict(zip(lst["Code"], lst["mkt"]))        # 코스피/코스닥 구분 (ETF는 코스피로 처리)
names.setdefault(DEFAULT_CODE, "KODEX 200TR")   # 종목 목록을 못 불러와도 이름 표시
if "code" not in st.session_state:
    st.session_state.code = DEFAULT_CODE

name_ph = st.empty()
q = st.text_input("종목 검색", placeholder="🔍 종목코드 또는 종목명 (예: 278530, 삼성전자, 네이버)",
                  label_visibility="collapsed").strip()
if q:
    qu = q.upper()
    if qu in names or (len(q) == 6 and q.isdigit()):
        st.session_state.code = qu                                  # 종목코드 → 바로 표시
    elif lst.empty:
        st.caption("종목 목록을 불러오지 못했습니다. 6자리 종목코드로 검색해 주세요.")
    else:
        kind, m = search(lst, q)
        if kind == "exact":                                         # 이름이 정확히 맞으면 바로 표시
            st.session_state.code = m.iloc[0]["Code"]
        elif kind == "list":                                        # 일부만 입력 → 연관 종목에서 선택
            opts = [f"{r.Name} ({r.Code})" for r in m.head(30).itertuples()]
            sel = st.selectbox("연관 종목", opts, index=None, key=f"sel_{norm(q)}",
                               placeholder=f"연관 종목 {len(m)}개 — 여기서 선택", label_visibility="collapsed")
            if sel:
                st.session_state.code = sel.rsplit("(", 1)[1].rstrip(")")
        else:
            st.caption("검색 결과가 없습니다.")
code = st.session_state.code
name = names.get(code, code)
name_ph.markdown(
    f'<div style="font-size:20px;font-weight:700;line-height:1.3">{html.escape(str(name))} '
    f'<span style="font-size:13px;color:#888;font-weight:400">{html.escape(code)}</span></div>',
    unsafe_allow_html=True)

@st.cache_data(ttl=600)
def load(code: str, days: int) -> pd.DataFrame:
    start = (pd.Timestamp.today() - pd.Timedelta(days=int(days * 1.6) + 10)).strftime("%Y-%m-%d")
    return fdr.DataReader(code, start).tail(int(days))


@st.cache_data(ttl=600, show_spinner=False)
def load_index(sym: str, days: int) -> pd.DataFrame:
    """지수 시세. 10초 안에 못 받으면 예외를 내서(캐시되지 않음) 차트가 멈추지 않게 함"""
    start = (pd.Timestamp.today() - pd.Timedelta(days=int(days * 1.6) + 10)).strftime("%Y-%m-%d")
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fdr.DataReader, sym, start)
    try:
        return fut.result(timeout=10).tail(int(days))
    finally:
        ex.shutdown(wait=False)


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


calc = max(int(calc), 180)
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
D = n
init_span = int(min(disp, D))
sl = slice(n - D, n)
dates = df.index.strftime("%Y-%m-%d")[sl]
yv, linev, curvev, upv, lov = y[sl], line[sl], curve[sl], upper[sl], lower[sl]

# ───────────── 지수(코스피/코스닥) 비교선 ─────────────
mkt = mkts.get(code, "KOSPI")
idx_sym, idx_name = ("KQ11", "코스닥") if mkt == "KOSDAQ" else ("KS11", "코스피")
idx_arr = None
try:
    _s = load_index(idx_sym, calc)["Close"].reindex(df.index).ffill().bfill()
    if len(_s) == n and np.isfinite(_s.to_numpy(dtype=float)).all():
        idx_arr = _s.to_numpy(dtype=float)
except Exception:
    idx_arr = None

# ───────────── 상단 시세 헤더 ─────────────
last, prev = y[-1], y[-2]
chg = last - prev
pct = chg / prev * 100
col = "#e5484d" if chg > 0 else ("#3b6fd4" if chg < 0 else "#333333")
arrow = "▲" if chg > 0 else ("▼" if chg < 0 else "-")
vol = df["Volume"].iloc[-1]
st.markdown(
    '<div style="display:flex;align-items:flex-end;gap:14px;flex-wrap:wrap">'
    f'<span style="font-size:42px;font-weight:700;color:{col};line-height:1">{last:,.0f}</span>'
    f'<span style="font-size:17px;color:{col};padding-bottom:4px">{arrow} {abs(chg):,.0f} &nbsp; {pct:+.2f}%</span>'
    '</div>'
    f'<div style="font-size:13px;color:#888;margin-top:4px">거래량 {vol:,.0f}</div>',
    unsafe_allow_html=True)

# ───────────── 차트 ─────────────
def direction_colors(v):
    d = np.diff(v, prepend=v[1])
    d[0] = d[1]
    return np.where(d >= 0, "#e5484d", "#3b6fd4")  # 상승=빨강, 하락=파랑


def base_traces(fig):
    fig.add_trace(go.Scatter(x=dates, y=yv, name="수정종가", line=dict(color="#e5484d", width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=linev, name="직선추세", line=dict(color="magenta", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=curvev, name="곡선추세", line=dict(color="royalblue", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=upv, name="곡선추세상한", line=dict(color="navy", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=lov, name="곡선추세하한", line=dict(color="navy", width=1)), row=1, col=1)
    if idx_arr is not None:      # 5번 트레이스: 지수(맨 왼쪽 종가에 맞춰 환산한 점선)
        a0 = n - init_span
        fig.add_trace(go.Scatter(x=dates, y=idx_arr * (yv[a0] / idx_arr[a0]), name=idx_name,
                                 line=dict(color="#ff8c00", width=1.3, dash="dot")), row=1, col=1)


CROSSHAIR_JS = """
function installCrosshair(gd, initSpan, S, B, IDX, IT, IDXNAME){
  var LP_MS=500;           // 길게 눌러 진동이 오기까지의 시간(ms)
  var CROSS_DELAY_MS=150;  // 진동 후 십자선이 나타나기까지의 시간(ms) (진동 → 십자선 순서 맞춤)
  var PAN_GAIN=1.5;        // 손가락 이동 속도 배율
  var ZOOM_GAIN=1.5;       // 확대/축소 속도 배율
  var SAME_BAND=2;         // 종목-지수 수익률 차이가 ±이 값(%p) 이내면 '비슷'
  var idxOn=true;          // 지수 비교선 표시 여부
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
  var span=clampSpan(initSpan), i0=N-span;
  var pending=false, mode=null, timer=null, sx=0, sy=0, startI0=0, lastT=null, mdown=false, pinch=null, markersOn=null;
  var tStart=0, lastTap=0, lastTouch=0;

  function mk(css){var d=document.createElement('div');
    d.style.cssText='position:absolute;pointer-events:none;display:none;z-index:50;'+css;
    gd.appendChild(d);return d;}
  var lab='background:#222;color:#fff;font:11px sans-serif;padding:2px 4px;white-space:nowrap;';
  var vl=mk('width:0;border-left:1px solid #333;');
  var hl=mk('height:0;border-top:1px solid #333;');
  var dl=mk(lab+'transform:translateX(-50%);');
  var pr=mk(lab+'width:50px;transform:translateY(-50%);');
  var sm=mk('left:4px;top:6px;font:600 12px sans-serif;white-space:nowrap;');   // 지수 대비 요약
  if(IDX)sm.style.display='block';
  function hide(){[vl,hl,dl,pr].forEach(function(e){e.style.display='none';});}

  function fmt(c){return c.slice(5).replace('-','.');}

  // 보이는 구간의 최고/최저에 맞춰 가격 범위를 즉시 변경 + 지수선을 맨 왼쪽 종가에 맞춰 환산
  function render(){
    pending=false;
    var a=Math.max(0,Math.floor(i0)), b=Math.min(N-1,Math.ceil(i0+span)-1), i, k, v;
    var lo=Infinity, hi=-Infinity, bl=0, bh=0, iv=null;
    if(IDX){var ratio=S[0][a]/IDX[a]; iv=new Array(N); for(i=0;i<N;i++)iv[i]=IDX[i]*ratio;}
    for(k=0;k<S.length;k++){for(i=a;i<=b;i++){v=S[k][i]; if(v<lo)lo=v; if(v>hi)hi=v;}}
    if(iv&&idxOn){for(i=a;i<=b;i++){v=iv[i]; if(v<lo)lo=v; if(v>hi)hi=v;}}
    for(i=a;i<=b;i++){v=B[i]; if(v<bl)bl=v; if(v>bh)bh=v;}
    var pad=(hi-lo)*0.06||1;
    var step=Math.max(3,Math.round(span/4)), tv=[], tt=[], idx;
    for(idx=N-1;idx>=a;idx-=step){tv.push(cats[idx]); tt.push(fmt(cats[idx]));}
    var xr=[i0-0.5,i0+span-0.5];
    var lay={'xaxis.range':xr,'xaxis2.range':xr,
      'yaxis.range':[lo-pad,hi+pad],'yaxis2.range':[bl*1.1,(bh*1.1)||1],
      'xaxis.tickvals':tv,'xaxis.ticktext':tt,'xaxis2.tickvals':tv,'xaxis2.ticktext':tt};
    if(IDX){
      var sr=(S[0][b]/S[0][a]-1)*100, ir=(IDX[b]/IDX[a]-1)*100, d=sr-ir;
      var up=d>SAME_BAND, dn=d<-SAME_BAND;
      sm.textContent=IDXNAME+' 대비 '+(d>=0?'+':'')+d.toFixed(1)+'%p '+(up?'강함':(dn?'약함':'비슷'));
      sm.style.color=up?'#e5484d':(dn?'#3b6fd4':'#666');
      Plotly.update(gd, idxOn?{y:[iv],visible:[true]}:{visible:[false]}, lay, [IT]);
    } else {
      Plotly.relayout(gd,lay);
    }
    var mk2=span<=40;                        // 확대했을 때는 일별 점을 표시
    if(mk2!==markersOn){markersOn=mk2; Plotly.restyle(gd,{mode:mk2?'lines+markers':'lines','marker.size':4},[0]);}
  }
  function schedule(){if(!pending){pending=true;requestAnimationFrame(render);}}

  function move(cx,cy){
    var r=gd.getBoundingClientRect(), x=cx-r.left, y=cy-r.top;
    var fl=gd._fullLayout, s=fl._size, xa=fl.xaxis, ya=fl.yaxis, yb=fl.yaxis2;
    if(x<s.l||x>s.l+s.w||y<s.t||y>s.t+s.h){hide();return;}
    var lo=xa.range[0], hi=xa.range[1];
    var idx=Math.round((x-xa._offset)/xa._length*(hi-lo)+lo);
    idx=Math.max(0,Math.min(N-1,idx));
    var xs=xa._offset+(idx-lo)/(hi-lo)*xa._length;
    var ax=(y>=yb._offset)?yb:ya;
    var v=ax.range[1]-(y-ax._offset)/ax._length*(ax.range[1]-ax.range[0]);
    var txt=Math.round(v).toLocaleString('en-US');
    vl.style.cssText+=';display:block;left:'+xs+'px;top:'+s.t+'px;height:'+s.h+'px;';
    hl.style.cssText+=';display:block;left:'+s.l+'px;top:'+y+'px;width:'+s.w+'px;';
    dl.style.display='block'; dl.style.left=xs+'px';
    dl.style.top=(ya._offset+ya._length+2)+'px'; dl.textContent=cats[idx];
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
    var tg=btn(IDXNAME,124,function(){idxOn=!idxOn; paintTg(); schedule();},46,12);
    var paintTg=function(){tg.style.color=idxOn?'#e07000':'#999'; tg.style.background=idxOn?'#fff3e0':'#f3f3f3'; tg.style.borderColor=idxOn?'#ff8c00':'#ccc';};
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
    fig.update_yaxes(side="right", tickformat=",.0f", tickfont=dict(size=11, color="#555"),
                     showgrid=True, gridcolor="#eeeeee", zeroline=False, showline=False, ticks="")
    fig.update_yaxes(nticks=4, row=2, col=1)
    tagc = "#e5484d" if chg > 0 else "#3b6fd4"
    # 현재가 태그(오른쪽 축)
    fig.add_annotation(xref="paper", x=1.0, xanchor="left", yref="y", y=yv[-1], showarrow=False,
                       text=f"{last:,.0f}<br>{pct:+.2f}%", align="left", borderpad=2,
                       font=dict(color="white", size=11), bgcolor=tagc)
    # 범례(차트 위 한 줄)
    fig.add_annotation(xref="paper", yref="paper", x=0.0, y=1.0, xanchor="left", yanchor="bottom",
                       showarrow=False, font=dict(size=11),
                       text=('<span style="color:#e5484d">━ 수정종가</span>  '
                             '<span style="color:#e020e0">━ 직선</span>  '
                             '<span style="color:#4169e1">━ 곡선</span>  '
                             '<span style="color:#000080">━ 상·하한</span>'))
    # 아래 차트 이름
    fig.add_annotation(xref="paper", yref="paper", x=0.01, y=fig.layout.yaxis2.domain[1],
                       xanchor="left", yanchor="top", showarrow=False,
                       text=low_label, font=dict(size=11, color="#555"))
    fig.update_layout(height=620, margin=dict(l=4, r=58, t=(50 if idx_arr is not None else 26), b=22),
                      showlegend=False, bargap=0.05, hovermode=False, dragmode=False,
                      plot_bgcolor="white", paper_bgcolor="white")
    return fig


def show(fig, div_id, bars):
    """좌우 이동 + 길게 눌러 십자선 기능을 얹어 차트를 표시"""
    series = [np.asarray(v, dtype=float).tolist() for v in (yv, linev, curvev, upv, lov)]
    idx_json = json.dumps(idx_arr.tolist()) if idx_arr is not None else "null"
    post = "installCrosshair(document.getElementById('{plot_id}'), %d, %s, %s, %s, %d, %s);" % (
        init_span, json.dumps(series), json.dumps(np.asarray(bars, dtype=float).tolist()),
        idx_json, (5 if idx_arr is not None else -1), json.dumps(idx_name))
    html = "<script>" + CROSSHAIR_JS + "</script>" + fig.to_html(
        full_html=False, include_plotlyjs="cdn", div_id=div_id,
        config={"staticPlot": True, "responsive": True},
        post_script=post)
    components.html(html, height=640)


tab1, tab2 = st.tabs(["추세전환차트", "추세사이클차트"])

with tab1:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.06)
    base_traces(fig)
    cv = curvev - curvev.min()
    fig.add_trace(go.Bar(x=dates, y=cv, marker_color=direction_colors(curvev),
                         name="곡선추세 방향", showlegend=False), row=2, col=1)
    show(finish(fig, "곡선추세 방향"), "chart_trend", cv)
    st.caption("조작: 좌우로 밀기=기간 이동 · 길게 누름(0.5초, 진동)=십자선 · 두 손가락 벌리기/모으기 또는 ＋/－=확대·축소 · 두 번 톡(더블클릭)=기본값 · 지수 버튼=비교선(점선) 켜기/끄기. "
               "하단 봉: 곡선추세가 상승 중이면 빨강, 하락 중이면 파랑.")

with tab2:
    P, r2, cyc = fit_cycle(resid)
    cyc = (cyc - cyc.mean())[sl]
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.06)
    base_traces(fig2)
    fig2.add_trace(go.Bar(x=dates, y=cyc, marker_color=direction_colors(cyc),
                          name="싸이클", showlegend=False), row=2, col=1)
    show(finish(fig2, "싸이클"), "chart_cycle", cyc)
    st.caption(f"추정 최적 주기 약 {P:.1f}거래일 · 잔차 설명력(R²) {r2 * 100:.0f}%. "
               "싸이클의 상승/하락은 추세의 상승/하락과 무관하며, 추세선보다 설명력이 낮은 보조 지표입니다.")

if idx_arr is None:
    st.caption("※ 지수 데이터를 불러오지 못해 비교선 없이 표시 중입니다.")
st.caption("HP필터 특성상 새 데이터가 추가되면 과거 곡선도 다시 계산되어 모양이 바뀝니다. "
           "미래에셋 차트의 근사 재현이며, 투자 판단은 본인 책임입니다.")
