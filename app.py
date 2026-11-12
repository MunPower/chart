import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import FinanceDataReader as fdr
import json
import streamlit.components.v1 as components

st.set_page_config(page_title="추세전환/사이클 차트", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
.block-container{padding:0.6rem 0.4rem 1rem 0.4rem;max-width:100%;}
[data-testid="stToolbar"],footer{display:none !important;}
</style>""", unsafe_allow_html=True)

# ───────────── 설정 (평소에는 접어 둠) ─────────────
with st.expander("⚙️ 설정", expanded=False):
    c1, c2, c3 = st.columns(3)
    code = c1.text_input("종목코드", "278530")
    disp = c2.number_input("보이는 기간(거래일, 60~120)", 60, 120, 120, 10,
                           help="처음 열 때 보이는 기간입니다. 차트에서 두 손가락으로 벌리거나 ＋/－ 버튼으로 20~120일까지 조절할 수 있습니다.")
    calc = c3.number_input("계산 기간(거래일)", 120, 2000, 500, 50,
                           help="직선/곡선 계산에 쓰는 기간. 직선추세 기울기를 맞출 때 조절하세요.")
    c4, c5, c6 = st.columns(3)
    lam = c4.select_slider("HP필터 λ (클수록 곡선이 완만)",
                           options=[5e3, 6e3, 7e3, 8e3, 9e3, 1e4, 2e4, 5e4, 1e5, 3e5, 1e6], value=8e3,
                           format_func=lambda v: f"{v:,.0f}")
    kmul = c5.slider("상·하한 폭 (σ 배수, 90%=1.645)", 0.5, 3.0, 1.4, 0.005)
    band_pct = c6.number_input("상·하한 폭 직접 지정(%) · 0이면 σ방식", 0.0, 30.0, 0.0, 0.1)


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
    f'<div style="font-size:13px;color:#888;margin-top:4px">{code.strip()} · 거래량 {vol:,.0f}</div>'
    f'<div style="font-size:12px;color:#888;margin-top:2px">직선추세 {linev[-init_span]:,.0f}→{linev[-1]:,.0f} · '
    f'상한대비 {last / upper[-1] * 100:.1f}% · 하한대비 {last / lower[-1] * 100:.1f}% · 상·하한폭 ±{width * 100:.1f}%</div>',
    unsafe_allow_html=True)

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
        st.caption("오차가 가장 작은 '계산기간'을 설정의 계산 기간에 넣어 보세요.")


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


CROSSHAIR_JS = """
function installCrosshair(gd, initSpan, S, B){
  var LP_MS=1500;          // 길게 눌러 진동이 오기까지의 시간(ms)
  var CROSS_DELAY_MS=150;  // 진동 후 십자선이 나타나기까지의 시간(ms) (진동 → 십자선 순서 맞춤)
  var PAN_GAIN=1.5;        // 손가락 이동 속도 배율
  var ZOOM_GAIN=1.5;       // 확대/축소 속도 배율
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
  function hide(){[vl,hl,dl,pr].forEach(function(e){e.style.display='none';});}

  function fmt(c){return c.slice(5).replace('-','.');}

  // 보이는 구간의 최고/최저에 맞춰 가격 범위를 즉시 변경
  function render(){
    pending=false;
    var a=Math.max(0,Math.floor(i0)), b=Math.min(N-1,Math.ceil(i0+span)-1), i, k, v;
    var lo=Infinity, hi=-Infinity, bl=0, bh=0;
    for(k=0;k<S.length;k++){for(i=a;i<=b;i++){v=S[k][i]; if(v<lo)lo=v; if(v>hi)hi=v;}}
    for(i=a;i<=b;i++){v=B[i]; if(v<bl)bl=v; if(v>bh)bh=v;}
    var pad=(hi-lo)*0.06||1;
    var step=Math.max(3,Math.round(span/4)), tv=[], tt=[], idx;
    for(idx=N-1;idx>=a;idx-=step){tv.push(cats[idx]); tt.push(fmt(cats[idx]));}
    var xr=[i0-0.5,i0+span-0.5];
    Plotly.relayout(gd,{'xaxis.range':xr,'xaxis2.range':xr,
      'yaxis.range':[lo-pad,hi+pad],'yaxis2.range':[bl*1.1,(bh*1.1)||1],
      'xaxis.tickvals':tv,'xaxis.ticktext':tt,'xaxis2.tickvals':tv,'xaxis2.ticktext':tt});
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
  function btn(txt,right,fn){
    var b=document.createElement('div'); b.textContent=txt;
    b.style.cssText='position:absolute;top:2px;right:'+right+'px;width:28px;height:22px;line-height:22px;text-align:center;'
      +'font:600 15px sans-serif;background:#f3f3f3;border:1px solid #ccc;border-radius:4px;z-index:60;cursor:pointer;'
      +'touch-action:manipulation;user-select:none;-webkit-user-select:none;';
    ['touchstart','touchmove','touchend','mousedown','mousemove','dblclick'].forEach(function(n){
      b.addEventListener(n,function(e){e.stopPropagation();},{passive:true});});
    b.addEventListener('click',function(e){e.stopPropagation();fn();});
    gd.appendChild(b);
  }
  btn('＋',92,function(){zoomBy(0.6);});
  btn('－',62,function(){zoomBy(1/0.6);});

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
    fig.update_layout(height=620, margin=dict(l=4, r=58, t=26, b=22),
                      showlegend=False, bargap=0.05, hovermode=False, dragmode=False,
                      plot_bgcolor="white", paper_bgcolor="white")
    return fig


def show(fig, div_id, bars):
    """좌우 이동 + 길게 눌러 십자선 기능을 얹어 차트를 표시"""
    series = [np.asarray(v, dtype=float).tolist() for v in (yv, linev, curvev, upv, lov)]
    post = "installCrosshair(document.getElementById('{plot_id}'), %d, %s, %s);" % (
        init_span, json.dumps(series), json.dumps(np.asarray(bars, dtype=float).tolist()))
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
    st.caption("조작: 좌우로 밀기=기간 이동 · 길게 누름(진동)=십자선 · 두 손가락 벌리기/모으기 또는 ＋/－=확대·축소 · 두 번 톡(더블클릭)=기본값. "
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

st.caption("HP필터 특성상 새 데이터가 추가되면 과거 곡선도 다시 계산되어 모양이 바뀝니다. "
           "미래에셋 차트의 근사 재현이며, 투자 판단은 본인 책임입니다.")
