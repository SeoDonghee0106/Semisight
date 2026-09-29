import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dataclasses import dataclass

st.set_page_config(page_title="웨이퍼 불량 패턴 판정", layout="wide")


@dataclass
class Result:
    pattern: str
    confidence: float
    yield_pct: float
    fail_count: int
    fail_radius_mean: float
    fail_radius_std: float
    action: str


def prepare(df, wafer_radius):
    df = df.copy()
    df["status"] = df["status"].astype(str).str.upper().str.strip()
    if "radius" not in df:
        df["radius"] = np.hypot(df["die_x"], df["die_y"])
    if "angle_deg" not in df:
        df["angle_deg"] = np.degrees(np.arctan2(df["die_y"], df["die_x"]))
    return df


def classify(df, R):
    fail = df[df["status"] == "FAIL"]
    total, n_fail = len(df), len(fail)
    yield_pct = (total - n_fail) / total * 100

    if n_fail == 0 or yield_pct > 98.0:
        return Result("CLEAN", 1.0, yield_pct, n_fail, 0.0, 0.0,
                      "조치 불필요. 추이만 모니터링하세요.")

    r_mean, r_std = fail["radius"].mean(), fail["radius"].std()
    ang = np.radians(fail["angle_deg"].values)
    conc = np.hypot(np.mean(np.sin(ang)), np.mean(np.cos(ang)))

    if r_mean > R * 0.75 and r_std < 20:
        p = "EDGE_RING"
        c = min(1.0, (r_mean / R) * (1 - r_std / 30))
        a = "엣지 링 불량 감지. 증착 균일도 점검 및 Edge exclusion 기준 재검토 권장."
    elif r_mean < R * 0.25:
        p = "CENTER"
        c = 1.0 - (r_mean / (R * 0.25))
        a = "중심부 불량 감지. 척(Chuck) 온도 프로파일, 중심 가스 플로우 점검 권장."
    elif conc > 0.6:
        p = "SCRATCH"
        c = conc
        a = "선형 스크래치 감지. CMP 패드 상태, 슬러리 파티클 오염 여부 확인 필요."
    else:
        p = "RANDOM"
        c = 1.0 - conc
        a = "랜덤 불량. 파티클 카운트, 노광 오차 통계 확인 및 수율 모니터링."

    return Result(p, float(np.clip(c, 0, 1)), yield_pct, n_fail,
                  r_mean, r_std, a)


def sample_data(mode, n=1200, R=150.0):
    r = R * np.sqrt(np.random.uniform(0, 1, n))
    th = np.random.uniform(0, 360, n)
    status = np.array(["PASS"] * n, dtype=object)
    idx = np.random.choice(n, 100, replace=False)
    if mode == "ring":
        r[idx] = np.random.normal(R * 0.93, 2, 100)
    elif mode == "center":
        r[idx] = np.clip(np.random.normal(0, R * 0.15, 100), 0, R)
    elif mode == "scratch":
        th[idx] = 45 + np.random.normal(0, 1.2, 100)
        r[idx] = np.random.uniform(R * 0.3, R * 0.95, 100)
    status[idx] = "FAIL"
    return pd.DataFrame({
        "status": status,
        "die_x": r * np.cos(np.radians(th)),
        "die_y": r * np.sin(np.radians(th)),
    })


def wafer_map(df, R):
    fig = go.Figure()
    for s, color, size in [("PASS", "#E0E0E0", 5), ("FAIL", "#D32F2F", 7)]:
        d = df[df["status"] == s]
        fig.add_trace(go.Scatter(x=d["die_x"], y=d["die_y"], mode="markers",
                                 name=s, marker=dict(color=color, size=size)))
    t = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(x=R * np.cos(t), y=R * np.sin(t), mode="lines",
                             line=dict(color="black", dash="dash"),
                             showlegend=False, hoverinfo="skip"))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(height=550, margin=dict(l=10, r=10, t=30, b=10))
    return fig


st.title("웨이퍼 불량 패턴 자동 판정")

with st.sidebar:
    R = st.number_input("웨이퍼 반경 (mm)", value=150.0, step=10.0)
    file = st.file_uploader("웨이퍼 데이터 CSV", type="csv")
    st.caption("필수 컬럼: status(PASS/FAIL), die_x, die_y")
    mode = st.selectbox("또는 샘플 데이터", ["사용 안 함", "ring", "center", "scratch", "random"])

df = None
if file:
    df = pd.read_csv(file)
    missing = {"status", "die_x", "die_y"} - set(df.columns)
    if missing:
        st.error(f"필수 컬럼이 없습니다: {', '.join(missing)}")
        st.stop()
elif mode != "사용 안 함":
    if "sample" not in st.session_state or st.session_state.get("mode") != mode:
        st.session_state["sample"] = sample_data(mode, R=R)
        st.session_state["mode"] = mode
    df = st.session_state["sample"]

if df is None:
    st.info("왼쪽에서 CSV를 올리거나 샘플 데이터를 선택하세요.")
    st.stop()

df = prepare(df, R)
res = classify(df, R)

c1, c2, c3, c4 = st.columns(4)
c1.metric("판정", res.pattern)
c2.metric("신뢰도", f"{res.confidence:.0%}")
c3.metric("수율", f"{res.yield_pct:.1f}%")
c4.metric("불량 다이", f"{res.fail_count}개")

if res.pattern == "CLEAN":
    st.success(res.action)
else:
    st.warning(res.action)

st.plotly_chart(wafer_map(df, R), use_container_width=True)
