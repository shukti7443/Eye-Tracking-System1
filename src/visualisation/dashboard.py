import json
import time
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Eye-Tracking Benchmark Dashboard",
    page_icon="👁️",
    layout="wide",
)

st.sidebar.title("👁️ ET Benchmark")
st.sidebar.markdown("---")
mode = st.sidebar.radio("Data Source", ["Demo Mode", "Load Report"])

def get_color(label):
    if label == "GOOD":
        return "#4CAF50"
    elif label == "MODERATE":
        return "#FF9800"
    else:
        return "#F44336"

def make_fake_report():
    targets = [(192 + (i % 3) * 576, 108 + (i // 3) * 432) for i in range(9)]
    per_target = []
    for i, (tx, ty) in enumerate(targets):
        per_target.append({
            "target_id": i,
            "target_x": tx,
            "target_y": ty,
            "mean_gaze_x": tx + np.random.uniform(-50, 50),
            "mean_gaze_y": ty + np.random.uniform(-40, 40),
            "error_px": abs(np.random.normal(40, 15)),
            "error_deg": abs(np.random.normal(1.2, 0.4)),
            "n_samples": int(np.random.uniform(40, 55)),
        })

    return {
        "session_id": "demo_session",
        "device_type": "webcam",
        "task_type": "grid_accuracy",
        "duration_s": 60.0,
        "screen": {"width_px": 1920, "height_px": 1080},
        "accuracy": {
            "per_target": per_target,
            "mean_error_px": float(np.mean([t["error_px"] for t in per_target])),
            "rmse_px": float(np.sqrt(np.mean([t["error_px"]**2 for t in per_target]))),
            "mean_error_deg": float(np.mean([t["error_deg"] for t in per_target])),
            "worst_target_error_px": float(max(t["error_px"] for t in per_target)),
            "best_target_error_px": float(min(t["error_px"] for t in per_target)),
        },
        "precision": {
            "rms_s2s_px": 18.0,
            "std_x_px": 14.0,
            "std_y_px": 12.0,
            "bcea_px2": 1100.0,
            "bcea_prob": 0.68,
            "n_samples": 1620,
        },
        "data_quality": {
            "total_samples": 1800,
            "valid_samples": 1620,
            "invalid_samples": 180,
            "data_loss_rate_pct": 10.0,
            "estimated_blink_count": 12,
            "mean_blink_duration_ms": 180.0,
            "inter_sample_jitter_ms": 1.2,
            "out_of_bounds_rate_pct": 0.5,
            "quality_label": "MODERATE",
            "recommendation": "Try improving your lighting conditions.",
        },
    }

if mode == "Load Report":
    uploaded = st.sidebar.file_uploader("Upload report.json", type="json")
    if uploaded:
        report = json.load(uploaded)
    else:
        st.info("Please upload a report.json file from the sidebar.")
        st.stop()
else:
    report = make_fake_report()
    st.info("Running in demo mode with simulated data.")

acc = report["accuracy"]
prec = report["precision"]
dq = report["data_quality"]
quality = dq["quality_label"]
color = get_color(quality)

st.title("👁️ Eye-Tracking Benchmark Dashboard")
st.markdown(f"**Session:** {report['session_id']} &nbsp;|&nbsp; **Device:** {report['device_type']} &nbsp;|&nbsp; **Task:** {report['task_type']}")

st.markdown(f"""
<div style='background:{color}22; border:2px solid {color};
border-radius:8px; padding:10px 20px; margin-bottom:20px'>
<span style='color:{color}; font-weight:700; font-size:18px'>{quality}</span>
&nbsp;&nbsp;
<span>{dq['recommendation']}</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.subheader("Key Metrics")

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Mean Error", f"{acc['mean_error_px']:.1f} px")
col2.metric("RMSE", f"{acc['rmse_px']:.1f} px")
col3.metric("Angular Error", f"{acc['mean_error_deg']:.2f}°")
col4.metric("RMS S2S", f"{prec['rms_s2s_px']:.1f} px")
col5.metric("Data Loss", f"{dq['data_loss_rate_pct']:.1f}%")
col6.metric("BCEA", f"{prec['bcea_px2']:.0f} px²")

st.markdown("---")
left, right = st.columns([3, 2])

with left:
    st.subheader("Where Did People Look vs Where They Should Have")
    per_target = acc["per_target"]
    screen_w = report["screen"]["width_px"]
    screen_h = report["screen"]["height_px"]
    max_err = max(t["error_px"] for t in per_target)

    fig = go.Figure()
    for t in per_target:
        err = t["error_px"]
        norm = err / max(max_err, 1)
        dot_color = f"rgb({int(255*norm)},{int(255*(1-norm))},80)"

        fig.add_trace(go.Scatter(
            x=[t["target_x"]], y=[t["target_y"]],
            mode="markers",
            marker=dict(symbol="circle-open", size=22, color="white", line=dict(width=2)),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[t["mean_gaze_x"]], y=[t["mean_gaze_y"]],
            mode="markers",
            marker=dict(symbol="x", size=10, color=dot_color, line=dict(width=2)),
            name=f"T{t['target_id']+1}: {err:.0f}px",
        ))
        fig.add_shape(
            type="line",
            x0=t["target_x"], y0=t["target_y"],
            x1=t["mean_gaze_x"], y1=t["mean_gaze_y"],
            line=dict(color=dot_color, width=1.5, dash="dot"),
        )

    fig.update_layout(
        height=400,
        paper_bgcolor="#0d0d1a",
        plot_bgcolor="#1a1a2e",
        font=dict(color="white"),
        xaxis=dict(range=[0, screen_w], showgrid=False),
        yaxis=dict(range=[screen_h, 0], showgrid=False),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Error Per Target")
    labels = [f"T{t['target_id']+1}" for t in per_target]
    errors = [t["error_px"] for t in per_target]

    bar_fig = go.Figure(go.Bar(
        x=labels,
        y=errors,
        marker_color=color,
        text=[f"{e:.0f}" for e in errors],
        textposition="outside",
    ))
    bar_fig.update_layout(
        height=400,
        paper_bgcolor="#0d0d1a",
        plot_bgcolor="#1a1a2e",
        font=dict(color="white"),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#333"),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(bar_fig, use_container_width=True)

st.markdown("---")
st.subheader("Data Quality")

q1, q2, q3 = st.columns(3)
q1.metric("Total Samples", f"{dq['total_samples']:,}")
q1.metric("Valid Samples", f"{dq['valid_samples']:,}")
q1.metric("Data Loss", f"{dq['data_loss_rate_pct']:.1f}%")

q2.metric("Blinks Detected", dq["estimated_blink_count"])
q2.metric("Avg Blink Duration", f"{dq['mean_blink_duration_ms']:.0f} ms")
q2.metric("Timing Jitter", f"{dq['inter_sample_jitter_ms']:.1f} ms")

q3.metric("Std X", f"{prec['std_x_px']:.1f} px")
q3.metric("Std Y", f"{prec['std_y_px']:.1f} px")
q3.metric("Out of Bounds", f"{dq['out_of_bounds_rate_pct']:.1f}%")

st.markdown("---")
st.caption("Eye-Tracking Benchmark Tool | Built for RUXAILAB")
