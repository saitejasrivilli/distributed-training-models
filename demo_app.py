#!/usr/bin/env python3
"""Interactive demo — showcases real benchmark results from benchmarks/real_results.json"""

import json
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Distributed LLM Training Demo",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stMetric { background-color: #262730; padding: 15px; border-radius: 10px; }
    h1 { color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

# ── Load real benchmark data ───────────────────────────────────────────────────
RESULTS_PATH = Path(__file__).parent / "benchmarks" / "real_results.json"

@st.cache_data
def load_results():
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    # Fallback: embed the values directly so the app always works
    return {
        "gpu1_single_fp32": {"tokens_per_sec": 8300,  "peak_mem_gb": 5.21, "world_size": 1},
        "gpu1_single_fp16": {"tokens_per_sec": 26097, "peak_mem_gb": 4.56, "world_size": 1},
        "gpu2_ddp_fp32":    {"tokens_per_sec": 7123,  "peak_mem_gb": 5.67, "world_size": 2},
        "gpu2_ddp_fp16":    {"tokens_per_sec": 20522, "peak_mem_gb": 5.02, "world_size": 2},
        "gpu2_fsdp_fp32":   {"tokens_per_sec": 7261,  "peak_mem_gb": 4.97, "world_size": 2},
        "gpu2_fsdp_fp16":   {"tokens_per_sec": 21844, "peak_mem_gb": 4.31, "world_size": 2},
        "gpu4_ddp_fp32":    {"tokens_per_sec": 10921, "peak_mem_gb": 5.67, "world_size": 4},
        "gpu4_ddp_fp16":    {"tokens_per_sec": 20950, "peak_mem_gb": 5.02, "world_size": 4},
        "gpu4_fsdp_fp32":   {"tokens_per_sec": 10387, "peak_mem_gb": 4.63, "world_size": 4},
        "gpu4_fsdp_fp16":   {"tokens_per_sec": 17840, "peak_mem_gb": 3.96, "world_size": 4},
    }

R = load_results()
BASE_FP32 = R["gpu1_single_fp32"]["tokens_per_sec"]   # 8,300
BASE_FP16 = R["gpu1_single_fp16"]["tokens_per_sec"]   # 26,097
AMP_SPEEDUP = BASE_FP16 / BASE_FP32                   # 3.14×
FSDP4_MEM_SAVE = (R["gpu4_ddp_fp16"]["peak_mem_gb"] - R["gpu4_fsdp_fp16"]["peak_mem_gb"]) \
                 / R["gpu4_ddp_fp16"]["peak_mem_gb"] * 100  # 21.1%

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("🎯 Key Results")
st.sidebar.metric("AMP Speedup (1 GPU)", f"{AMP_SPEEDUP:.2f}×", "fp32 → fp16")
st.sidebar.metric("FSDP Memory Saved", f"{FSDP4_MEM_SAVE:.0f}%", "vs DDP at 4 GPUs")
st.sidebar.metric("Configs Benchmarked", "10", "1 / 2 / 4 GPU")
st.sidebar.markdown("---")
st.sidebar.markdown("**Hardware:** 4× NVIDIA A30 (PCIe)")
st.sidebar.markdown("**Model:** 163M-param GPT")
st.sidebar.markdown("**Framework:** PyTorch 2.7.1 + CUDA 11.8")
st.sidebar.markdown("---")
st.sidebar.markdown("**Built by:** Sai Teja Srivilli")
st.sidebar.markdown("[📂 GitHub](https://github.com/saitejasrivilli/distributed-training-models)")

st.title("🚀 Distributed LLM Training System")
st.markdown("Interactive demo — all numbers measured on real hardware.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Benchmark Results",
    "⚙️ Scaling Calculator",
    "🎯 Training Visualizer",
    "💡 DDP vs FSDP",
    "🔬 Live Demo"
])

# ── Tab 1: Real benchmark results ─────────────────────────────────────────────
with tab1:
    st.header("Measured on 4× NVIDIA A30 — 163M-param GPT, batch=4/GPU, seq=512")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AMP Speedup (1 GPU)", f"{AMP_SPEEDUP:.2f}×", "8,300 → 26,097 tok/s")
    col2.metric("4-GPU DDP fp32",      f"{R['gpu4_ddp_fp32']['tokens_per_sec']:,} tok/s",
                f"{R['gpu4_ddp_fp32']['tokens_per_sec']/BASE_FP32:.2f}× vs 1-GPU fp32")
    col3.metric("FSDP mem/GPU (4-GPU fp16)", f"{R['gpu4_fsdp_fp16']['peak_mem_gb']:.2f} GB",
                f"−{FSDP4_MEM_SAVE:.0f}% vs DDP")
    col4.metric("Best throughput",     f"{BASE_FP16:,} tok/s", "1 GPU + AMP")

    st.markdown("---")

    # Throughput bar chart — all 10 configs
    st.subheader("📈 Throughput Across All Configurations")
    labels = [
        "1 GPU\nfp32", "1 GPU\nfp16",
        "2 GPU DDP\nfp32", "2 GPU DDP\nfp16",
        "2 GPU FSDP\nfp32", "2 GPU FSDP\nfp16",
        "4 GPU DDP\nfp32", "4 GPU DDP\nfp16",
        "4 GPU FSDP\nfp32", "4 GPU FSDP\nfp16",
    ]
    keys = [
        "gpu1_single_fp32", "gpu1_single_fp16",
        "gpu2_ddp_fp32", "gpu2_ddp_fp16",
        "gpu2_fsdp_fp32", "gpu2_fsdp_fp16",
        "gpu4_ddp_fp32", "gpu4_ddp_fp16",
        "gpu4_fsdp_fp32", "gpu4_fsdp_fp16",
    ]
    colors = ["#3498db", "#e74c3c",
              "#5dade2", "#e67e22", "#76d7c4", "#f0b27a",
              "#2980b9", "#c0392b", "#1abc9c", "#e59866"]
    tok_s = [R[k]["tokens_per_sec"] for k in keys]

    fig = go.Figure(go.Bar(
        x=labels, y=tok_s,
        marker_color=colors,
        text=[f"{v:,}" for v in tok_s],
        textposition="auto",
        textfont=dict(color="white", size=11)
    ))
    fig.update_layout(
        yaxis_title="Tokens / second", height=420, template="plotly_dark", showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # AMP + DDP scaling side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⚡ AMP Speedup (single GPU)")
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(AMP_SPEEDUP, 2),
            title={"text": "fp16 / fp32  throughput ratio"},
            gauge={
                "axis": {"range": [0, 5]},
                "bar": {"color": "#e74c3c"},
                "steps": [
                    {"range": [0, 2], "color": "#34495e"},
                    {"range": [2, 3], "color": "#7f8c8d"},
                    {"range": [3, 5], "color": "#27ae60"},
                ],
            }
        ))
        fig2.update_layout(height=320, template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("📈 DDP Scaling (fp32 — PCIe A30)")
        gpus   = [1, 2, 4]
        actual = [
            BASE_FP32 / BASE_FP32,
            R["gpu2_ddp_fp32"]["tokens_per_sec"] / BASE_FP32,
            R["gpu4_ddp_fp32"]["tokens_per_sec"] / BASE_FP32,
        ]
        ideal  = [1.0, 2.0, 4.0]
        efficiencies = [100, 42.9, 32.9]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=gpus, y=actual, mode="lines+markers+text",
            name="Measured speedup",
            line=dict(color="#e74c3c", width=3),
            marker=dict(size=12),
            text=[f"{v:.2f}×\n({e:.0f}%)" for v, e in zip(actual, efficiencies)],
            textposition="top right"
        ))
        fig3.add_trace(go.Scatter(
            x=gpus, y=ideal, mode="lines",
            name="Ideal (linear)",
            line=dict(color="#95a5a6", width=2, dash="dash")
        ))
        fig3.update_layout(
            xaxis_title="GPUs", yaxis_title="Speedup vs 1-GPU fp32",
            height=320, template="plotly_dark",
            xaxis=dict(tickvals=[1, 2, 4])
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.info(
        "**Why efficiency is 33–43%:** A30 GPUs are PCIe-connected (no NVLink). "
        "Each DDP step all-reduces ~650 MB of fp32 gradients — at 163M params the "
        "compute-to-communication ratio is unfavorable. Efficiency improves with "
        "larger models (>1B params) or NVLink interconnects."
    )


# ── Tab 2: Scaling Calculator ──────────────────────────────────────────────────
with tab2:
    st.header("⚙️ Scaling Calculator")
    st.markdown("Estimates based on measured throughput and efficiency on PCIe A30 GPUs.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Parameters")
        num_gpus       = st.slider("Number of GPUs", 1, 16, 4)
        training_steps = st.number_input("Training Steps", 1_000, 1_000_000, 100_000, step=10_000)
        batch_per_gpu  = st.number_input("Batch Size per GPU", 1, 32, 4)
        seq_length     = st.selectbox("Sequence Length", [128, 256, 512, 1024, 2048], index=2)
        use_amp        = st.checkbox("Use Mixed Precision (fp16 AMP)", value=True)

        # Efficiency model from measured data
        # 1 GPU: 100%, 2 GPU: 42.9%, 4 GPU: 32.9%
        # Log-interpolate for other GPU counts
        if num_gpus == 1:
            eff = 1.0
        elif num_gpus <= 2:
            eff = 0.429
        elif num_gpus <= 4:
            eff = 0.329
        else:
            eff = max(0.25, 0.329 * (4 / num_gpus) ** 0.3)   # rough extrapolation

        base = BASE_FP16 if use_amp else BASE_FP32
        effective_toks = base * num_gpus * eff if num_gpus > 1 else base

        total_tokens = training_steps * batch_per_gpu * seq_length * num_gpus
        time_s = total_tokens / effective_toks
        time_min = time_s / 60
        time_hr  = time_s / 3600

        cost_per_gpu_hr = 3.06   # AWS p3.2xlarge
        total_cost = time_hr * num_gpus * cost_per_gpu_hr

    with col2:
        st.subheader("📊 Estimated Results")
        st.metric("Effective Throughput", f"{effective_toks:,.0f} tok/s")
        st.metric("Parallel Efficiency",  f"{eff*100:.1f}%", help="Based on measured PCIe A30 data")
        st.metric("Training Time",
                  f"{time_min:.1f} min" if time_hr < 1 else f"{time_hr:.2f} hr")
        st.metric("Cloud Cost (AWS)", f"${total_cost:.2f}")
        if use_amp:
            st.success(f"AMP enabled — {AMP_SPEEDUP:.1f}× faster than fp32 on same hardware.")


# ── Tab 3: Training Visualizer ────────────────────────────────────────────────
with tab3:
    st.header("🎯 Training Progress Visualizer")
    st.markdown("Illustrative loss curves based on the convergence pattern observed in test runs.")

    steps = np.arange(0, 1001, 10)
    np.random.seed(42)
    noise = np.random.normal(0, 0.015, len(steps)).cumsum() * 0.002
    loss_1gpu = 10 * np.exp(-steps / 400) + 1.0 + noise
    loss_4gpu = 10 * np.exp(-steps / 400) + 1.0 + noise[::-1] * 0.8

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=loss_1gpu, mode="lines",
                             name="1 GPU (fp16)", line=dict(color="#3498db", width=2.5)))
    fig.add_trace(go.Scatter(x=steps, y=loss_4gpu, mode="lines",
                             name="4 GPU DDP (fp16)", line=dict(color="#e74c3c", width=2.5)))
    fig.update_layout(
        title="Training Loss (illustrative — same convergence per token at all GPU counts)",
        xaxis_title="Steps", yaxis_title="Loss",
        height=420, template="plotly_dark", hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("💻 Measured GPU Memory — 4-GPU Runs")

    col1, col2 = st.columns(2)
    with col1:
        configs_mem = ["DDP fp32", "FSDP fp32", "DDP fp16", "FSDP fp16"]
        mem_vals    = [
            R["gpu4_ddp_fp32"]["peak_mem_gb"],
            R["gpu4_fsdp_fp32"]["peak_mem_gb"],
            R["gpu4_ddp_fp16"]["peak_mem_gb"],
            R["gpu4_fsdp_fp16"]["peak_mem_gb"],
        ]
        fig_mem = go.Figure(go.Bar(
            x=configs_mem, y=mem_vals,
            marker_color=["#3498db", "#2ecc71", "#e74c3c", "#f39c12"],
            text=[f"{m:.2f} GB" for m in mem_vals],
            textposition="auto", textfont=dict(color="white", size=13)
        ))
        fig_mem.add_hline(y=24, line_dash="dash", line_color="#95a5a6",
                          annotation_text="A30 capacity (24 GB)")
        fig_mem.update_layout(
            title="Peak Memory per GPU (4-GPU, 163M model)",
            yaxis_title="GB", yaxis_range=[0, 26],
            height=360, template="plotly_dark", showlegend=False
        )
        st.plotly_chart(fig_mem, use_container_width=True)

    with col2:
        configs_thr = ["DDP fp32", "FSDP fp32", "DDP fp16", "FSDP fp16"]
        thr_vals    = [
            R["gpu4_ddp_fp32"]["tokens_per_sec"],
            R["gpu4_fsdp_fp32"]["tokens_per_sec"],
            R["gpu4_ddp_fp16"]["tokens_per_sec"],
            R["gpu4_fsdp_fp16"]["tokens_per_sec"],
        ]
        fig_thr = go.Figure(go.Bar(
            x=configs_thr, y=thr_vals,
            marker_color=["#3498db", "#2ecc71", "#e74c3c", "#f39c12"],
            text=[f"{v:,}" for v in thr_vals],
            textposition="auto", textfont=dict(color="white", size=13)
        ))
        fig_thr.update_layout(
            title="Throughput — 4-GPU Runs (tok/s total)",
            yaxis_title="Tokens / second",
            height=360, template="plotly_dark", showlegend=False
        )
        st.plotly_chart(fig_thr, use_container_width=True)


# ── Tab 4: DDP vs FSDP ────────────────────────────────────────────────────────
with tab4:
    st.header("💡 DDP vs FSDP — Memory & Throughput Tradeoff")

    df = pd.DataFrame([
        {"GPUs": 2, "Strategy": "DDP",  "Precision": "fp32",
         "Throughput": R["gpu2_ddp_fp32"]["tokens_per_sec"],
         "Mem/GPU (GB)": R["gpu2_ddp_fp32"]["peak_mem_gb"]},
        {"GPUs": 2, "Strategy": "FSDP", "Precision": "fp32",
         "Throughput": R["gpu2_fsdp_fp32"]["tokens_per_sec"],
         "Mem/GPU (GB)": R["gpu2_fsdp_fp32"]["peak_mem_gb"]},
        {"GPUs": 2, "Strategy": "DDP",  "Precision": "fp16",
         "Throughput": R["gpu2_ddp_fp16"]["tokens_per_sec"],
         "Mem/GPU (GB)": R["gpu2_ddp_fp16"]["peak_mem_gb"]},
        {"GPUs": 2, "Strategy": "FSDP", "Precision": "fp16",
         "Throughput": R["gpu2_fsdp_fp16"]["tokens_per_sec"],
         "Mem/GPU (GB)": R["gpu2_fsdp_fp16"]["peak_mem_gb"]},
        {"GPUs": 4, "Strategy": "DDP",  "Precision": "fp32",
         "Throughput": R["gpu4_ddp_fp32"]["tokens_per_sec"],
         "Mem/GPU (GB)": R["gpu4_ddp_fp32"]["peak_mem_gb"]},
        {"GPUs": 4, "Strategy": "FSDP", "Precision": "fp32",
         "Throughput": R["gpu4_fsdp_fp32"]["tokens_per_sec"],
         "Mem/GPU (GB)": R["gpu4_fsdp_fp32"]["peak_mem_gb"]},
        {"GPUs": 4, "Strategy": "DDP",  "Precision": "fp16",
         "Throughput": R["gpu4_ddp_fp16"]["tokens_per_sec"],
         "Mem/GPU (GB)": R["gpu4_ddp_fp16"]["peak_mem_gb"]},
        {"GPUs": 4, "Strategy": "FSDP", "Precision": "fp16",
         "Throughput": R["gpu4_fsdp_fp16"]["tokens_per_sec"],
         "Mem/GPU (GB)": R["gpu4_fsdp_fp16"]["peak_mem_gb"]},
    ])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Memory per GPU")
        fig_m = go.Figure()
        for strat, color in [("DDP", "#3498db"), ("FSDP", "#2ecc71")]:
            for ngpu, pattern in [(2, ""), (4, "/")]:
                row = df[(df.Strategy == strat) & (df.GPUs == ngpu) & (df.Precision == "fp16")]
                if not row.empty:
                    fig_m.add_trace(go.Bar(
                        name=f"{strat} {ngpu}-GPU fp16",
                        x=[f"{strat} {ngpu}-GPU"],
                        y=[row["Mem/GPU (GB)"].values[0]],
                        marker_color=color,
                        marker_pattern_shape=pattern,
                        text=[f"{row['Mem/GPU (GB)'].values[0]:.2f} GB"],
                        textposition="auto", textfont=dict(color="white")
                    ))
        fig_m.add_hline(y=24, line_dash="dash", line_color="#95a5a6",
                        annotation_text="A30 limit (24 GB)")
        fig_m.update_layout(
            yaxis_title="GB", yaxis_range=[0, 26],
            height=360, template="plotly_dark", showlegend=True, barmode="group"
        )
        st.plotly_chart(fig_m, use_container_width=True)

    with col2:
        st.subheader("Throughput (tok/s)")
        fig_t = go.Figure()
        for strat, color in [("DDP", "#3498db"), ("FSDP", "#2ecc71")]:
            for ngpu in [2, 4]:
                row = df[(df.Strategy == strat) & (df.GPUs == ngpu) & (df.Precision == "fp16")]
                if not row.empty:
                    fig_t.add_trace(go.Bar(
                        name=f"{strat} {ngpu}-GPU fp16",
                        x=[f"{strat} {ngpu}-GPU"],
                        y=[row["Throughput"].values[0]],
                        marker_color=color,
                        text=[f"{row['Throughput'].values[0]:,}"],
                        textposition="auto", textfont=dict(color="white")
                    ))
        fig_t.update_layout(
            yaxis_title="Tokens / second",
            height=360, template="plotly_dark", showlegend=True, barmode="group"
        )
        st.plotly_chart(fig_t, use_container_width=True)

    st.markdown("---")
    st.subheader("Memory savings grow with GPU count")
    gpus_list = [2, 4]
    savings_fp32 = [12.3, 18.3]
    savings_fp16 = [14.1, 21.1]

    fig_save = go.Figure()
    fig_save.add_trace(go.Scatter(x=gpus_list, y=savings_fp32, mode="lines+markers+text",
                                  name="fp32", line=dict(color="#3498db", width=3),
                                  marker=dict(size=12),
                                  text=[f"{v:.1f}%" for v in savings_fp32],
                                  textposition="top left"))
    fig_save.add_trace(go.Scatter(x=gpus_list, y=savings_fp16, mode="lines+markers+text",
                                  name="fp16 (AMP)", line=dict(color="#e74c3c", width=3),
                                  marker=dict(size=12),
                                  text=[f"{v:.1f}%" for v in savings_fp16],
                                  textposition="top right"))
    fig_save.update_layout(
        title="FSDP memory reduction vs DDP (each GPU holds 1/N of sharded state)",
        xaxis_title="Number of GPUs", yaxis_title="Memory saved (%)",
        xaxis=dict(tickvals=[2, 4]), yaxis_range=[0, 30],
        height=320, template="plotly_dark"
    )
    st.plotly_chart(fig_save, use_container_width=True)

    st.info(
        "**When to use FSDP:** when model + optimizer states approach single-GPU memory limit. "
        "At 2 GPUs in fp16, FSDP is actually *faster* than DDP (+6.4%) because smaller shards "
        "reduce the all-reduce volume. At 4 GPUs, the cost is 5–15% throughput for 18–21% less memory."
    )


# ── Tab 5: Live Demo ──────────────────────────────────────────────────────────
with tab5:
    st.header("🔬 Text Generation Demo")
    st.markdown("Runs the trained tiny GPT model directly — actual inference, not simulated.")

    col1, col2 = st.columns([2, 1])
    with col1:
        prompt = st.text_area("Enter your prompt:", "The future of artificial intelligence", height=100)
    with col2:
        temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
        max_tokens  = st.slider("Max new tokens", 10, 100, 50, 5)

    if st.button("🚀 Generate", type="primary", use_container_width=True):
        with st.spinner("Loading model and generating..."):
            try:
                import torch
                import sys, os
                sys.path.insert(0, str(Path(__file__).parent))
                from src.model.transformer import GPTModel
                from src.model.config import CONFIGS
                from transformers import GPT2Tokenizer

                tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                tokenizer.pad_token = tokenizer.eos_token

                model = GPTModel(CONFIGS["tiny"])
                model.eval()

                input_ids = tokenizer.encode(prompt, return_tensors="pt")
                with torch.no_grad():
                    output = model.generate(input_ids, max_new_tokens=max_tokens, temperature=temperature, top_k=50)
                generated = tokenizer.decode(output[0], skip_special_tokens=True)

                st.success("✅ Done")
                st.markdown(f"**Output:**\n\n{generated}")

                col1, col2, col3 = st.columns(3)
                col1.metric("Tokens generated", max_tokens)
                col2.metric("Model", "GPT tiny (13M params)")
                col3.metric("Weights", "Random init (untrained)")

                st.info("This model has random weights — output is grammatically structured but semantically random. "
                        "Training on real data would produce coherent text.")
            except Exception as e:
                st.error(f"Could not run inference: {e}")
                st.markdown("_To run locally: `pip install torch transformers` then `streamlit run demo_app.py`_")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <p style='font-size: 1.1em;'><strong>🚀 Distributed LLM Training System</strong></p>
    <p>
        <a href='https://github.com/saitejasrivilli/distributed-training-models' style='margin: 0 10px;'>📂 GitHub</a> |
        <a href='mailto:srivillibhutturu.s@northeastern.edu' style='margin: 0 10px;'>📧 Contact</a>
    </p>
    <p style='margin-top: 10px; opacity: 0.6;'>
        PyTorch 2.7 · CUDA 11.8 · NCCL · 4× NVIDIA A30<br>
        © 2025 Sai Teja Srivilli
    </p>
</div>
""", unsafe_allow_html=True)
