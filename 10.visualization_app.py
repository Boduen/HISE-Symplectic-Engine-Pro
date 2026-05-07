import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
from typing import Dict, List, Optional

# ==========================================
# AICA Constraint Definition: Semantic Constants & Enums
# ==========================================
class SimMode(Enum):
    MIXED = "mixed"
    SYS1 = "sys1"
    SYS2 = "sys2"

class UIConstants:
    BG_COLOR = '#0e1117'
    TEXT_COLOR = 'white'
    ALERT_COLOR = 'red'
    SAFE_COLOR = '#00ff00'
    FSI_THRESHOLD = 1.0
    MASS_HEAVY_THRESHOLD = 1.0

# Page Configuration (Must remain at top for Streamlit)
st.set_page_config(
    page_title="HISE-Pro Cognitive Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. Deterministic Telemetry Simulator
# ==========================================
def mock_inference_data(
    steps: int = 100, 
    mode: SimMode = SimMode.MIXED,
    seed: Optional[int] = 42  # AICA Fix: Explicit Random Source Injection
) -> pd.DataFrame:
    """
    Simulates HISE physics engine telemetry.
    
    AICA Asserts:
        - Strict Determinism Assumption: Utilizes local RandomState to prevent global PRNG pollution.
        - Strict Type Hinting for DataFrame return.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(steps)

    if mode == SimMode.SYS1:
        entropy = rng.normal(0.2, 0.05, steps)
        mass = rng.normal(0.1, 0.01, steps)
        epsilon = np.ones(steps) * 0.1
        fsi = rng.normal(2.5, 0.2, steps)

    elif mode == SimMode.SYS2:
        entropy = rng.normal(2.5, 0.5, steps)
        mass = rng.normal(2.0, 0.5, steps)
        epsilon = np.ones(steps) * 0.01
        fsi = rng.normal(0.6, 0.1, steps)

    else: 
        entropy = np.concatenate([
            rng.normal(0.2, 0.1, steps // 2), 
            rng.normal(2.5, 0.3, steps - steps // 2)
        ])
        mass = np.sqrt(np.maximum(entropy * 1.5, 0)) + rng.normal(0, 0.1, steps)
        fsi = 1.0 / (mass + 1e-6)
        epsilon = 0.1 / (mass + 1.0)

    q = np.cumsum(np.sin(t * 0.1) * epsilon) 
    p = np.cos(t * 0.1) * mass

    return pd.DataFrame({
        "Step": t,
        "Entropy (H)": entropy,
        "Semantic Mass (M)": mass,
        "Step Size (epsilon)": epsilon,
        "FSI Score": fsi,
        "Position (q)": q,
        "Momentum (p)": p
    })

# ==========================================
# 2. Main UI & Explicit Rendering
# ==========================================
st.title("HISE-Pro: Holographic Inertial Syntax Engine")
st.markdown("### Real-time Cognitive Thermodynamics Monitor")

# Sidebar Controls
st.sidebar.header("Physics Engine Controls")

sim_mode_selection = st.sidebar.selectbox(
    "Simulation Scenario", 
    ["Mixed (System 1->2)", "System 1 (Reflex)", "System 2 (Deep Thought)"]
)

steps = st.sidebar.slider("Generation Steps", 50, 500, 100)
# AICA Fix: Allow users to change seed to see variance, preserving determinism per run
sim_seed = st.sidebar.number_input("Random Seed", value=42, step=1)
run_btn = st.sidebar.button("Run Inference Simulation")

if run_btn:
    # Safely map string selection to Enum
    if "Mixed" in sim_mode_selection:
        mode_key = SimMode.MIXED
    elif "Reflex" in sim_mode_selection:
        mode_key = SimMode.SYS1
    else:
        mode_key = SimMode.SYS2

    data = mock_inference_data(steps, mode=mode_key, seed=sim_seed)

    # --- Top Level KPIs ---
    col1, col2, col3, col4 = st.columns(4)
    avg_mass = float(data["Semantic Mass (M)"].mean())
    min_fsi = float(data["FSI Score"].min())
    sys2_active_count = int((data['Semantic Mass (M)'] > UIConstants.MASS_HEAVY_THRESHOLD).sum())

    with col1:
        st.metric(
            "Avg Semantic Mass", 
            f"{avg_mass:.2f}", 
            delta="Heavy" if avg_mass > UIConstants.MASS_HEAVY_THRESHOLD else "Light", 
            delta_color="inverse"
        )
    with col2:
        st.metric(
            "Min FSI Score", 
            f"{min_fsi:.2f}", 
            delta="Risk" if min_fsi < UIConstants.FSI_THRESHOLD else "Safe", 
            delta_color="normal"
        )
    with col3:
        st.metric(
            "System 2 Activation", 
            f"{(sys2_active_count / steps) * 100:.0f}%"
        )
    with col4:
        is_cooling = bool(data["Step Size (epsilon)"].iloc[-1] < 0.05)
        st.metric(
            "Thermodynamic Status", 
            "Cooling (Converging)" if is_cooling else "Ballistic (Flowing)"
        )

    st.markdown("---")

    # Row 1: Inertia & Safety
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("1. Cognitive Inertia (Mass Dynamics)")
        st.line_chart(data, x="Step", y=["Semantic Mass (M)", "Step Size (epsilon)"])
        st.caption(
            "Observation: As Semantic Mass (M) increases, the Symplectic Step Size (epsilon) "
            "automatically decreases. This represents the transition to System 2 deliberative thought."
        )

    with c2:
        st.subheader("2. Axiom Smuggling Detector (FSI)")

        # AICA Fix: Object-Oriented Matplotlib context, explicitly styled to avoid global pollution
        with plt.style.context('dark_background'):
            fig_fsi, ax_fsi = plt.subplots(figsize=(10, 4))
            ax_fsi.plot(data["Step"], data["FSI Score"], color=UIConstants.SAFE_COLOR, label='FSI Metric')
            ax_fsi.axhline(y=UIConstants.FSI_THRESHOLD, color=UIConstants.ALERT_COLOR, linestyle='--', label='Nyquist Limit (Hallucination)')

            ax_fsi.fill_between(
                data["Step"], 0, UIConstants.FSI_THRESHOLD, 
                alpha=0.2, color=UIConstants.ALERT_COLOR, 
                where=(data["FSI Score"] < UIConstants.FSI_THRESHOLD),
                label='Smuggling Zone'
            )

            ax_fsi.legend(loc='upper right')
            ax_fsi.set_ylabel("Fisher Semantic Information")
            ax_fsi.set_xlabel("Token Step")

            ax_fsi.set_facecolor(UIConstants.BG_COLOR)
            fig_fsi.patch.set_facecolor(UIConstants.BG_COLOR)
            ax_fsi.tick_params(axis='x', colors=UIConstants.TEXT_COLOR)
            ax_fsi.tick_params(axis='y', colors=UIConstants.TEXT_COLOR)
            for spine in ['bottom', 'left']:
                ax_fsi.spines[spine].set_color(UIConstants.TEXT_COLOR)

            st.pyplot(fig_fsi)
            
        st.caption(
            "Red Zone indicates 'Axiom Smuggling'. If the trajectory enters this region, "
            "the RAG Safety Valve is triggered immediately."
        )

    # Row 2: Phase Space Topology
    st.markdown("---")
    st.subheader("3. Semantic Phase Space (Hamiltonian Orbit)")

    col_phase, col_desc = st.columns([2, 1])

    with col_phase:
        # AICA Fix: Safe isolated styling context
        with plt.style.context('dark_background'):
            fig_phase, ax_phase = plt.subplots(figsize=(8, 6))

            sns.scatterplot(
                data=data, 
                x="Position (q)", 
                y="Momentum (p)", 
                hue="Semantic Mass (M)", 
                palette="rocket_r", 
                ax=ax_phase
            )
            ax_phase.plot(data["Position (q)"], data["Momentum (p)"], color=UIConstants.TEXT_COLOR, alpha=0.3)

            ax_phase.set_title("q-p Trajectory Evolution")
            ax_phase.set_xlabel("Semantic Position (q)")
            ax_phase.set_ylabel("Semantic Momentum (p)")

            ax_phase.set_facecolor(UIConstants.BG_COLOR)
            fig_phase.patch.set_facecolor(UIConstants.BG_COLOR)
            ax_phase.tick_params(colors=UIConstants.TEXT_COLOR)
            for spine in ['bottom', 'left']:
                ax_phase.spines[spine].set_color(UIConstants.TEXT_COLOR)

            st.pyplot(fig_phase)

    with col_desc:
        st.markdown("""
        **Physics Interpretation:**
        
        * **Spiral Sink**: Indicates System 2 is applying "Thermodynamic Friction" to force logical convergence.
        * **Limit Cycle**: Indicates System 1 is in a stable, reflexive generation loop (Grammar flow).
        * **Divergence**: If the trajectory escapes the bounded region, it indicates gradient explosion or physical parameter mismatch.
        """)

else:
    st.info("Awaiting input... Click 'Run Inference Simulation' on the sidebar to visualize HISE-Pro dynamics.")
