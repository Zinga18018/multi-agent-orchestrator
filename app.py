"""
streamlit UI for the multi-agent orchestrator.
give it a task, watch agents collaborate.
"""

import streamlit as st
from core import AgentPool, Orchestrator, OrchestratorConfig

st.set_page_config(page_title="Multi-Agent Orchestrator", layout="wide")


@st.cache_resource
def load_agents():
    config = OrchestratorConfig()
    pool = AgentPool(config)
    pool.load()
    return pool, Orchestrator(pool)


st.title("Multi-Agent Orchestrator")
st.caption("TinyLlama-1.1B · plan -> execute -> synthesize")

pool, orchestrator = load_agents()

task = st.text_area(
    "describe your task", height=120,
    placeholder="Write a Python function to find prime numbers and analyze its time complexity",
)

agents = st.multiselect(
    "specialist agents to use",
    ["coder", "researcher", "analyst"],
    default=["coder", "researcher"],
)

max_tokens = st.slider("max tokens per agent", 100, 500, 300)

if st.button("run pipeline", type="primary", use_container_width=True) and task.strip():
    full_agents = ["planner"] + agents + ["synthesizer"]

    with st.spinner("running multi-agent pipeline..."):
        result = orchestrator.run(task, agents=full_agents, max_tokens=max_tokens)

    # show the plan
    with st.expander("phase 1: planning", expanded=True):
        st.markdown(result["plan"])

    # show each specialist's output
    for output in result["agent_outputs"]:
        role = output["agent"]
        if role in ("planner", "synthesizer"):
            continue
        with st.expander(f"phase 2: {role}", expanded=True):
            st.markdown(output["output"])
            st.caption(f"{output['inference_ms']:.0f}ms")

    # final synthesis
    st.divider()
    st.subheader("final synthesis")
    st.markdown(result["final_synthesis"])

    st.caption(
        f"total pipeline: {result['total_time_ms']:.0f}ms · "
        f"{result['agents_used']} agents"
    )
