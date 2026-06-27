import streamlit as st
from agents import crew

st.set_page_config(
    page_title="AI Competitor Intelligence Engine",
    page_icon=":art:",
    layout="wide"
)

st.title("Multi-Agent Competitor Intelligence & Content Engine")

st.markdown(
    "Enter a company name, a competitor sector, or a trending tech topic. "
    "Our specialized AI agent crew will scan the live internet via Tavily, analyze recent updates with Gemini, "
    "and draft a polished, ready-to-publish LinkedIn post for you."
)

st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Configurator")
    # User text input field
    topic_input = st.text_input(
        "Target Company / Topic to Research:",
        placeholder="e.g., OpenAI feature drops, Stripe payment updates, Nvidia Blackwell GPUs"
    )
    
    # Action button
    submit_button = st.button("Launch Agent Crew", use_container_width=True)

with col2:
    st.subheader("Engine Output")
    
    # Trigger agent execution upon button click
    if submit_button:
        if not topic_input.strip():
            st.warning("Please enter a valid topic or company name first!")
        else:
            # Display a professional loading spinner while the agents are working
            with st.spinner(f"Agents are actively searching the web and analyzing '{topic_input}'... This takes about 30-40 seconds."):
                try:
                    # Pass the user input dynamically into the crew execution pipeline
                    inputs = {"topic": topic_input}
                    result = crew.kickoff(inputs=inputs)
                    
                    # Display the final result cleanly on the web page
                    st.success("Analysis Complete!")
                    st.markdown(result.raw)  # Displays the generated LinkedIn post markdown beautifully
                    
                except Exception as e:
                    st.error(f"An error occurred during agent execution: {e}")
    else:
        st.info("Enter a topic on the left and click 'Launch Agent Crew' to generate insights.")

