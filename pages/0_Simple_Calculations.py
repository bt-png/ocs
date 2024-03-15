import streamlit as st

st.session_state.accesskey = st.session_state.accesskey

st.set_page_config(page_title="Simple Calcs", page_icon="📹")
st.markdown("# Simple Reference Calculations")
st.sidebar.header("Simple Calculations")
st.write(
    """Use this app for simple reference calculations.  
    You are welcome!."""
)

if st.session_state['accesskey'] != st.secrets['accesskey']:
    st.stop()

conductor, wiring, resetting = st.tabs(['Conductor Data', 'Wiring Plan', 'Cantilever Resetting Force'])
