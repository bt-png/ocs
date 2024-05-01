import streamlit as st
import conductor
import wiring

if 'accesskey' not in st.session_state:
    st.session_state.accesskey = ''
else:
    st.session_state.accesskey = st.session_state.accesskey


st.set_page_config(
    page_title="Simple Calcs", 
    page_icon="🚋",
    )

st.markdown("# Simple Reference Calculations")
#st.sidebar.header("Simple Calculations")
st.write(
    """Use this app for simple reference calculations.  
    You are welcome!."""
)

selection = st.selectbox(
    label='Reference Type', 
    options=('Conductor Data', 'Wiring Plan', 'Clearances', 'Cantilever Resetting Forces'), 
    placeholder='Select calculation method...',
    index=None, 
    label_visibility='hidden'
    )
st.markdown('''---''')
match selection:
    case 'Conductor Data':
        conductor.run()
    case 'Wiring Plan':
        st.write('Coming soon...')
    case 'Clearances':
        wiring.run()
    case 'Cantilever Resetting Forces':
        st.write('Coming soon...')
    case _:
        st.warning('Please make a selection')
