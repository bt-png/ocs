import streamlit as st
import pandas as pd

if 'accesskey' not in st.session_state:
    st.session_state.accesskey = ''
else:
    st.session_state.accesskey = st.session_state.accesskey

from pathlib import Path
from streamlit.source_util import (
    page_icon_and_name, 
    calc_md5, 
    get_pages,
    _on_pages_changed
)

def delete_page(main_script_path_str, page_name):

    current_pages = get_pages(main_script_path_str)

    for key, value in current_pages.items():
        if value['page_name'] == page_name:
            del current_pages[key]
            break
        else:
            pass
    _on_pages_changed.send()


def run():
    st.set_page_config(
        page_title='OCS Calculations',
        page_icon='🚊'
    )
    delete_page('Hello.py', "a_st_Video_Workflow_Process")
    st.write('# Welcome to OCS Calculation Templates!')
    st.markdown(
        """
        These calculation sets are developed by Brett Tharp and
        made available as an open-source app framework built 
        using Python and Streamlit.  
        
        ### What to expect?
        **👈 Select a calculation set from the sidebar** to get started!
        - You may utilize sidebar and main frame input fields to customize
        the input data for your needs.
        - The applications are instance based and nothing will be stored.
        You will be responsible to **export** any data generated that you
        would like to reference for the future.
        - Ask a question through email BrettT_OCS@outlook.com
    """
    )
    st.sidebar.text_input('access key', key='accesskey', type='password')
    if st.session_state['accesskey'] in st.secrets['accesskey']:
        st.write('#### access granted')
    st.write(df.tail(2))

df = pd.DataFrame({
'Updated By': ['BMT', 'BMT', 'BMT', 'BMT', 'BMT', 'BMT', 'BMT'],
'Updated On': ['3/2/2024' ,'3/3/2024' ,'3/4/20243', '3/5/2024', '3/13/2024', '4/3/2024', '4/24/2024']
})

if __name__ == '__main__':
    run()
