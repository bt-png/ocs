# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
import pandas as pd

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="STV SYSTEMS OCS",
        page_icon="🚊",
    )

    st.write("# Welcome to OCS Calculation Templates!")

    st.sidebar.success("Select a calculation set above.")

    st.markdown(
        """
        These calculation sets are developed by Brett Tharp and
        made available as an open-source app framework built 
        using Python and Streamlit.
        <br>**👈 Select a calculation set from the sidebar** to get started!
        ### What to expect?
        - Select a calculation set from the sidebar.
        - The calculation will be shown using **template** input data so you 
        may have an idea of the purpose for the calculation.
        - You may utilize the sidebar and main frame input fields to customize
        the input data for your needs.
        - The applications are instance based and nothing will be stored.
        You will be responsible to **export** any data generated that you
        would like to reference for the future.
        - Once a selection has been made to the calculation set you would
        like to use on the sidebar.You may utilize the input fields in the sidebar
        - Ask a question through teams to brett.tharp@stvinc.com
    """
    )
    st.write(df.tail(2))

df = pd.DataFrame({
'Updated By': [BMT, BMT, BMT, BMT],
'Updated On': ['3/2/2024' ,'3/3/2024' ,'3/4/20243, '3/5/2024']
})

if __name__ == "__main__":
    run()
