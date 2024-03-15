import time
import pandas as pd
import numpy as np
import altair as alt

import streamlit as st
import system as OCS
import general as GenFun

st.session_state.accesskey = st.session_state.accesskey
Nom = None
Ref = None
ec = None

#st.cache_data.clear()
def plotdimensions(staList,elList,yscale=1):
    max_x = max(staList) - min(staList)
    max_y = max(elList) - min(elList)
    width = 1200
    widthratio = width/max_x
    height = widthratio*yscale*max_y
    return int(width), int(height)

def dataframe_with_selections(df, inputkey):
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)
    df_with_selections.iloc[0,0] = True
    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=df.columns,
        key=inputkey
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop('Select', axis=1)

@st.cache_data()
def ddData(dd):
    return OCS.create_df_dd(dd)

@st.cache_data()
def wrData(wr):
    return OCS.wire_run(wr)

@st.cache_data()
def SagData(val, wirerun) -> None:
    return OCS.CatenaryFlexible(val, wirerun)
    
@st.cache_data()
def altSagData(val,  _ORIG) -> None:
    return OCS.AltCondition_Series(val, _ORIG)

@st.cache_data()
def elasticity(val, _BASE, pUplift, stepSize, startSPT, endSPT) -> None:
    EL = OCS.Elasticity(val, _BASE, pUplift, stepSize, startSPT, endSPT)
    df = EL.dataframe()
    return df

@st.cache_data()
def elasticityalt(val, _BASE, pUplift, stepSize, startSPT, endSPT) -> None:
    EL = OCS.Elasticity_series(val, _BASE, pUplift, stepSize, startSPT, endSPT)
    df = EL.dataframe()
    return df

@st.cache_data()
def preview_ddfile(ddfile) -> None:
    cdd0, cdd1, cdd2 = st.columns([0.1, 0.6, 0.3])
    with cdd1:
        #st.write(_dd)
        st.write('###### Loaded conductor particulars data:')
        if bool:
            st.dataframe(_df_acd, hide_index=True)
        else:
            st.dataframe(_df_cd, hide_index=True)
        st.write('###### Loaded span point loads:')
        st.dataframe(_df_sl, hide_index=True)
    with cdd2:
        st.write('###### Loaded hanger design variables:')
        st.dataframe(_df_hd, hide_index=True)
        st.write('###### Loaded calculation constants:')
        st.dataframe(_df_bd, hide_index=True)

@st.cache_data()
def preview_wrfile(wrfile) -> None:
    cwr0, cwr1 = st.columns([0.1, 0.9])
    with cwr1:
        st.write('###### First several rows of input file:')
        st.dataframe(wr.head(), hide_index=True)  

@st.cache_data()
def PlotSag(_REF, yscale) -> None:
    df = _REF.dataframe()
    pwidth, pheight = plotdimensions(df['Stationing'],df['Elevation'],yscale)
    st.write('### Catenary Wire Sag Plot')
    selection = alt.selection_point(fields=['type'], bind='legend')
    chart = alt.Chart(df).mark_line().encode(
        alt.X('Stationing:Q').scale(zero=False), 
        alt.Y('Elevation:Q').scale(zero=False),
        alt.Detail('cable'),
        alt.Color('type'),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.1))
        ).add_params(selection).properties(
            width=pwidth,
            height=pheight
        ).interactive()
    st.write(chart)

@st.cache_data()
def PlotSagaltCond(_REF, yscale) -> None:
    dfa = _REF.dataframe()
    pwidth, pheight = plotdimensions(dfa['Stationing'],dfa['Elevation'],yscale)
    st.write('### Catenary Wire Sag Plot')
    selection = alt.selection_point(fields=['type'], bind='legend')
    chart = alt.Chart(dfa).mark_line().encode(
        alt.X('Stationing:Q').scale(zero=False), 
        alt.Y('Elevation:Q').scale(zero=False),
        alt.Detail('cable'),
        alt.Color('type'),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.1))
        ).add_params(selection).properties(
            width=pwidth,
            height=pheight
        ).interactive()
    st.write(chart)

@st.cache_data()
def PlotCWDiff(_REF) -> None:
    df = _REF.dataframe_cwdiff()
    df['Elevation'] *= 12
    pwidth, pheight = plotdimensions(df['Stationing'],df['Elevation'])
    st.write('### Difference from BASE')
    nearest = alt.selection_point(nearest=True, on='mouseover', fields=['Stationing'], empty=False)
    line = alt.Chart(df).mark_line().encode(
        alt.X('Stationing:Q').scale(zero=False), 
        alt.Y('Elevation:Q', title='CW Diff (in)').scale(zero=False),
        alt.Detail('cable'),
        alt.Color('type')
    )
    selectors = alt.Chart(df).mark_point().encode(
        alt.X('Stationing:Q'),
        opacity=alt.value(0)
    ).add_params(nearest)
    points = line.mark_point().encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
    text = line.mark_text(align='left', dx=5, dy=-5).encode(text=alt.condition(nearest, 'Elevation:Q', alt.value(' ')))
    rules = alt.Chart(df).mark_rule(color='gray').encode(x='Stationing:Q').transform_filter(nearest)
    chart = alt.layer(line + selectors + points + rules + text).properties(width=pwidth, height=300)
    st.write(chart)
    
@st.cache_data()
def PlotSagSample(_BASE, yscale) -> None:
    df = _BASE.dataframe()
    pwidth, pheight = plotdimensions(df['Stationing'],df['Elevation'],yscale)
    st.markdown('### SAMPLE DATA')
    st.write('### Catenary Wire Sag Plot')
    selection = alt.selection_point(fields=['type'], bind='legend')
    chart = alt.Chart(df).mark_line().encode(
        alt.X('Stationing:Q').scale(zero=False), 
        alt.Y('Elevation:Q').scale(zero=False, type='linear'),
        alt.Detail('cable'),
        alt.Color('type'),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.1))
        ).add_params(selection).properties(
            width=pwidth,
            height=pheight
        ).interactive()
    st.write(chart)

@st.cache_data()
def Plotelasticity(df) -> None:
    st.write('### Elasticity')
    #selection = alt.selection_point(fields=['type'], bind='legend')
    #chart = alt.Chart(df).mark_line().encode(
    #    alt.X('Stationing:Q').scale(zero=False), 
    #    alt.Y('Rise (in):Q').scale(zero=False),
    #    alt.Color('type'),
    #    opacity=alt.condition(selection, alt.value(1), alt.value(0.1))
    #    ).add_params(selection).interactive()
    #st.write(chart)
    pwidth, pheight = plotdimensions(df['Stationing'],df['Rise (in)'],1)
    nearest = alt.selection_point(nearest=True, on='mouseover', fields=['Stationing'], empty=False)
    line = alt.Chart(df).mark_line().encode(
        alt.X('Stationing:Q').scale(zero=False), 
        alt.Y('Rise (in):Q', title='CW Diff (in)').scale(zero=False),
        alt.Detail('cable'),
        alt.Color('type')
    )
    selectors = alt.Chart(df).mark_point().encode(
        alt.X('Stationing:Q'),
        opacity=alt.value(0)
    ).add_params(nearest)
    points = line.mark_point().encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
    text = line.mark_text(align='left', dx=5, dy=-5).encode(text=alt.condition(nearest, 'Rise (in):Q', alt.value(' ')))
    rules = alt.Chart(df).mark_rule(color='gray').encode(x='Stationing:Q').transform_filter(nearest)
    chart = alt.layer(line + selectors + points + rules + text).properties(width=pwidth, height=300)
    st.write(chart)

@st.cache_data()
def OutputSag(_BASE) -> None:
    st.write('#### Sag Data ', _BASE.dataframe())
    st.write('#### HA Data', _BASE.dataframe_ha())

@st.cache_data()
def OutputAltCond(_Ref, _BASE) -> None:
    #LC = altSagData(_df_cd, _df_acd, _BASE)
    dfa = _Ref.dataframe()
    st.write('#### Conductor Data', dfa)
    st.write('#### HA Data', _BASE.dataframe_ha())

@st.cache_data()
def Outputelasticity(df) -> None:
    st.write('#### Elasticity', df)

@st.cache_data()
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

def SPTID(num):
    st.markdown(
        'Pole: ' + str(wr.loc[num,'PoleID']) + 
        ', STA: ' + str(wr.loc[num, 'STA'])
        )

st.set_page_config(
    page_title="CAT SAG", 
    page_icon="📹",
    layout='wide')

st.markdown("# Simple Catenary Sag")
st.sidebar.header("CAT SAG")

st.write(
    """This app shows you the simple sag curves for a flexible catenary system
    utilizing a **sum of moments** method. The CW is assumed to be supported only
    at hanger locations, with elevations calculated based on designed elevation at 
    supports and the designated pre-sag."""
    )

if st.session_state['accesskey'] != st.secrets['accesskey']:
        st.stop()

tab1, tab2, tab3, tab4 = st.tabs(['Input', 'Sag Plot', 'Elasticity', 'Output'])

st.sidebar.checkbox('Consider Alt Conductors', key='altConductors', value=True)
st.sidebar.checkbox('Perform Elasticity Check', key='elasticity', value=True)

with tab1:
    cbd = st.container(border=True)
    cdd = st.container(border=True)
    cdd_val = st.container(border=False)
    cwr = st.container(border=True)
    cwr_val = st.container(border=False)
    with cbd:
        yExagg = st.number_input(label='Y Scale Exaggeration', value=20, min_value=1, step=1)
    with cdd:
        ##Design Data
        st.markdown("#### Load catenary system design data")
        sample_dd_csv = convert_df(OCS.sample_df_dd())
        st.download_button(
            label="### press to download sample csv file for Design data",
            data=sample_dd_csv,
            file_name="_dd_Data.csv",
            mime="text/csv"
        )

        ddfile = st.file_uploader(
                'Upload a properly formatted csv file containing Design data',
                type={'csv', 'txt'},
                accept_multiple_files = False,
                key='_ddfile'
                )
        if ddfile is not None:
            _dd = pd.read_csv(ddfile)
            _df_cd, _df_acd, _df_hd, _df_bd, _df_sl = ddData(_dd)
            preview_ddfile(ddfile)
    with cwr:
        ##Wire Run Data
        st.markdown("#### Load layout design data for wire run")
        sample_wr_csv = convert_df(OCS.sample_df_wr())
        st.download_button(
            label="### press to download sample csv file for Wire Run data",
            data=sample_wr_csv,
            file_name="_WR_Data.csv",
            mime="text/csv"
        )
        wrfile = st.file_uploader(
                'Upload a properly formatted csv file containing Wire Run data',
                type={'csv', 'txt'},
                accept_multiple_files = False,
                key='_wrfile'
                )
        if wrfile is not None:
            wr = wrData(wrfile)
            preview_wrfile(wrfile)
with tab2:
    if ddfile is not None and wrfile is not None:
        st.write('Hanger lengths are driven based on the BASE condition.')
        new_df_acd = dataframe_with_selections(_df_acd, 'plotdataframe')
        #_df_acd_ref = _df_acd.copy()
        #_df_acd_ref.insert(0,'Calculate',False)
        #_df_acd_ref.loc[0,'Calculate'] = True
        #e_df_acd = st.data_editor(_df_acd_ref, hide_index=True, key='1')
        #st.write(_df_acd_ref.iloc[0,:].transpose)
        conditions = len(new_df_acd)
        stepSize = _df_bd.iloc[0,1]
        steps = (max(wr.STA)-min(wr.STA))/stepSize
        estCalcTime = ((0.0116 * conditions) + (0.00063)) * steps #process time for iterative + base
        m, s = divmod(estCalcTime, 60)
        if m > 0:
            st.warning('###### This could take ' + '{:02.0f} minute(s) {:02.0f} seconds'.format(m, s))
        else:
            st.markdown('###### Estimated compute time is ' + '{:02.0f} seconds'.format(s))
        submit_altCond = st.button('Calculate', key="calcAltCond")
        if submit_altCond:
            st_time = time.time()
            SagData.clear()
            Nom = SagData(_dd, wr)
            tmp = Nom._solve()
            PlotSag.clear()
            altSagData.clear()
            Ref = altSagData(new_df_acd, Nom)
            tmp = Ref._solve()
            PlotSagaltCond.clear()
            PlotCWDiff.clear()
            et_time = time.time()
            m, s = divmod(et_time-st_time, 60)
            msg = 'Done!' + ' That took ' + '{:02.0f} minute(s) {:02.0f} seconds'.format(m, s)
            st.success(msg)
        if new_df_acd.empty and Nom is not None:
            PlotSag(Nom, yExagg)
        elif Ref is not None:
            PlotSagaltCond(Ref, yExagg)
            PlotCWDiff(Ref)
    elif wrfile is None and ddfile is None:
        SagData.clear()
        Nom = SagData(OCS.sample_df_dd(), OCS.sample_df_wr())
        PlotSagSample(Nom, yExagg)
    else:
        st.warning('Provide both "System Design" and "Layout Design" files to run calculation.')
with tab3:
    if ddfile is not None and wrfile is not None and st.session_state['elasticity']:
        ec = None
        #_df_acd_ref = _df_acd.copy()
        #_df_acd_ref.insert(0,'Calculate',False)
        st.write('Consider load conditions')
        new_df_acd_elastic = dataframe_with_selections(_df_acd, 'elasticdf')
        #elastic_df_acd = st.data_editor(_df_acd_ref, hide_index=True, key='2')
        startSPT = st.number_input(label='Starting structure', value=1, min_value=0, max_value=len(wr)-2)
        SPTID(startSPT)
        endSPT = st.number_input(label='Ending structure', value=startSPT+1, min_value=1, max_value=len(wr)-1)
        SPTID(endSPT)
        stepSize = st.number_input(label='Resolution (ft)', min_value = 1, step=1, value=10)
        pUplift = st.number_input(label='Uplift Force (lbf)', value=25)
        conditions = len(new_df_acd_elastic)
        steps = (wr.loc[endSPT, 'STA']-wr.loc[startSPT, 'STA'])/stepSize
        estCalcTime = 0.789 * conditions * steps
        #estEndTime = time.localtime(time.mktime(time.localtime()) + estCalcTime)
        m, s = divmod(estCalcTime, 60)
        if m > 3:
            st.warning('###### This could take ' + '{:02.0f} minute(s) {:02.0f} seconds'.format(m, s), ', check back at', time.strftime("%H:%M", estEndTime))
        else:
            st.markdown('###### Estimated compute time is ' + '{:02.0f} minute(s) {:02.0f} seconds'.format(m, s))
        submit_elastic = st.button('Calculate', key="calcElasticity")
        if submit_elastic:
            st_time = time.time()
            SagData.clear()
            Nom = SagData(_dd, wr)
            #new_df_acd_elastic = _df_acd[elastic_df_acd.Calculate]
            if len(new_df_acd_elastic) > 0:
                elasticityalt.clear()
                ec = elasticityalt(new_df_acd_elastic, Nom, pUplift, stepSize, startSPT, endSPT)
                et_time = time.time()
                m, s = divmod(et_time-st_time, 60)
                msg = 'Done!' + ' That took ' + '{:02.0f} minute(s) {:02.0f} seconds'.format(m, s)
                st.success(msg)
            else:
                st.error('Please select at least one load condition')
                #if st.session_state['altConductors']:
                #    ec = elasticityalt(_df_acd, Nom, pUplift, stepSize, startSPT, endSPT)
                #else:
                #    ec = elasticity(_df_cd, Nom, pUplift, stepSize, startSPT, endSPT)
        if ec is not None:
            Plotelasticity(ec)

with tab4:
    if ddfile is not None and wrfile is not None:
        cdd0, cdd1, cdd2 = st.columns([0.1, 0.5, 0.4])
        with cdd1:
            #if st.session_state['altConductors']:
            #    if submit_altCond:
            #        if not new_df_acd.empty:
            #            OutputAltCond(Ref, Nom)
            if submit_altCond:
                OutputSag(Nom)
            elif ec is not None:
                Outputelasticity(ec)
            else:
                st.warning('Perform a calculation and the download data will be available.')
        with cdd2:
            if submit_altCond:
                st.markdown('#### Download CAD Scripts')
                acadScript1 = OCS.SagtoCAD(Nom, 1)
                acadScript2 = OCS.SagtoCAD(Nom, yExagg)
                a1, b1, c1 = st.columns([0.4, 0.3, 0.3])
                with a1:
                    st.write('Cable Profile')
                with b1:
                    bt1 = st.download_button(
                        label="### 1:1 scale",
                        data=acadScript1,
                        file_name="_sag_.scr",
                        mime="text/scr"
                    )
                with c1:
                    bt2 = st.download_button(
                        label="### yExaggeration",
                        data=acadScript2,
                        file_name="_sag_.scr",
                        mime="text/scr"
                    )
