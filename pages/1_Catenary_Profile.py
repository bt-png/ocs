import time
import pandas as pd
import numpy as np
import altair as alt

import streamlit as st
import system as OCS
import general as GenFun

if 'accesskey' not in st.session_state:
    st.session_state.accesskey = ''
else:
    st.session_state.accesskey = st.session_state.accesskey

st.set_page_config(
    page_title = "CAT SAG", 
    page_icon = "📹",
    layout = 'wide')

Nom = None
Ref = None
ec = None


def plotdimensions(staList,elList,yscale=1):
    max_x = max(staList) - min(staList)
    max_y = max(elList) - min(elList)
    width = 1200
    widthratio = width/max_x
    height = widthratio*yscale*max_y
    height = max(height, 300)
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
def PlotSag(_REF_Base, _REF=pd.DataFrame(), yscale=1) -> None:
    if _REF.empty:
        df = _REF_Base.dataframe_w_ha()
    else:
        dfbase = _REF_Base.dataframe_w_ha()
        dfbase.type = 'BASE'
        dfalt = _REF.dataframe()
        df = pd.concat([dfbase, dfalt], ignore_index=True)
    df['FtInText'] = df['Elevation'].apply(lambda x : GenFun.FtIn_Simple(x))
    pwidth, pheight = plotdimensions(df['Stationing'],df['Elevation'],yscale)
    st.write('### Catenary Wire Sag Plot')
    nearest = alt.selection_point(nearest=True, on='mouseover', fields=['Stationing'], empty=False)
    line = alt.Chart(df).mark_line().encode(
        alt.X('Stationing:Q').scale(zero=False), 
        alt.Y('Elevation:Q').scale(zero=False),
        alt.Detail('cable'),
        alt.Color('type'),
    )
    selectors = alt.Chart(df).mark_point().encode(
        alt.X('Stationing:Q'),
        opacity=alt.value(0)
    ).add_params(nearest)
    points = line.mark_point().encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
    text = line.mark_text(align='left', dx=5, dy=-5).encode(text=alt.condition(nearest, 'FtInText:N', alt.value(' ')))
    rules = alt.Chart(df).mark_rule(color='gray').encode(x='Stationing:Q').transform_filter(nearest)
    chart = alt.layer(line + selectors + points + rules + text).properties(width=pwidth, height=pheight).interactive()
    st.write(chart)

@st.cache_data()
def PlotHALength(_REF) -> None:
    df = _REF.dataframe_halength()
    df['FtInText'] = df['Length'].apply(lambda x : GenFun.FtIn_Simple(x/12))
    pwidth, pheight = plotdimensions(df['Stationing'],df['Length'])
    st.write('### Hanger Lengths')
    nearest = alt.selection_point(nearest=True, on='mouseover', fields=['Stationing'], empty=False)
    line = alt.Chart(df).mark_point().encode(
        alt.X('Stationing:Q').scale(zero=False), 
        alt.Y('Length:Q', title='HA Length (in)').scale(zero=False),
        alt.Detail('cable'),
        alt.Color('type')
    )
    selectors = alt.Chart(df).mark_point().encode(
        alt.X('Stationing:Q'),
        opacity=alt.value(0)
    ).add_params(nearest)
    points = line.mark_point().encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
    text = line.mark_text(align='left', dx=5, dy=-5).encode(text=alt.condition(nearest, 'FtInText:N', alt.value(' ')))
    rules = alt.Chart(df).mark_rule(color='gray').encode(x='Stationing:Q').transform_filter(nearest)
    chart = alt.layer(line + selectors + points + rules + text).properties(width=pwidth, height=300).interactive()
    st.write(chart)

@st.cache_data()
def ShowHATable(_REF) -> None:
    df = Nom.dataframe_ha()
    df_spt = _REF.dataframe_spt()
    st.write('### BASE Design: Hanger Table')
    st.dataframe(df)
    a, b, minHAL = GenFun.FtIn(min(df['HA Length']))
    a, b, maxHAL = GenFun.FtIn(max(df['HA Length']))
    spans_HA = df.iloc[:,0].diff(periods=1).abs()
    spans_MW = df_spt.loc[df_spt['type'] == 'MW', 'Stationing'].reset_index(drop=True).diff(periods=1).abs()
    LEq_HA = GenFun.EquivalentSpan(spans_HA)
    LEq_MW = GenFun.EquivalentSpan(spans_MW)
    st.caption(f"Min/Max HA Lengths (ft-in)= {minHAL}/{maxHAL} \
               \n  Min/Max HA Load (lb)= {format(min(df['HA Load']),'n')}/{format(max(df['HA Load']),'n')} \
               \n  Min/Max HA Spacing (ft) = {np.nanmin(spans_HA)}/{np.nanmax(spans_HA)} \
               \n  Min/Max Span Spacing (ft) = {np.nanmin(spans_MW)}/{np.nanmax(spans_MW)} \
               \n  MW/CW Equivalent Spans (ft) = {format(round(LEq_MW,0),'n')}/{format(round(LEq_HA,0),'n')}")

@st.cache_data()
def ShowResults(_REF_Base, _REF=pd.DataFrame()):
    if _REF.empty:
        df = _REF_Base.dataframe_spt()
        df.type = 'BASE'
    else:
        dfbase = _REF_Base.dataframe_spt()
        dfbase.type = 'BASE'
        dfalt = _REF.dataframe_spt()
        df = pd.concat([dfbase, dfalt], ignore_index=True)
    sindex = ['Load Condition', 'MW Max (lb)', 'MW Min (lb)', 'HA Max (lb)', 'HA Min (lb)']
    df_results = pd.DataFrame({
        'Load Condition': [],
        'MW Max (lb)': [],
        'MW Min (lb)': [],
        'HA Max (lb)': [],
        'HA Min (lb)': [],})
    for ty in df['type'].unique():
        ser = pd.DataFrame(np.array([[
            ty, 
            format(max(df.loc[(df['type'] == ty) & (df['cable'] == 'MW'), 'Load']),'n'), 
            format(min(df.loc[(df['type'] == ty) & (df['cable'] == 'MW'), 'Load']),'n'), 
            format(max(df.loc[(df['type'] == ty) & (df['cable'] == 'HA'), 'Load']),'n'), 
            format(min(df.loc[(df['type'] == ty) & (df['cable'] == 'HA'), 'Load']),'n'), 
            ]]), columns=sindex)
        df_results = pd.concat([df_results, ser], ignore_index=False)
    st.dataframe(df_results, hide_index=True)
        
@st.cache_data()
def PlotCWDiff(_REF) -> None:
    if not _REF.empty:
        df = _REF.dataframe_cwdiff()
        df['FtInText'] = df['Elevation'].apply(lambda x : GenFun.FtIn_Simple(x))
        df['Elevation'] *= 12
        pwidth, pheight = plotdimensions(df['Stationing'],df['Elevation'])
        st.write('### CW Elevation difference from BASE')
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
        text = line.mark_text(align='left', dx=5, dy=-5).encode(text=alt.condition(nearest, 'FtInText:N', alt.value(' ')))
        rules = alt.Chart(df).mark_rule(color='gray').encode(x='Stationing:Q').transform_filter(nearest)
        chart = alt.layer(line + selectors + points + rules + text).properties(width=pwidth, height=300).interactive()
        st.write(chart)

@st.cache_data()
def PlotSupportLoad(_REF_Base, _REF=pd.DataFrame()) -> None:
    df_temp = _REF_Base.dataframe()
    if _REF.empty:
        df = _REF_Base.dataframe_spt()
    else:
        dfbase = _REF_Base.dataframe_spt()
        dfbase.type = 'BASE'
        dfalt = _REF.dataframe_spt()
        df = pd.concat([dfbase, dfalt], ignore_index=True)
    pwidth, pheight = plotdimensions(df['Stationing'],df_temp['Elevation'])
    st.write('### Support Loading')
    
    nearest = alt.selection_point(nearest=True, on='mouseover', fields=['Stationing'], empty=False)
    line = alt.Chart(df).mark_point().encode(
        alt.X('Stationing:Q').scale(zero=False), 
        alt.Y('Load:Q').scale(zero=False),
        alt.Detail('cable'),
        alt.Color('type')
    )
    selectors = alt.Chart(df).mark_point().encode(
        alt.X('Stationing:Q'),
        opacity=alt.value(0)
    ).add_params(nearest)
    points = line.mark_point().encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
    text = line.mark_text(align='left', dx=5, dy=-5).encode(text=alt.condition(nearest, 'Load:Q', alt.value(' ')))
    rules = alt.Chart(df).mark_rule(color='gray').encode(x='Stationing:Q').transform_filter(nearest)
    chart = alt.layer(line + selectors + points + rules + text).properties(width=pwidth, height=300).interactive()
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
    
def SPTID_range(num1, num2):
    st.markdown(
        'Calculation range is ' + str(abs(wr.loc[num2, 'STA']-wr.loc[num1, 'STA'])) +
        '\' from ' +
        'Pole: ' + str(wr.loc[num1,'PoleID']) + 
        '(STA: ' + str(wr.loc[num1, 'STA']) + ')'
        ' to ' +
        'Pole: ' + str(wr.loc[num2,'PoleID']) + 
        '(STA: ' + str(wr.loc[num2, 'STA']) + ')'
        )

@st.experimental_fragment
def show_download_CAD_Script(_Ref, _yExag):
    acadScript1 = OCS.SagtoCAD(_Ref, _yExag)
    st.download_button(
                        label="### " + str(_yExag) + ":1 scale",
                        data=acadScript1,
                        file_name="_sag_" + str(_yExag) + ".1.scr",
                        mime="text/scr"
                    )


st.markdown("# Simple Catenary Sag")
#st.sidebar.header("Profile Tool")

st.write(
    """This app shows you the simple sag curves for a flexible catenary system
    utilizing a **sum of moments** method. The CW is assumed to be supported only
    at hanger locations, with elevations calculated based on designed elevation at 
    supports and the designated pre-sag."""
    )

if st.secrets['accesskey'] not in st.session_state['accesskey']:
        st.stop()

tab1, tab2, tab3, tab4 = st.tabs(['Input', 'Sag Plot', 'Elasticity', 'Output'])

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
            ddData.clear()
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
        startSPT, endSPT = st.slider(label='Portion of the full span to investigate', min_value=0, max_value=len(wr)-1, value = (2,4), key='spanslider')
        SPTID_range(startSPT, endSPT)
        conditions = len(new_df_acd)+1
        stepSize = _df_bd.iloc[0,1]
        range = endSPT-startSPT
        calcspan=False
        if range == 0:
            st.warning('You need to include a minimum of one support')
            calcspan = True
        else:
            steps = (abs(wr.loc[endSPT, 'STA']-wr.loc[startSPT, 'STA']))/stepSize
            estCalcTime = ((0.0116 * conditions) + (0.00063)) * steps #process time for iterative + base
            m, s = divmod(estCalcTime, 60)
            if m > 0:
                st.warning('###### This could take ' + '{:02.0f} minute(s) {:02.0f} seconds'.format(m, s))
            else:
                st.markdown('###### Estimated compute time is ' + '{:02.0f} seconds'.format(s))
        submit_altCond = st.button('Calculate', key="calcAltCond", disabled=calcspan)
        verified_alt_conditions = False
        if submit_altCond:
            st_time = time.time()
            submit_altCond = False
            SagData.clear()
            PlotSag.clear()
            PlotSupportLoad.clear()
            ShowResults.clear()
            PlotHALength.clear()
            ShowHATable.clear()
            altSagData.clear()
            PlotCWDiff.clear()
            tmp_wr = wr.iloc[startSPT:endSPT+1]
            tmp_wr = tmp_wr.reset_index(drop=True)
            Nom = SagData(_dd, tmp_wr)
            tmp = Nom._solve()
            Ref = pd.DataFrame() # Placeholder for 
            if not new_df_acd.empty:
                verified_alt_conditions = True
                Ref = altSagData(new_df_acd, Nom)
                tmp = Ref._solve()
            et_time = time.time()
            m, s = divmod(et_time-st_time, 60)
            msg = 'Done!' + ' That took ' + '{:02.0f} minute(s) {:02.0f} seconds'.format(m, s)
            st.success(msg)
            if Nom is not None:
                PlotSag(_REF_Base=Nom, _REF=Ref, yscale=yExagg)
                PlotSupportLoad(_REF_Base=Nom, _REF=Ref)
                ShowResults(Nom, Ref)
                PlotHALength(Nom,)
                PlotCWDiff(Ref)
                ShowHATable(Nom)
    elif wrfile is None and ddfile is None:
        SagData.clear()
        PlotSag.clear()
        PlotSupportLoad.clear()
        PlotHALength.clear()
        ShowHATable.clear()
        Nom = SagData(OCS.sample_df_dd(), OCS.sample_df_wr())
        PlotSag(_REF_Base=Nom, yscale=yExagg)
        PlotSupportLoad(_REF_Base=Nom)
        PlotHALength(Nom)
        ShowHATable(Nom)
    else:
        tab2warning = st.warning('Provide both "System Design" and "Layout Design" files to run calculation.')
with tab3:
    if ddfile is not None and wrfile is not None:
        ec = None
        st.write('Consider load conditions')
        new_df_acd_elastic = dataframe_with_selections(OCS.add_base_acd(_df_cd, _df_acd), 'elasticdf')
        startSPT, endSPT = st.slider(label='Portion of the full span to investigate', min_value=0, max_value=len(wr)-1, value = (2,4), key='elasticslider')
        SPTID_range(startSPT, endSPT)
        spanBuffer = st.number_input(label='Span Buffer', min_value = 0, step=1, value=3)
        bufferStart = max(0,startSPT-spanBuffer)
        bufferEnd = min(len(wr)-1, endSPT+spanBuffer)
        stepSize = st.number_input(label='Resolution (ft)', min_value = 1, step=1, value=10)
        pUplift = st.number_input(label='Uplift Force (lbf)', value=25)
        conditions = len(new_df_acd_elastic)
        range = endSPT-startSPT
        calcspan=False
        if range == 0:
            st.warning('You need to include a minimum of one support')
            calcspan = True
        else:
            steps = (wr.loc[endSPT, 'STA']-wr.loc[startSPT, 'STA'])/stepSize
            estCalcTime = 0.789 * conditions * steps
            m, s = divmod(estCalcTime, 60)
            if m > 3:
                st.warning('###### This could take ' + '{:02.0f} minute(s) {:02.0f} seconds'.format(m, s)) #, ', check back at', time.strftime("%H:%M", estEndTime))
            else:
                st.markdown('###### Estimated compute time is ' + '{:02.0f} minute(s) {:02.0f} seconds'.format(m, s))
        submit_elastic = st.button('Calculate', key="calcElasticity", disabled=calcspan)
        if submit_elastic:
            submit_elastic = False
            st_time = time.time()
            #Make the wire run extend up to 3 spans further
            tmp_wr = wr.iloc[bufferStart:bufferEnd+1]
            tmp_wr = tmp_wr.reset_index(drop=True)
            if len(new_df_acd_elastic) > 0:
                SagData.clear()
                elasticityalt.clear()
                Nom = SagData(_dd, tmp_wr)
                ec = elasticityalt(new_df_acd_elastic, Nom, pUplift, stepSize, startSPT-bufferStart, endSPT-bufferStart)
                et_time = time.time()
                m, s = divmod(et_time-st_time, 60)
                msg = 'Done!' + ' That took ' + '{:02.0f} minute(s) {:02.0f} seconds'.format(m, s)
                st.success(msg)
            else:
                st.error('Please select at least one load condition')
        if ec is not None:
            Plotelasticity(ec)

with tab4:
    if ddfile is not None and wrfile is not None:
        cdd0, cdd1, cdd2 = st.columns([0.1, 0.5, 0.4])
        with cdd1:
            if Nom is not None:
                OutputSag(Nom)
            elif ec is not None:
                Outputelasticity(ec)
            else:
                st.warning('Perform a calculation and the download data will be available.')
        with cdd2:
            if Nom is not None:
                st.markdown('#### Download CAD Scripts')

                a1, b1, c1 = st.columns([0.4, 0.3, 0.3])
                with a1:
                    st.write('Cable Profile')
                with b1:
                    show_download_CAD_Script(Nom, 1)
                if yExagg != 1:
                    with c1:
                        show_download_CAD_Script(Nom, yExagg)
