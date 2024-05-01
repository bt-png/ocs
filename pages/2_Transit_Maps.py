import streamlit as st
import pandas as pd
import geopandas
import folium
from streamlit_folium import st_folium

st.set_page_config(layout="wide")

CENTER_START = [39.949610, -75.150282]
ZOOM_START = 8

@st.experimental_fragment
def filter_dataframe(df: pd.DataFrame) -> None:
    df = df.copy()
    modification_container = st.container()
    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            user_cat_input = right.multiselect(
                f"Values for {column}",
                df[column].unique(),
                default=list(df[column].unique()),
            )
            df = df[df[column].isin(user_cat_input)]
    st.dataframe(
        df, 
        hide_index=True, 
        column_config={
            'route_url': st.column_config.LinkColumn(
                label='Link',
                width='small'
                ),
            'start_time': st.column_config.TimeColumn(
                format='hh:mm:ss a'
                ),
            'peak_start_time': st.column_config.TimeColumn(
                format='hh:mm:ss a'
                ),
            'peak_end_time': st.column_config.TimeColumn(
                format='hh:mm:ss a'
                ),
            
            }
        )
    
def load_GeoJSON(_route, _country):
    val = st.session_state["GeoJSON"]
    path = 'sample_data\\saved_geometry_' + _country + '_' + str(_route) +'.geojson'
    try:
        new_val = geopandas.read_file(path)
        if val.empty:
            val = new_val
        else:
            val = pd.concat([val,new_val])
            val = val.reset_index(drop=True)
        st.session_state["GeoJSON"] = val
    except:
        pass

if "center" not in st.session_state:
    st.session_state["center"] = [39.949610, -75.150282]
if "zoom" not in st.session_state:
    st.session_state["zoom"] = 8
if "GeoJSON" not in st.session_state:
    st.session_state["GeoJSON"] = []

st.header('Data generated from [The Mobility Database](https://mobilitydatabase.org/)')
st.caption('data cached on 05/01/2024')
s_country = st.multiselect(label='Select the Country', options=['CA', 'MX', 'US'])
s_route_codes = (0, 1, 2, 11) #, 3, 4, 5, 6, 7, 11, 12)
s_route_desc = (
    '0 - Tram, Streetcar, Light rail. Any light rail or street level system within a metropolitan area.',
    '1 - Subway, Metro. Any underground rail system within a metropolitan area.',
    '2 - Rail. Used for intercity or long-distance travel.',
    '3 - Bus. Used for short- and long-distance bus routes.',
    '4 - Ferry. Used for short- and long-distance boat service.',
    '5 - Cable tram. Used for street-level rail cars where the cable runs beneath the vehicle (e.g., cable car in San Francisco).',
    '6 - Aerial lift, suspended cable car (e.g., gondola lift, aerial tramway). Cable transport where cabins, cars, gondolas or open chairs are suspended by means of one or more cables.',
    '7 - Funicular. Any rail system designed for steep inclines.',
    'empty',
    'empty',
    'empty',
    '11 - Trolleybus. Electric buses that draw power from overhead wires using poles.',
    '12 - Monorail. Railway in which the track consists of a single rail or a beam.',
    )
s_routes = st.multiselect(label='Select the Transit Type', options=s_route_codes, format_func=lambda x: s_route_desc[x])

if len(s_country) > 0 and len(s_routes)>0:
    if st.button('Load Filtered'):
        val = geopandas.GeoDataFrame()
        st.session_state["GeoJSON"] = val
        for country in s_country:
            for route in s_routes:
                load_GeoJSON(route, country)

m = folium.Map(location=CENTER_START, zoom_start=8)

if len(st.session_state["GeoJSON"])>0:
    folium.GeoJson(
        data=st.session_state["GeoJSON"],
        style_function=lambda x: {
            'color': '#' + str(x['properties']['route_color'])
            },
        tooltip=folium.GeoJsonTooltip(
            fields=['route_type', 'agency_id', 'route_long_name'],
            aliases=['Type', 'Agency', 'Route'],
            localize=True,
            sticky=False,
            labels=True
            ),
        popup = folium.GeoJsonPopup(
            fields=['route_type', 'agency_id', 'route_short_name', 'route_long_name', 'route_url'],
            aliases=['Type', 'Agency', 'Line Name', 'Full Line Name', 'URL'],
            localize=True,
            labels=True
            )
        ).add_to(m)    

st_folium(
    m,
    center=st.session_state["center"],
    zoom=st.session_state["zoom"],
    key="new",
    height=700,
    width=1200,
    returned_objects=[]
)

if len(st.session_state["GeoJSON"])>0:
    _df = st.session_state["GeoJSON"].copy()
    _df = _df.drop(['geometry', 'route_desc', 'route_sort_order', 'date'], axis=1)
    _df = _df.dropna(axis=1, how='all')
    filter_dataframe(_df)
#'route_id', 'agency_id', 'route_short_name_x', 'route_long_name', 'route_type_x', 'route_url', 'route_color', 'route_text_color', 'route_sort_order', 'network_id', 'direction0_name', 'direction1_name', 'route_group', 'route_pattern1', 'route_pattern2', 'route_short_name_y', 'route_type_y', 'num_trips', 'num_trip_starts', 'num_trip_ends', 'is_loop', 'is_bidirectional', 'start_time', 'end_time', 'max_headway', 'min_headway', 'mean_headway', 'peak_num_trips', 'peak_start_time', 'peak_end_time', 'service_distance', 'service_duration', 'service_speed', 'mean_trip_distance', 'mean_trip_duration', 'date', 'route_desc', 'ext_route_type', 'route_division', 'alt_route_type', 'route_fare_class', 'line_id', 'listed_route', 'as_route', 'min_headway_minutes', 'eligibility_restricted', 'continuous_pickup', 'continuous_drop_off', 'tts_route_short_name', 'tts_route_long_name'