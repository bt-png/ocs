import os
import time
import json
from datetime import timedelta
from shutil import rmtree

import pickle
from io import BytesIO
from PIL import Image
import base64
import streamlit as st
import pandas as pd
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import save_one_box, Annotator, colors
import torch

from st_Video_Workflow import load_model
# from st_Video_Workflow import shift_tensor
# from st_Video_Workflow import video_get_single_frame
# from st_Video_Workflow import test_extract_panview
# from st_Video_Workflow import test_annotate_panview

if 'accesskey' not in st.session_state:
    st.session_state.accesskey = ''
else:
    st.session_state.accesskey = st.session_state.accesskey

def draw_text(img, text,
              font=cv2.FONT_HERSHEY_PLAIN,
              pos=(0, 0),
              font_scale=1,
              font_thickness=1,
              text_color=(255, 255, 255),
              text_color_bg=(0, 0, 0),
              align='left'
              ):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    if align == 'right':
        cv2.rectangle(img, pos, (x - text_w, y + text_h), text_color_bg, -1)
        cv2.putText(
            img, text, (x - text_w, y + text_h + font_scale - 1),
            font, font_scale, text_color, font_thickness
            )
    elif align == 'center':
        cv2.rectangle(img, pos, (x - int(0.5*text_w), y + int(0.5*text_h)), text_color_bg, -1)
        cv2.putText(
            img, text, (x - int(0.5*text_w), y + int(0.5*text_h) + font_scale - 1),
            font, font_scale, text_color, font_thickness
            )
    else:
        cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
        cv2.putText(
            img, text, (x, y + text_h + font_scale - 1),
            font, font_scale, text_color, font_thickness
            )
    return y+text_h+font_scale*3


def update_arc_frame(v_dict, idx, value):
    st.write(idx)
    st.write(value)


def flip_arc_frame(v_dict, idx):
    v_dict['Arc df'].at[idx, 'arc'] = not v_dict['Arc df'].at[idx, 'arc']


def box_to_xywh(box):
    # x1y1x2y2
    x = 0.5 * (box[0] + box[2])
    y = 0.5 * (box[1] + box[3])
    w = box[2] - box[0]
    h = box[3] - box[1]
    return x, y, w, h


def box_within_box(v_dict, main, test):
    x1 = int(main[0] - 0.5 * v_dict['PanView Width'])
    y1 = int(main[1] - 0.5 * main[3] - v_dict['PanView Vert Offset'])
    x2 = int(main[0] + 0.5 * v_dict['PanView Width'])
    y2 = int(y1 + v_dict['PanView Height'])
    test_x = test[0]
    test_y = test[1]
    x_pad_perc = 0.2
    y_pad_perc = 0.2
    if test_x > (x2 - x_pad_perc * v_dict['PanView Width']) or test_x < (x1 + x_pad_perc * v_dict['PanView Width']):
        return False
    if test_y > (y2 - y_pad_perc * v_dict['PanView Height']) or test_y < (y1 + y_pad_perc * v_dict['PanView Height']):
        return False
    return True


def Pantograph_ObjectDetection_Initial(v_dict: dict):
    panmodel = load_model('pantograph.pt')
    arcmodel = load_model('arcing.pt')
    progress_text = 'In Progress. Please wait.'
    progress_bar = st.progress(0.0, text=progress_text)
    stop_button = st.button("Stop")  # Button to stop the inference

    col1, col2 = st.columns(2)
    pan_frame = col1.empty()
    pan_data = col1.empty()
    arc_frame = col2.empty()
    arc_data = col2.empty()

    fps_display = st.empty()  # Placeholder for FPS display

    videocapture = cv2.VideoCapture(v_dict['Video Path'])  # Capture the video
    if not videocapture.isOpened():
        st.error('Could not open video.')
    prev_time = time.time()

    w, h, fps = (int(videocapture.get(x)) for x in (
        cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS
        ))

    # Setup dictionaries to capture the YOLO results
    pan_objects = {}
    arc_objects = {}

    # Codec Options: *'mp4v' Not supported in browsers
    # Codec Options: *'H264', doesnt work directly in Python 3. Use hex = 0x00000021
    pan_video_writer = cv2.VideoWriter(v_dict['Pan_Yolo_file'], 0x00000021, fps, (w, h))
    arc_video_writer = cv2.VideoWriter(v_dict['Arc_Yolo_file'], 0x00000021, fps, (w, h))

    idx = v_dict['Start Frame'] - 1
    while videocapture.isOpened():
        videocapture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = videocapture.read()  # Read the selected frame
        if not success:
            st.warning('Failed to read frame from video.')
            v_dict['Processed through idx'] = idx
            break
        # The object can be called itself, but .predict is more clear
        pan_results = panmodel.predict(frame, conf=v_dict['Pan conf'], iou=v_dict['Pan iou'], verbose=False)
        pan_objects.update({idx: json.loads(pan_results[0].tojson())})
        # pan_data.write(pan_objects)
        pan_im = pan_results[0].plot()
        pan_frame.image(pan_im, channels="BGR")
        pan_video_writer.write(pan_im)
        arc_results = arcmodel.predict(frame, conf=v_dict['Arc conf'], iou=v_dict['Arc iou'], verbose=False)
        arc_objects.update({idx: json.loads(arc_results[0].tojson())})
        # arc_data.write(arc_objects)
        arc_im = arc_results[0].plot()
        arc_frame.image(arc_im, channels="BGR", caption='Object detection will be refined to the pan window in post processing')
        arc_video_writer.write(arc_im)
        if stop_button:
            v_dict['Processed through idx'] = idx
            break
        # Calculate model FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_display.metric("Process Speed in FPS", f"{fps:.2f}")
        idx += 1
        progress_val = (idx-v_dict['Start Frame'])/(v_dict['End Frame'] - v_dict['Start Frame'])
        progress_val = max(0.0, progress_val)
        progress_val = min(1.0, progress_val)
        progress_bar.progress(progress_val, text=progress_text)
        if idx > v_dict['End Frame']:
            v_dict['Processed through idx'] = idx
            break
    fps_display.empty()
    progress_bar.empty()
    videocapture.release()  # Release the capture
    pan_video_writer.release()
    arc_video_writer.release()
    torch.cuda.empty_cache()  # Clear CUDA memory
    cv2.destroyAllWindows()  # Destroy window

    if os.path.exists(v_dict['Pan_Yolo_json']):
        os.unlink(v_dict['Pan_Yolo_json'])
    pan_json_object = json.dumps(pan_objects)
    with open(v_dict['Pan_Yolo_json'], 'w') as f:
        f.write(pan_json_object)
    if os.path.exists(v_dict['Arc_Yolo_json']):
        os.unlink(v_dict['Arc_Yolo_json'])
    arc_json_object = json.dumps(arc_objects)
    with open(v_dict['Arc_Yolo_json'], 'w') as f:
        f.write(arc_json_object)
    st.rerun()


def shift_box(v_dict, main, test):
    x = int(main[0] - 0.5 * v_dict['PanView Width'])
    y = int(main[1] - 0.5 * main[3] - v_dict['PanView Vert Offset'])
    sc = v_dict['PanView Scale']
    test = [
        int((test[0] - x) * sc),
        int((test[1] - y) * sc),
        int((test[2] - x) * sc),
        int((test[3] - y) * sc)
        ]
    return test


def video_get_single_frame(v_dict, idx):
    videocapture = cv2.VideoCapture(v_dict['Video Path'])  # Capture the video
    if not videocapture.isOpened():
        st.error("Could not open video file.")
    videocapture.set(cv2.CAP_PROP_POS_FRAMES, idx)
    success, frame = videocapture.read()  # Read the selected frame
    videocapture.release()  # Release the capture
    cv2.destroyAllWindows()  # Destroy window
    return frame


def test_extract_panview(v_dict, frame, xywh):
    x1 = int(xywh[0] - 0.5 * v_dict['PanView Width'])
    y1 = int(xywh[1] - 0.5 * xywh[3] - v_dict['PanView Vert Offset'])
    x2 = int(xywh[0] + 0.5 * v_dict['PanView Width'])
    y2 = int(y1 + v_dict['PanView Height'])
    icrop = frame[y1 : y2, x1 : x2, :: (-1)]
    gray_image = cv2.cvtColor(
        cv2.cvtColor(icrop, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR
        )
    larger_obj = cv2.resize(
        gray_image,
        (v_dict['PanView ScaleWidth'], v_dict['PanView ScaleHeight']),
        interpolation=cv2.INTER_LINEAR
        )
    return larger_obj


def test_annotate_panview_single(v_dict, frame, arc_row, pan_box):
    annotator = Annotator(frame, line_width=2)
    arc_box = [arc_row[1], arc_row[2], arc_row[3], arc_row[4]]
    annotator.box_label(
        shift_box(v_dict, pan_box, arc_box),
        color=colors(0, True),
        label=arc_row[5])
    return frame


def test_annotate_panview(v_dict, frame, arc_rows, pan_box):
    annotator = Annotator(frame, line_width=2)
    for row in arc_rows.values:
        # st.write(row)
        arc_box = [row[1], row[2], row[3], row[4]]
        annotator.box_label(
            shift_box(v_dict, pan_box, arc_box),
            color=colors(0, True),
            label=row[5])
        # int(list(names.values()).index(cls))
    return frame


def annotate_panview(v_dict, pan_row, arc_rows):
    pan_box = box_to_xywh([pan_row['x1'], pan_row['y1'], pan_row['x2'], pan_row['y2']])
    frame = video_get_single_frame(v_dict, int(pan_row['Frame']))
    frame = test_extract_panview(v_dict, frame, pan_box)
    frame = test_annotate_panview_single(v_dict, frame, arc_rows, pan_box)
    # st.write(pan_row)
    # st.write(arc_rows)
    return frame


def run(video_dict):
    if st.session_state['accesskey'] not in st.secrets['accesskey']:
        st.stop()
    upload_dir = video_dict['Upload Folder']
    base_name, ext = os.path.splitext(video_dict['Video Name'])
    save_folder = os.path.join(upload_dir, base_name)
    Pan_Yolo_name = base_name + '_PanYOLO_' + ext
    Pan_Yolo_file = os.path.join(save_folder, Pan_Yolo_name)
    video_dict['Pan_Yolo_file'] = Pan_Yolo_file
    video_dict['Pan_Yolo_json'] = os.path.join(save_folder, base_name + '_PanYOLO_' + '.json')
    Arc_Yolo_name = base_name + '_ArcYOLO_' + ext
    Arc_Yolo_file = os.path.join(save_folder, Arc_Yolo_name)
    video_dict['Arc_Yolo_file'] = Arc_Yolo_file
    video_dict['Arc_Yolo_json'] = os.path.join(save_folder, base_name + '_ArcYOLO_' + '.json')

    if 'Pan df' in video_dict:
        if type(video_dict['Pan df']) == str:
            video_dict['Pan df'] = pd.DataFrame.from_dict(json.loads(video_dict['Pan df']))
    if 'Arc df' in video_dict:
        if type(video_dict['Arc df']) == str:
            video_dict['Arc df'] = pd.DataFrame.from_dict(json.loads(video_dict['Arc df']))

    col1, col2 = st.columns([1, 9])
    with col1:
        st.subheader('1')
        st.caption('YOLOv8 object detection of the video for Pantograph')
    with col2:
        button = st.empty()
        if button.button('Process Video for Pan and Arc Detection'):
            button.empty()
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            Pantograph_ObjectDetection_Initial(video_dict)  # -> *_YOLO.csv, *_YOLO.mp4
        else:
            col1, col2 = st.columns([1, 1])
            if os.path.exists(Pan_Yolo_file):
                col1.video(Pan_Yolo_file)
                col1.caption('Processed Pan Detection Video')
            if os.path.exists(Arc_Yolo_file):
                col2.video(Arc_Yolo_file)
                col2.caption('Processed Arc Detection Video')
    col1, col2 = st.columns([1, 9])
    with col1:
        st.subheader('2')
        st.caption('Data Analysis and Smoothing')
    with col2:
        button = st.empty()
        # Station = ['RTS', 'SRS', 'DRS']
        # NomHeight = [14.5, 18.5, 15]
        # Route = ['RTS', 'DRS']  # SRS left out as included in both
        # MinHeight = [14, 13 + 11/12] # 13.9166666666666
        # MaxHeight = [18.5, 18.5]
        video_dict['minHeight'] = st.number_input('Minimum CW Height in Video Section', value=video_dict.get('minHeight', 13.5))
        video_dict['maxHeight'] = st.number_input('Maximum CW Height in Video Section', value=video_dict.get('maxHeight', 18.0))
        if button.button('Apply Smoothing to Raw YOLOv8 Data'):
            # Apply Smoothing to Pantograph Detection
            if os.path.exists(video_dict['Pan_Yolo_json']):
                # Load Json Data
                with open(video_dict['Pan_Yolo_json'], 'r') as f:
                    pan_json = json.load(f)
                _df_pan = pd.DataFrame({
                    'Frame': [], 'x1': [], 'y1': [], 'x2': [], 'y2': [], 'arc counts': []
                    })
                for idx in pan_json.keys():
                    i = len(_df_pan.index)
                    try:
                        val = pan_json[idx][0]['box']
                        _df_pan.loc[i] = [idx, val['x1'], val['y1'], val['x2'], val['y2'], 0]
                    except Exception:
                        pass
                # Smooth
                try:
                    _df_pan['pan_x'] = [0.5 * (x1 + x2) for x1, x2 in zip(_df_pan['x1'], _df_pan['x2'])]
                    _df_pan['pan_y'] = [0.5 * (y1 + y2) for y1, y2 in zip(_df_pan['y1'], _df_pan['y2'])]
                    _df_pan['pan_w'] = [x2 - x1 for x1, x2 in zip(_df_pan['x1'], _df_pan['x2'])]
                    _df_pan['pan_h'] = [y2 - y1 for y1, y2 in zip(_df_pan['y1'], _df_pan['y2'])]
                except Exception:
                    pass
                _df_pan['Frame'] = _df_pan['Frame'].astype(int)
                _df_pan.set_index('Frame', inplace=True, drop=False)
                _df_pan = _df_pan.reindex(np.arange(int(video_dict['Start Frame']), int(video_dict['End Frame']+1), 1))
                for idx, row in _df_pan.iterrows():
                    r = np.arange(max(int(video_dict['Start Frame']), idx-10), min(int(video_dict['End Frame']+1), idx+10), 1)
                    _df = _df_pan.loc[_df_pan.index.isin(r)]
                    if not _df.empty:
                        _df_pan.at[idx, 'pan_y_smooth'] = _df['pan_y'].mean()
                        _df_pan.at[idx, 'pan_x_smooth'] = _df['pan_x'].mean()
                # Apply CW Heights
                # MIN
                min_pany = _df_pan[_df_pan['pan_y_smooth'] <= _df_pan['pan_y_smooth'].quantile(0.95)]['pan_y_smooth'].quantile(.95)
                idx = _df_pan['pan_y_smooth'].sub(min_pany).abs().idxmin()
                _df_pan.at[idx, 'CW_Height'] = video_dict['minHeight']
                # MAX
                max_pany = _df_pan[_df_pan['pan_y_smooth'] <= _df_pan['pan_y_smooth'].quantile(0.05)]['pan_y_smooth'].quantile(.05)
                idx = _df_pan['pan_y_smooth'].sub(max_pany).abs().idxmin()
                _df_pan.at[idx, 'CW_Height'] = video_dict['maxHeight']
                video_dict['Pan df'] = _df_pan
                st.success('Smoothing Data Updated')
                # st.dataframe(_df_pan)
                # if os.path.exists(video_dict['Pan_Yolo_json']):
                #     os.unlink(video_dict['Pan_Yolo_json'])
                # pan_json_object = json.dumps(_df_pan)
                # with open(video_dict['Pan_Yolo_json'], 'w') as f:
                #     f.write(pan_json_object)
        button = st.empty()
        status = st.empty()
        iframe = st.empty()
        if button.button('Filter Arcing'):
            # Filter Arcing to Pantograph Box
            if os.path.exists(video_dict['Pan_Yolo_json']) and os.path.exists(video_dict['Arc_Yolo_json']):
                with open(video_dict['Pan_Yolo_json'], 'r') as f:
                    pan_json = json.load(f)
                with open(video_dict['Arc_Yolo_json'], 'r') as f:
                    arc_json = json.load(f)
                _df_pan = pd.DataFrame({
                    'Frame': [], 'x1': [], 'y1': [], 'x2': [], 'y2': [], 'arc counts': []
                    })
                _df_arc = pd.DataFrame({
                    'Frame': [], 'x1': [], 'y1': [], 'x2': [], 'y2': [], 'item': [], 'image': [], 'arc': []
                    })
                for idx in pan_json.keys():
                    i = len(_df_pan.index)
                    status.write(f'Pan Frame {i}')
                    try:
                        val = pan_json[idx][0]['box']
                        _df_pan.loc[i] = [idx, val['x1'], val['y1'], val['x2'], val['y2'], 0]
                        pan_box = box_to_xywh([val['x1'], val['y1'], val['x2'], val['y2']])
                        for ix, item in enumerate(arc_json.get(idx)):
                            status.write(f'Pan Frame {i}, Arc Item {ix}')
                            val = item['box']
                            arc_box = box_to_xywh([val['x1'], val['y1'], val['x2'], val['y2']])
                            if box_within_box(video_dict, pan_box, arc_box):
                                j = len(_df_arc.index)
                                _df_pan.at[i, 'arc counts'] += 1
                                _df_arc.loc[j] = [idx, val['x1'], val['y1'], val['x2'], val['y2'], item['name'], bytes(0), False]
                                image = annotate_panview(video_dict, _df_pan.loc[i], _df_arc.loc[j])
                                iframe.image(image)
                                _df_arc.at[j, 'image'] = image.dumps()
                    except Exception:
                        pass
                        # st.write(f'No Pantograph for Frame {idx}')
                status.empty()
                iframe.empty()
                video_dict['Pan df'] = _df_pan
                video_dict['Arc df'] = _df_arc
            else:
                st.warning('Process Video First, no JSON data available.')
    col1, col2 = st.columns([1, 9])
    with col1:
        st.subheader('3')
        st.caption('Validate Arcing Images')
    with col2:
        # st.write('Manually add events')
        # _df_tmp = video_dict['Arc df'].iloc[:0].copy()
        # _df_tmp = st.data_editor(_df_tmp, num_rows='dynamic', hide_index=True)
        # if st.button('Update Arc Table'):
        #     st.write(_df_tmp)
        if 'Arc df' in video_dict:
            _df_arc = video_dict['Arc df']
            for idx in _df_arc.index:
                try:
                    dictimage = _df_arc.at[idx, 'image']
                    image = pickle.loads(dictimage)
                    col1, col2, col3 = st.columns([1, 9, 1])
                    col1.write(_df_arc.at[idx, 'Frame'])
                    col2.image(image)
                    col3.toggle('Item of Interest', value=_df_arc.at[idx, 'arc'], key=f'del_{idx}', on_change=flip_arc_frame, args=(video_dict, idx))
                except Exception:
                    pass
            st.write('Events of Interest')
            st.dataframe(video_dict['Arc df'][video_dict['Arc df']['arc']], hide_index=True)
    if st.button('Prepare Save Data'):
        settingsout = video_dict.copy()
        if 'Pan df' in settingsout:
            settingsout.pop('Pan xywh')
            settingsout['Pan df'] = settingsout['Pan df'].to_json()
            settingsout['Arc df']['image'] = 0
            settingsout['Arc df'] = settingsout['Arc df'].to_json()
        st.write(settingsout)
        settingsout = json.dumps(settingsout)
        base_name, ext = os.path.splitext(video_dict['Video Name'])
        st.download_button(
            label='Save Video Settings',
            data=settingsout,
            file_name=f'Settings_{base_name}.json',
            )
    if st.button('GoTo Next Page to Create Final Video'):
        st.session_state.video_dict = video_dict
        st.switch_page('pages\a_st_Video_Workflow_Final.py')
    if st.button('Go Back to Settings'):
        st.switch_page('pages\3_Video_Processing.py')


if __name__ == '__main__':
    st.set_page_config(
        page_title='Live Wire Testing Toolbox_CNN',
        page_icon='🚊',
        layout='wide'
    )
    if 'video_dict' not in st.session_state:
        st.warning('Couldnt Complete')
        st.switch_page('st_Video_Workflow.py')
    else:
        val, *vals = (st.session_state.video_dict.pop(x, None) for x in ('PanView', 'Pan YOLO Results', 'Pax xywh', 'Arc YOLO Results'))
        run(st.session_state.video_dict)
