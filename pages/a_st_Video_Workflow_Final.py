import os
import time
import json
from datetime import timedelta, datetime

import pickle
import streamlit as st
import pandas as pd
import numpy as np
import cv2

from ultralytics.utils.plotting import Annotator, colors
from pages.st_Video_Workflow_Process import box_to_xywh
from pages.st_Video_Workflow_Process import video_get_single_frame
# from pages.st_Video_Workflow_Process import test_extract_panview
from pages.st_Video_Workflow_Process import shift_box
# from pages.st_Video_Workflow_Process import annotate_panview
# from pages.st_Video_Workflow_Process import test_annotate_panview
# from pages.st_Video_Workflow_Process import test_annotate_panview_single
# from st_Video_Workflow import shift_tensor
# from st_Video_Workflow import video_get_single_frame
# from st_Video_Workflow import test_extract_panview
# from st_Video_Workflow import test_annotate_panview

if 'accesskey' not in st.session_state:
    st.session_state.accesskey = ''
else:
    st.session_state.accesskey = st.session_state.accesskey

def annotate_panview(v_dict, pan_row, arc_rows):
    pan_box = pan_row['pan_x'], pan_row['pan_y'], pan_row['pan_w'], pan_row['pan_h']
    # pan_box = box_to_xywh([pan_row['x1'], pan_row['y1'], pan_row['x2'], pan_row['y2']])
    frame = video_get_single_frame(v_dict, pan_row.index[0])
    frame = test_extract_panview(v_dict, frame, pan_box)
    frame = test_annotate_panview_single(v_dict, frame, arc_rows, pan_box)
    # st.write(pan_row)
    # st.write(arc_rows)
    return frame


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


def test_extract_panview(v_dict, frame, xywh):
    x1 = int(xywh[0] - 0.5 * v_dict['PanView Width'])
    y1 = int(xywh[1] - 0.5 * v_dict['PanView Height'] - v_dict['PanView Vert Offset'])
    x2 = int(xywh[0] + 0.5 * v_dict['PanView Width'])
    y2 = int(y1 + v_dict['PanView Height'])
    icrop = frame[y1 : y2, x1 : x2, :: (-1)]
    # gray_image = cv2.cvtColor(
    #     cv2.cvtColor(icrop, cv2.COLOR_BGR2HSV), cv2.COLOR_HSV2BGR
    #     )
    gray_image = icrop
    # gray_image = cv2.cvtColor(icrop, cv2.COLOR_BGR2LAB)
    # gray_image = cv2.cvtColor(gray_image, cv2.COLOR_HSV2BGR)
    larger_obj = cv2.resize(
        gray_image,
        (v_dict['PanView ScaleWidth'], v_dict['PanView ScaleHeight']),
        interpolation=cv2.INTER_LINEAR
        )
    return larger_obj


def video_getshape(v_dict, idx):
    videocapture = cv2.VideoCapture(v_dict['Video Path'])  # Capture the video
    if not videocapture.isOpened():
        st.error("Could not open video file.")
    videocapture.set(cv2.CAP_PROP_POS_FRAMES, idx)
    success, frame = videocapture.read()  # Read the selected frame
    videocapture.release()  # Release the capture
    cv2.destroyAllWindows()  # Destroy window
    return frame.shape


def test_annotate_panview_single(v_dict, frame, arc_row, pan_box):
    annotator = Annotator(frame, line_width=2)
    try:
        arc_box = [arc_row['x1'].values[0], arc_row['y1'].values[0], arc_row['x2'].values[0], arc_row['y2'].values[0]]
        annotator.box_label(
            shift_box(v_dict, pan_box, arc_box),
            color=colors(0, True),
            label=arc_row['item'].values[0])
    except Exception:
        arc_box = [arc_row['x1'], arc_row['y1'], arc_row['x2'], arc_row['y2']]
        annotator.box_label(
            shift_box(v_dict, pan_box, arc_box),
            color=colors(0, True),
            label=arc_row['item'])
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
    return frame


def draw_ruler_cwHeight(img: np, x_pos: int, cw_height: np.array):
    df = pd.DataFrame(cw_height)
    df[0] = [int(x) for x in df[0]]
    df.set_index([0], inplace=True)
    df = df.reindex(np.arange(min(df.index), max(df.index) + 1, 1))
    df.interpolate(inplace=True)
    df['Height'] = df[1]*12
    df['round'] = [int(n) for n in df['Height']]
    df['print'] = [n % 6 == 0 for n in df['round']]
    for height in df[df['print']]['round'].unique():
        if height > 0:
            idx = int(0.5*(min(df[df['round'] == height].index) + max(df[df['round'] == height].index)))
            cv2.putText(
                img,
                str(f'{int(df.at[idx, "Height"])/12}\' Ref'),
                (x_pos, idx), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1
                )
            cv2.putText(
                img,
                str('----------'),
                (x_pos+100, idx), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1
                )


def panframe(v_dict, frame, pan_row, arc_rows, empty):
    _pan_df = v_dict['Pan df']
    # pan_box = box_to_xywh([pan_row['x1'], pan_row['y1'], pan_row['x2'], pan_row['y2']])
    pan_box = pan_row['pan_x_smooth'], pan_row['pan_y_smooth'], pan_row['pan_w'], pan_row['pan_h']
    # frame = video_get_single_frame(v_dict, int(pan_row['Frame']))
    if not pan_row.empty:
        frame = test_extract_panview(v_dict, frame, pan_box)
        cw_height = _pan_df[_pan_df['CW_Height'] > 0][['pan_y_smooth', 'CW_Height']].values
        for row in cw_height:
            row[0] -= pan_row['pan_y_smooth'] - 1.5*v_dict['PanView Vert Offset']*v_dict['PanView Scale']
            row[0] *= v_dict['PanView Scale']
        draw_ruler_cwHeight(frame, 20, cw_height)
        if not arc_rows.empty:
            # st.text(arc_rows['Frame'].values[0])
            if len(arc_rows) == 1:
                frame = test_annotate_panview_single(v_dict, frame, arc_rows, pan_box)
            else:
                frame = test_annotate_panview(v_dict, frame, arc_rows, pan_box)
            # st.image(frame)
    return frame


def sub_arc_frame(arc_row, item, idx, fcnt, v_dict, scale):
    frame = pickle.loads(arc_row.get('image').iloc[item])
    frame = cv2.resize(
        frame,
        (int(scale * v_dict['PanView ScaleWidth']), int(scale * v_dict['PanView ScaleHeight'])),
        interpolation=cv2.INTER_LINEAR
        )
    txtpos = draw_text(frame, f"Event in {int(arc_row.get('Frame').iloc[item])-idx} frames",
        pos=(0, 0),
        font_scale=1,
        font_thickness=1,
    )
    txtpos = draw_text(
        frame, str(f"{int(arc_row.get('Frame').iloc[item])}/{fcnt}"),
        pos=(frame.shape[1]-10, frame.shape[0]-10), align='right'
        )
    return frame


def arcframe(v_dict, _df_arc, idx, fcnt, comb_w, st_container):
    scale = 0.5
    _df_arc = _df_arc[_df_arc['arc']].copy()
    _df_arc['Frame'] = _df_arc['Frame'].astype(int)
    # _df_arc.set_index('Frame', drop=True, inplace=True)
    buffer = 50
    # r = np.arange(idx-buffer, idx+buffer+300, 1)
    # dictimage = _df_arc.loc[_df_arc.index.isin(r)]
    dictimage = _df_arc[((_df_arc['Frame'] > int(idx-buffer)) * (_df_arc['Frame'] < int(idx+buffer)))]
    # frames = dictimage.get('Frame').unique()
    # st_container.write(dictimage)
    if dictimage.empty:
        frame = np.zeros((int(scale * v_dict['PanView ScaleHeight']), int(scale * v_dict['PanView ScaleWidth']), 1), dtype='uint8')
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if int(_df_arc.get('Frame').iloc[-1]) < (idx + buffer):
            text = 'No more events.'
        else:
            dictimage = _df_arc[(_df_arc['Frame'] > int(idx-buffer))]
            next = dictimage.get('Frame').iloc[0] - idx
            text = f"Next event in {next} frames."
        pos = draw_text(frame, text,
            pos=(0, 10),
            font_scale=1,
            font_thickness=1,
        )
    else:
        # items = dictimage.get('Frame').unique
        # df_sort = _df_arc.iloc[(_df_arc['Frame']-idx).abs().argsort()[:4]]
        items = len(dictimage)
        while items > 4:
            buffer = 0.5*buffer
            dictimage = _df_arc[((_df_arc['Frame'] > int(idx-buffer)) * (_df_arc['Frame'] < int(idx+buffer)))]
            items = len(dictimage)
        # print(dictimage)
        if items == 1:
            frame = sub_arc_frame(dictimage, 0, idx, fcnt, v_dict, scale)
        elif items == 2:
            frame1 = sub_arc_frame(dictimage, 0, idx, fcnt, v_dict, scale)
            frame2 = sub_arc_frame(dictimage, 1, idx, fcnt, v_dict, scale)
            frame = np.concatenate((frame1, frame2), axis=1)
        elif items == 3:
            frame1 = sub_arc_frame(dictimage, 0, idx, fcnt, v_dict, 0.5*scale)
            frame2 = sub_arc_frame(dictimage, 1, idx, fcnt, v_dict, 0.5*scale)
            lframe = np.concatenate((frame1, frame2), axis=0)
            frame3 = sub_arc_frame(dictimage, 2, idx, fcnt, v_dict, 0.5*scale)
            frame4 = np.zeros((int(0.5*scale * v_dict['PanView ScaleHeight']), int(0.5*scale * v_dict['PanView ScaleWidth']), 1), dtype='uint8')
            frame4 = cv2.cvtColor(frame4, cv2.COLOR_GRAY2BGR)
            rframe = np.concatenate((frame3, frame4), axis=0)
            frame = np.concatenate((lframe, rframe), axis=1)
        elif items == 4:
            frame1 = sub_arc_frame(dictimage, 0, idx, fcnt, v_dict, 0.5*scale)
            frame2 = sub_arc_frame(dictimage, 1, idx, fcnt, v_dict, 0.5*scale)
            lframe = np.concatenate((frame1, frame2), axis=0)
            frame3 = sub_arc_frame(dictimage, 2, idx, fcnt, v_dict, 0.5*scale)
            frame4 = sub_arc_frame(dictimage, 3, idx, fcnt, v_dict, 0.5*scale)
            rframe = np.concatenate((frame3, frame4), axis=0)
            frame = np.concatenate((lframe, rframe), axis=1)
        elif items == 5:
            frame1 = sub_arc_frame(dictimage, 0, idx, fcnt, v_dict, 0.5*scale)
            frame2 = sub_arc_frame(dictimage, 1, idx, fcnt, v_dict, 0.5*scale)
            lframe = np.concatenate((frame1, frame2), axis=0)
            frame3 = sub_arc_frame(dictimage, 2, idx, fcnt, v_dict, 0.5*scale)
            frame4 = sub_arc_frame(dictimage, 3, idx, fcnt, v_dict, 0.5*scale)
            mframe = np.concatenate((frame3, frame4), axis=0)
            frame5 = sub_arc_frame(dictimage, 4, idx, fcnt, v_dict, 0.5*scale)
            frame6 = np.zeros((int(0.5*scale * v_dict['PanView ScaleHeight']), int(0.5*scale * v_dict['PanView ScaleWidth']), 1), dtype='uint8')
            frame6 = cv2.cvtColor(frame6, cv2.COLOR_GRAY2BGR)
            rframe = np.concatenate((frame5, frame6), axis=0)
            frame = np.concatenate((lframe, mframe, rframe), axis=1)
        elif items == 6:
            frame1 = sub_arc_frame(dictimage, 0, idx, fcnt, v_dict, 0.5*scale)
            frame2 = sub_arc_frame(dictimage, 1, idx, fcnt, v_dict, 0.5*scale)
            lframe = np.concatenate((frame1, frame2), axis=0)
            frame3 = sub_arc_frame(dictimage, 2, idx, fcnt, v_dict, 0.5*scale)
            frame4 = sub_arc_frame(dictimage, 3, idx, fcnt, v_dict, 0.5*scale)
            mframe = np.concatenate((frame3, frame4), axis=0)
            frame5 = sub_arc_frame(dictimage, 4, idx, fcnt, v_dict, 0.5*scale)
            frame6 = sub_arc_frame(dictimage, 5, idx, fcnt, v_dict, 0.5*scale)
            rframe = np.concatenate((frame5, frame6), axis=0)
            frame = np.concatenate((lframe, mframe, rframe), axis=1)
        else:
            frame = np.zeros((int(scale * v_dict['PanView ScaleHeight']), int(scale * v_dict['PanView ScaleWidth']), 1), dtype='uint8')
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            pos = draw_text(frame, f"More then 4 frames",
                pos=(0, 10),
                font_scale=1,
                font_thickness=1,
            )
    padding = abs(comb_w - frame.shape[1])
    lpad = int(0.5 * padding)
    rpad = int(padding - lpad)
    full_frame = cv2.copyMakeBorder(
                frame, top=0, bottom=0,
                left=lpad, right=rpad, borderType=cv2.BORDER_CONSTANT
                )
    text = f'Total events: {len(_df_arc)}'
    pos = draw_text(full_frame, text,
            pos=(0, 10),
            font_scale=1,
            font_thickness=1,
        )
    return full_frame


def annotate_fullframe(v_dict, fullframe, fps, new_frame_rate, idx, fcnt):
    directory, oVideo_name = os.path.split(v_dict['Video Path'])
    track, Start, End, direction, *vals = oVideo_name.split('_')
    rawvideoname = '_'.join(vals) + '.mp4'
    PanNumber = int(vals[1].strip('ch'))
    dt = datetime.strptime(vals[3], '%Y%m%d%H%M%S')
    txtpos = draw_text(fullframe, f'Track: {track}', pos=(10, 10))
    txtpos = draw_text(fullframe, f'From: {Start}, To: {End}', pos=(10, txtpos))
    direction = 'Normal' if direction == 'N' else 'Reverse'
    txtpos = draw_text(fullframe, f'Direction: {direction}', pos=(10, txtpos))
    txtpos = draw_text(fullframe, '', pos=(10, txtpos))
    txtpos = draw_text(fullframe, f'Run Date: {dt.date()}', pos=(10, txtpos))
    txtpos = draw_text(fullframe, f'Run Start: {dt.time()}', pos=(10, txtpos))
    txtpos = draw_text(fullframe, '', pos=(10, txtpos))
    txtpos = draw_text(fullframe, str('Video Frame Rate reduced:'), pos=(10, txtpos))
    draw_text(fullframe, str(f'{int(fps)} to {int(new_frame_rate)} frames/sec'), pos=(10, txtpos))
    txtpos = draw_text(
        fullframe, str(f'{idx}/{fcnt}'),
        pos=(fullframe.shape[1]-10, fullframe.shape[0]-30), align='right'
        )
    draw_text(fullframe, rawvideoname, pos=(fullframe.shape[1]-10, txtpos), align='right')
    return fullframe


def Create_Final_Video(v_dict: dict):
    _pan_df = v_dict['Pan df']
    _arc_df = v_dict['Arc df']

    if True:
        for id in _arc_df[_arc_df['arc']].index:
            pan_row = _pan_df[_pan_df.index == int(_arc_df.loc[id]['Frame'])]
            arc_row = _arc_df.loc[id]
            # st.write(arc_row)
        # j = len(_arc_df.index)
        # st.write(_arc_df)
            image = annotate_panview(v_dict, pan_row, arc_row)
            # st.image(image)
            _arc_df.at[id, 'image'] = image.dumps()
    progress_text = 'In Progress. Please wait.'
    progress_bar = st.progress(0.0, text=progress_text)
    stop_button = st.button("Stop")  # Button to stop the inference

    col1, col2 = st.columns([10, 1])
    pan_frame = col1.empty()
    vid_frame = col1.empty()
    vid_data = col1.empty()

    fps_display = st.empty()  # Placeholder for FPS display

    videocapture = cv2.VideoCapture(v_dict['Video Path'])  # Capture the video
    if not videocapture.isOpened():
        st.error('Could not open video.')
    prev_time = time.time()

    w, h, fps, fcnt = (int(videocapture.get(x)) for x in (
        cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT
        ))
    new_frame_rate = int(0.5 * fps)
    # Codec Options: *'mp4v' Not supported in browsers
    # Codec Options: *'H264', doesnt work directly in Python 3. Use hex = 0x00000021
    vid_shape = video_getshape(v_dict, v_dict['Start Frame'] - 1)
    comb_h = v_dict['PanView ScaleHeight'] + vid_shape[0] + int(0.5 * v_dict['PanView ScaleHeight'])
    comb_w = max(v_dict['PanView ScaleWidth'], vid_shape[1])
    # st.write(f'Height {comb_h} and Width {comb_w}')
    vid_writer = cv2.VideoWriter(v_dict['Final_Video_file'], 0x00000021, new_frame_rate, (comb_w, comb_h))

    idx = v_dict['Start Frame'] - 1
    while videocapture.isOpened():
        videocapture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = videocapture.read()  # Read the selected frame
        if not success:
            st.warning('Failed to read frame from video.')
            v_dict['Processed through idx'] = idx
            break
        try:
            pan_row = _pan_df.iloc[idx, :]
            arc_rows = _arc_df[_arc_df['Frame'] == str(idx)]
            arc_rows = arc_rows[arc_rows['arc']]
            pan_image = panframe(v_dict, frame, pan_row, arc_rows, vid_data)
        except Exception:
            pan_image = np.zeros((v_dict['PanView ScaleHeight'], v_dict['PanView ScaleWidth'], 1), dtype='uint8')
            pan_image = cv2.cvtColor(pan_image, cv2.COLOR_GRAY2BGR)
        arc_image = arcframe(v_dict, _arc_df, idx, fcnt, comb_w, vid_data)
        width_pan = pan_image.shape[1]
        width_full = frame.shape[1]
        padding = abs(width_full - width_pan)
        lpad = int(0.5 * padding)
        rpad = int(padding - lpad)
        if width_pan > width_full:
            full_pan = pan_image
            full_frame = cv2.copyMakeBorder(
                frame, top=0, bottom=0,
                left=lpad, right=rpad, borderType=cv2.BORDER_CONSTANT
                )
        else:
            full_pan = cv2.copyMakeBorder(
                pan_image, top=0, bottom=0,
                left=lpad, right=rpad, borderType=cv2.BORDER_CONSTANT
                )
            full_frame = frame
        full_frame = annotate_fullframe(v_dict, full_frame, fps, new_frame_rate, idx, fcnt)
        comb = np.concatenate((full_pan, arc_image, full_frame), axis=0)
        vid_frame.image(comb)
        if comb.shape != (comb_h, comb_w, 3):
            st.write(f'{idx} shape {comb.shape} != {(comb_h, comb_w, 3)}')
            # st.image(pan_image)
        vid_writer.write(comb)
        if stop_button:
            v_dict['Processed through idx'] = idx
            break
        # Calculate model FPS
        curr_time = time.time()
        p_fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_display.metric("Process Speed in FPS", f"{p_fps:.2f}")
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
    vid_writer.release()
    cv2.destroyAllWindows()  # Destroy window
    st.rerun()


def run(video_dict):

    upload_dir = video_dict['Upload Folder']
    base_name, ext = os.path.splitext(video_dict['Video Name'])
    save_folder = os.path.join(upload_dir, base_name)
    Final_Video_name = base_name + '_Final_comb' + ext
    Final_Video_file = os.path.join(save_folder, Final_Video_name)
    video_dict['Final_Video_file'] = Final_Video_file

    if 'Pan df' in video_dict:
        if type(video_dict['Pan df']) == str:
            video_dict['Pan df'] = pd.DataFrame.from_dict(json.loads(video_dict['Pan df']))
    if 'Arc df' in video_dict:
        if type(video_dict['Arc df']) == str:
            video_dict['Arc df'] = pd.DataFrame.from_dict(json.loads(video_dict['Arc df']))

    col1, col2 = st.columns([1, 9])
    with col1:
        st.subheader('1')
        st.caption('Create Final Video')
    with col2:
        # if 'Arc df' in video_dict:
        #     st.write('Events of Interest')
        #     st.dataframe(video_dict['Arc df'][video_dict['Arc df']['arc']], hide_index=True,
        #                  column_order=['Frame', 'item'])
        if st.button('Create'):
            Create_Final_Video(video_dict)
        else:
            if os.path.exists(Final_Video_file):
                st.video(Final_Video_file)
                st.caption('Processed Final Video')
                with open(Final_Video_file, 'rb') as f:
                    st.download_button('Download', f, video_dict['Video Name'])

    if st.button('GoTo Previous Page to Process the Video for Arcing'):
        st.session_state.video_dict = video_dict
        st.switch_page('pages\a_st_Video_Workflow_Process.py')
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
