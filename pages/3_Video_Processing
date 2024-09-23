import os
import time
import json
from datetime import timedelta
from shutil import rmtree

import streamlit as st

import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import torch
global Testing
Testing = True


def clean_uploads_sessionstate(upload_folder, video_name):
    if Testing:
        return
    base_name, ext = os.path.splitext(video_name)
    # Keep the folder directly clean
    for file_object in os.listdir(upload_folder):
        if base_name in file_object:
            file_object_path = os.path.join(upload_folder, file_object)
            if os.path.isfile(file_object_path) or os.path.islink(file_object_path):
                os.unlink(file_object_path)
            else:
                rmtree(file_object_path)


def upload_and_show_video_details(upload_folder, video_object):
    v_dict = {}
    v_dict['Upload Folder'] = upload_folder
    v_dict['Video Name'] = video_object.name
    st.session_state.last_video_upload = v_dict['Video Name']
    st.caption('Raw Original Video: ' + v_dict['Video Name'])
    played_video = st.video(video_object)
    v_dict['Video Path'] = os.path.join(upload_folder, v_dict['Video Name'])
    with open(v_dict.get('Video Path'), mode='wb') as f:
        f.write(video_object.read())
    vidcap = cv2.VideoCapture(v_dict['Video Path'])
    assert vidcap.isOpened()
    v_dict['Frame Width'] = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_dict['Frame Height'] = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    v_dict['Frames per Second'] = int(vidcap.get(cv2.CAP_PROP_FPS))
    v_dict['Frame Count'] = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, v_dict['Frame Count']-1)
    vidcap.read()  # Read progresses a frame
    v_dict['Total Length (ms)'] = vidcap.get(cv2.CAP_PROP_POS_MSEC)
    vidcap.release()
    v_dict['idx'] = 0
    v_dict['Start Frame'] = 0
    v_dict['End Frame'] = v_dict['Frame Count']
    # v_dict['Total Length (s)'] = int(v_dict['Frame Count'] / v_dict['Frames per Second'])
    st.markdown(f"""
                - Frame Count: {v_dict['Frame Count']}  
                - Frame Rate: {v_dict['Frames per Second']}  
                - Video Size: {v_dict['Frame Height']} x {v_dict['Frame Width']}  
                - Total Length: {timedelta(milliseconds=v_dict['Total Length (ms)'])}
                """
                )
    # st.text(played_video)
    return v_dict


def update_trim_timing(v_dict, frames) -> None:
    v_dict['Start Frame'] = frames[0]
    v_dict['End Frame'] = frames[1]
    vidcap = cv2.VideoCapture(v_dict['Video Path'])
    assert vidcap.isOpened()
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, v_dict['Start Frame']-1)
    success, im0 = vidcap.read()  # Read progresses a frame
    v_dict['Start Time (ms)'] = vidcap.get(cv2.CAP_PROP_POS_MSEC)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, v_dict['End Frame']-1)
    success, im1 = vidcap.read()  # Read progresses a frame
    v_dict['End Time (ms)'] = vidcap.get(cv2.CAP_PROP_POS_MSEC)
    vidcap.release()
    col1, col2 = st.columns([1,1])
    col1.caption('Starting Frame')
    col1.image(im0)
    col2.caption('Ending Frame')
    col2.image(im1)
    v_dict['Trimmed Length (ms)'] = v_dict['End Time (ms)'] - v_dict['Start Time (ms)']
    st.markdown(f"""
                - Start Time: {timedelta(milliseconds=v_dict['Start Time (ms)'])}  
                - End Time: {timedelta(milliseconds=v_dict['End Time (ms)'])}  
                - Trimmed Length: {timedelta(milliseconds=v_dict['Trimmed Length (ms)'])}
                """
                )

#@st.cache_data
def load_model(model_name):
    with st.spinner("Model is loading..."):
        loaded_model = YOLO(model_name)  # Load the YOLO model
    return loaded_model

# @st.cache_data
def video_get_single_frame(v_dict, idx):
    videocapture = cv2.VideoCapture(v_dict['Video Path'])  # Capture the video
    if not videocapture.isOpened():
        st.error("Could not open video file.")
    videocapture.set(cv2.CAP_PROP_POS_FRAMES, idx)
    success, frame = videocapture.read()  # Read the selected frame
    videocapture.release()  # Release the capture
    cv2.destroyAllWindows()  # Destroy window
    return frame

#@st.cache_data
def object_detection_single_frame(frame, _model, conf, iou):
    results = _model.predict(frame, conf=conf, iou=iou, verbose=False)  # The object can be called itself, but .predict is more clear
    torch.cuda.empty_cache()  # Clear CUDA memory
    return results


def tensor_within_tensor(v_dict, main, test):
    x1 = int(main[0, 0] - 0.5 * v_dict['PanView Width'])
    y1 = int(main[0, 1] - 0.5 * main[0, 3] - v_dict['PanView Vert Offset'])
    x2 = int(main[0, 0] + 0.5 * v_dict['PanView Width'])
    y2 = int(y1 + v_dict['PanView Height'])
    test_x = test[0, 0]
    test_y = test[0, 1]
    if test_x > x2 or test_x < x1:
        return False
    if test_y < y1 or test_y > y2:
        return False
    return True


def shift_tensor(v_dict, main, test):
    x = int(main[0, 0] - 0.5 * v_dict['PanView Width'])
    y = int(main[0, 1] - 0.5 * main[0,3] - v_dict['PanView Vert Offset'])
    sc = v_dict['PanView Scale']
    test = [
        int((test[0][0]-x) * sc),
        int((test[0][1] - y) * sc),
        int((test[0][2] - x) * sc),
        int((test[0][3] - y) * sc)
        ]
    return test


def test_extract_panview(v_dict, frame, xywh):
    x1 = int(xywh[0, 0] - 0.5 * v_dict['PanView Width'])
    y1 = int(xywh[0, 1] - 0.5 * xywh[0,3] - v_dict['PanView Vert Offset'])
    x2 = int(xywh[0, 0] + 0.5 * v_dict['PanView Width'])
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


def test_annotate_panview(v_dict, frame, xyxys, clss, names):
    annotator = Annotator(frame, line_width=2)
    for xyxy, cls in zip(xyxys, clss):
        annotator.box_label(
            shift_tensor(v_dict, v_dict['Pan xywh'], xyxy),
            color=colors(int(list(names.values()).index(cls)), True),
            label=cls)
    return frame


def get_panview(v_dict, idx, xywh):
    fullframe = video_get_single_frame(v_dict, idx)
    panview = test_extract_panview(v_dict, fullframe, xywh)
    return panview


def arcing_panview(v_dict, _model, idx, conf, iou):
    frame = video_get_single_frame(v_dict, idx)  #, v_dict['Pan xywh'])
    # Object detection run on the full frame to match the training model
    names = _model.names
    results = object_detection_single_frame(frame, _model, conf, iou)
    xyxy = []
    clss = []
    if results is not None:
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # if the center of the detected object falls within the panview window, it is added
                    if tensor_within_tensor(v_dict, v_dict['Pan xywh'], box.xywh):
                        xyxy.append(box.xyxy.cpu().tolist())
                        clss.append(names[box.cls.cpu().tolist()[0]])
    # Crop the View
    pan_view = test_extract_panview(v_dict, frame, v_dict['Pan xywh'])
    # Annotate the view with the filtered results
    pan_view = test_annotate_panview(v_dict, pan_view, xyxy, clss, names)
    return pan_view, results


#@st.fragment
def confirm_pan_model(v_dict) -> None:
    st.caption('''
               The video is first processed to try and detect the bounding box of a pantograph.  
               The coordinates of these objects are recorded for each frame for future processing.
               ''')
    panmodel = load_model('pantograph.pt')
    # col1, col2 = st.columns([1, 1])
    v_dict['Pan conf'] = float(st.slider(
        "Confidence Threshold", min_value=0.0, max_value=1.0,
        value=v_dict.get('Pan conf', 0.80),
        step=0.01, key='panmodel_conf'))
    v_dict['Pan iou'] = float(st.slider(
        "IoU Threshold", min_value=0.0, max_value=1.0,
        value=v_dict.get('Pan iou', 0.05),
        step=0.01, key='panmodel_iou'))
    frame = video_get_single_frame(v_dict, v_dict['idx'])
    results = object_detection_single_frame(frame, panmodel, v_dict['Pan conf'], v_dict['Pan iou'])
    annotated_frame = results[0].plot()  # Add annotations on frame, line_width=2
    st.image(annotated_frame, channels="BGR")  # Present the Object Detection Image
    # v_dict['Pan conf'] = conf
    # v_dict['Pan iou'] = iou
    v_dict['Pan YOLO Results'] = results
    for result in results:
        if len(result.boxes) == 0:
            st.warning('No objects detected, try a different frame or settings.')
        elif len(result.boxes) == 1:
            v_dict['Pan xywh'] = result.boxes[0].xywh
        else:
            st.warning('Too many objects detected, try a different frame or settings.')
    v_dict['PanView'] = test_extract_panview(v_dict, frame, v_dict['Pan xywh'])
    st.markdown(f"""Detected Object  
                - Center (x, y): ({v_dict['Pan xywh'][0, 0]}, {v_dict['Pan xywh'][0, 1]})  
                - Width: {v_dict['Pan xywh'][0, 2]}  
                - Height: {v_dict['Pan xywh'][0, 3]} 
                """
                )
    # if st.button('rerun', key='rerun_pan_model'):
    #     st.rerun(scope="fragment")

#@st.fragment
def confirm_arc_model(v_dict) -> None:
    st.caption('''
               The Pan View video is then processed to try and detect the bounding box of all light sources.  
               The coordinates of these objects are recorded for each frame for future processing.
               ''')
    arcmodel = load_model('arcing.pt')
    v_dict['Arc conf'] = float(st.slider(
        "Confidence Threshold", min_value=0.0, max_value=1.0,
        value=v_dict.get('Arc conf', 0.05),
        step=0.01, key='arcmodel_conf'))
    v_dict['Arc iou'] = float(st.slider(
        "IoU Threshold", min_value=0.0, max_value=1.0,
        value=v_dict.get('Arc iou', 0.20),
        step=0.01, key='arcmodel_iou'))
    # Single Full Frame
    frame = video_get_single_frame(v_dict, v_dict['idx'])
    results = object_detection_single_frame(frame, arcmodel, v_dict['Arc conf'], v_dict['Arc iou'])
    annotated_frame = results[0].plot()  # Add annotations on frame, line_width=2
    st.image(annotated_frame, channels="BGR", caption='Full Frame')  # Present the Object Detection Image
    v_dict['Arc YOLO Results'] = results
    # Pan Views
    st.subheader('Sequential Pan View Frames')
    frame_padding = 2
    for i in range(v_dict['idx'] - frame_padding, v_dict['idx'] + frame_padding + 1, 1):  # -frame_padding
        cap_txt = f'Specified Frame: {i}' if i == v_dict['idx'] else f'Frame: {i}'
        pan_view, results = arcing_panview(v_dict, arcmodel, i, v_dict['Arc conf'], v_dict['Arc iou'])
        # st.image(results[0].plot(),  channels="BGR", caption='Arc detection on the full image')
        st.image(pan_view, channels="BGR", caption=cap_txt)  # Present the Object Detection Image
    # if st.button('rerun', key='rerun_arc_model'):
    #     st.rerun(scope="fragment")

#@st.fragment
def confirm_pan_size(v_dict) -> None:
    st.caption('''
               The Pantograph View is built from the "smoothed" data collected during the object detection phase.  
               The top-middle point of the bounding box is used as the reference point.  
               Reduce the Pan View to focus the object detectio model to the areas of most interest.
               ''')
    col1, col2, col3 = st.columns([1,1,1])
    v_dict['PanView Width'] = col1.number_input('Pan View Width', value=v_dict.get('PanView Width', 720))
    v_dict['PanView Height'] = col2.number_input('Pan View Height', value=v_dict.get('PanView Height', 100))
    v_dict['PanView Vert Offset'] = col3.number_input('Pan View Vertical Offset', value=v_dict.get('PanView Vert Offset', 25))
    v_dict['PanView Scale'] = st.number_input('Pan View Scale', value=v_dict.get('PanView Scale', 2.0))
    v_dict['PanView ScaleWidth'] = int(v_dict['PanView Scale'] * v_dict['PanView Width'])
    v_dict['PanView ScaleHeight'] = int(v_dict['PanView Scale'] * v_dict['PanView Height'])


def YOLOprocess(v_dict) -> None:
    with st.spinner("Model is downloading..."):
        panmodel = YOLO("pantograph.pt")
        # panmodel = YOLO("yolov8n.pt")  # Load the YOLO model
        class_names = panmodel.names  # Convert dictionary to list of class names
    st.success("Pantograph Model loaded successfully!")

    conf = float(st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01))
    iou = float(st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.01))

    col1, col2 = st.columns(2)
    org_frame = col1.empty()
    ann_frame = col2.empty()

    fps_display = st.empty()  # Placeholder for FPS display

    if st.button("Start"):
        videocapture = cv2.VideoCapture(v_dict['Video Path'])  # Capture the video

        if not videocapture.isOpened():
            st.error("Could not open video file.")

        stop_button = st.button("Stop")  # Button to stop the inference

        idx = v_dict['Start Frame'] - 1
        videocapture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        while videocapture.isOpened():
            success, frame = videocapture.read()
            if not success:
                st.warning("Failed to read frame from video.")
                break

            prev_time = time.time()

            # Store model predictions
            enable_trk = False
            if enable_trk == "Yes":
                results = panmodel.track(frame, conf=conf, iou=iou, persist=True)
            else:
                results = panmodel(frame, conf=conf, iou=iou)
            annotated_frame = results[0].plot()  # Add annotations on frame

            # Calculate model FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # display frame
            org_frame.image(frame, channels="BGR")
            ann_frame.image(annotated_frame, channels="BGR")

            if stop_button:
                videocapture.release()  # Release the capture
                torch.cuda.empty_cache()  # Clear CUDA memory
                st.stop()  # Stop streamlit app

            # Display FPS in sidebar
            fps_display.metric("FPS", f"{fps:.2f}")
            idx += 1
            if idx > v_dict['End Frame']:
                break

        # Release the capture
        videocapture.release()

    # Clear CUDA memory
    torch.cuda.empty_cache()

    # Destroy window
    cv2.destroyAllWindows()


def Step4_1(v_dict):
    panmodel = load_model('pantograph.pt')
    button = st.empty()
    col1, col2 = st.columns(2)
    org_frame = col1.empty()
    # ann_frame = col2.empty()

    fps_display = st.empty()  # Placeholder for FPS display

    if button.button('Step 1. Locate the Pantograph on all Frames'):
        button.empty()
        with st.spinner("Video is processing..."):
            videocapture = cv2.VideoCapture(v_dict['Video Path'])  # Capture the video
            if not videocapture.isOpened():
                st.error('Could not open video.')
            prev_time = time.time()
            stop_button = st.button("Stop")  # Button to stop the inference

            idx = v_dict['Start Frame'] - 1
            while videocapture.isOpened():
                videocapture.set(cv2.CAP_PROP_POS_FRAMES, idx)
                success, frame = videocapture.read()  # Read the selected frame
                if not success:
                    st.warning('Failed to read frame from video.')
                    break
                
                results = panmodel.predict(frame, conf=v_dict['Pan conf'], iou=v_dict['Pan iou'], verbose=False)  # The object can be called itself, but .predict is more clear
                org_frame.image(results[0].plot(), channels="BGR")
                # ann_frame.image(annotated_frame, channels="BGR")

                if stop_button:
                    break
                # Calculate model FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time

                fps_display.metric("FPS", f"{fps:.2f}")
                idx += 1
                if idx > v_dict['End Frame']:
                    break
            videocapture.release()  # Release the capture
            torch.cuda.empty_cache()  # Clear CUDA memory
            cv2.destroyAllWindows()  # Destroy window


#@st.fragment
def Step1():
    upload_folder = 'uploads'
    video_object = st.file_uploader('Video to Process', ["mp4", "mov", "avi", "mkv"], accept_multiple_files=False)
    if video_object is None:
        if 'last_video_upload' in st.session_state:
            if len(st.session_state.last_video_upload) > 0:
                clean_uploads_sessionstate(upload_folder, st.session_state.last_video_upload)
        st.stop()
    video_dict = upload_and_show_video_details(upload_folder, video_object)
    # if st.radio('Video Trim:', ['Leave Full Length', 'Apply Trim']) == 'Apply Trim':
    if 'video_trim' not in st.session_state:
        st.session_state.video_trim = (0, video_dict.get('Frame Count'))
    return video_dict

#@st.fragment
def Step2(video_dict):
    maxframes = video_dict.get('Frame Count')
    frame_range = st.slider(
        'Apply a Trim to the Video (frames)',
        min_value=0,
        max_value=maxframes,
        value=[video_dict.get('Start Frame', 0), video_dict.get('End Frame', maxframes)],
        key='video_trim'
        )
    update_trim_timing(video_dict, frame_range)

#@st.fragment
def Step3a(video_dict):
    st.caption('''
                   Select a single frame to test the object detection model settings.
                   This will be used to verify proper detection of Pantograph and Arcing.
                   Try to select a frame where both exist.
                   ''')
    # video_dict['idx']=670
    video_dict['idx'] = st.number_input(
        'Select a Frame',
        value=video_dict.get('idx', video_dict['Start Frame']),
        min_value=video_dict['Start Frame'],
        max_value=video_dict['End Frame'])
    st.image(video_get_single_frame(video_dict, video_dict['idx']))

#@st.fragment
def Step3b(video_dict):
    st.markdown('---')
    st.subheader('a. Pan View Setup')
    confirm_pan_size(video_dict)
    st.markdown('---')
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader('b. Pan Model Thresholds')
        confirm_pan_model(video_dict)
    with col2:
        st.subheader('c. Arcing Model Thresholds')
        confirm_arc_model(video_dict)


def run():
    # columns = [0.5, 9.5]
    # col1, col2 = st.columns(columns)
    # col1.header('1')
    # col2.subheader('Upload Video File')
    if 'video_dict' not in st.session_state:
        with st.expander('Upload Files', expanded=True):
            video_dict = Step1()
            settings_object = st.file_uploader('Upload Previous Settings', ['json'], accept_multiple_files=False)
            if settings_object is not None:
                settingsfile = json.loads(settings_object.read())
                if settingsfile['Video Name'] == video_dict['Video Name']:
                    video_dict = settingsfile.copy()
                    st.session_state.video_dict = video_dict
                    st.switch_page('pages\st_Video_Workflow_Process.py')
                    # st.success('Settings Loaded Succesfully')
                else:
                    st.warning('Settings do not match the uploaded video')
                # st.write(settingsfile)
    else:
        video_dict = st.session_state.video_dict
        if st.button('Start Over'):
            clean_uploads_sessionstate(video_dict['Upload Folder'], video_dict['Video Name'])
            st.session_state.pop('video_dict', None)
            st.rerun()

    # col1, col2 = st.columns(columns)
    # col1.header('2')
    # col2.subheader('Choose the portion of the video to Process')
    with st.expander('2. Choose the portion of the video to Process', expanded=False):
        Step2(video_dict)
    # col1, col2 = st.columns(columns)
    # col1.header('3')
    # col2.subheader('Choose the Test Frame for Determining the Model Thresholds')
    with st.expander('3. Set the Model Thresholds', expanded=False):
        Step3a(video_dict)
        if not st.toggle('Continue to Object Detection on Frame'):
            st.stop()
        Step3b(video_dict)
        # if not st.toggle('Continue to Process Video'):
        #     st.stop()
    # col1, col2 = st.columns(columns)
    # col1.header('4')
    # col2.subheader('Choose what Items to Display')
    # with st.expander('4. Process the Video', expanded=False):
        # Step4(video_dict)
    settingsout = video_dict.copy()
    settingsout.pop('Pan YOLO Results')
    settingsout.pop('Arc YOLO Results')
    settingsout.pop('PanView')
    settingsout['Pan xywh'] = settingsout['Pan xywh'].tolist()
    if 'Pan df' in settingsout:
        settingsout['Pan df'] = settingsout['Pan df'].to_json()
        settingsout['Arc df']['image'] = 0
        settingsout['Arc df'] = settingsout['Arc df'].to_json()
    st.write(settingsout)
    settingsout = json.dumps(settingsout)
    base_name, ext = os.path.splitext(video_dict['Video Name'])
    # st.write(settingsout)
    st.download_button(
        label='Save Video Settings',
        data=settingsout,
        file_name=f'Settings_{base_name}.json',
        )
    if st.button('GoTo Next Page to Process the Video using these settings'):
        st.session_state.video_dict = video_dict
        st.switch_page('pages\a_st_Video_Workflow_Process.py')
    # st.page_link('pages\a_st_Video_Workflow_Process.py', label='4. Process the Video')


if __name__ == '__main__':
    st.set_page_config(
        page_title='Live Wire Testing Toolbox_CNN',
        page_icon='🚊',
        layout='wide'
    )
    run()
