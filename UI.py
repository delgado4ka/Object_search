import streamlit as sl
import numpy as np
import os
import cv2
from scipy import spatial
import matplotlib.pyplot as plt
from ultralytics import YOLO
import tempfile
from numpy import linalg as LA
import datetime
import base64

from forensicsUI import obj_detection
from forensicsUI import feature_extract
from forensicsUI import feature_comparison
from forensicsUI.database import DatabaseManager
from forensicsUI.objs import Objects
from forensicsUI.helpers import *
from forensicsUI.visualization import *
from forensicsUI.features import Features
from forensicsUI.output import Output

# Initialize the YOLO model
od_model = YOLO('yolov8l.pt')

# Function that saves the cropped objects
def crop(img, output_dir, start_index):
    result = od_model(img)
    original_image = result[0].orig_img

    for i, box in enumerate(result[0].boxes.xyxy):
        x1, y1, x2, y2 = box.tolist()
        cropped_object = original_image[int(y1):int(y2), int(x1):int(x2)]
        output_path = os.path.join(output_dir, f"object_{start_index}.jpg")
        cv2.imwrite(output_path, cropped_object)
        start_index += 1

# Temporarily saves the input file and returns its path
def save_uploaded_file(uploaded_file):
    _, temp_file = tempfile.mkstemp()
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getvalue())
    return temp_file

# Save the frames of the video
def save_video_frames(input_video, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_capture = cv2.VideoCapture(input_video)
    success, frame = video_capture.read()
    count = 1

    while success:
        if count % 10 == 0:
            frame_filename = os.path.join(output_dir, f"frame_{count}.jpg")
            cv2.imwrite(frame_filename, frame)
        success, frame = video_capture.read()
        count += 1

    video_capture.release()
    sl.write(f"Frames saved to {output_dir}")

def save_image(image_file, dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

    image_path = os.path.join(dir, image_file.name)
    with open(image_path, "wb") as f:
        f.write(image_file.getbuffer())

    return image_path

logo_path = "AI_Ninajas_logo.ico"

sl.set_page_config(
    page_title="Object Search for Forensics",
    page_icon=logo_path
)

def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()
    
logo_base64 = get_image_base64(logo_path)

sl.markdown(
    f"""
    <style>
    .center {{
      display: flex;
      justify-content: center;
      align-items: center;
    }}
    .center img {{
      max-width: 20%;  /* Adjust this value to change the image size */
      max-height: 20%; /* Adjust this value to change the image size */
    }}
    </style>
    <div class="center">
      <img src="data:image/png;base64,{logo_base64}" alt="Centered Image">
    </div>
    """,
    unsafe_allow_html=True
)

sl.markdown("<h1 style='text-align: center; color: #56aeff;'> Object Search for Forensics", unsafe_allow_html=True)
sl.markdown("<p style='text-align: center; color: #56aeff;'> by AI Ninjas", unsafe_allow_html=True)

col1, col2 = sl.columns(2)

with col1:
    image_file = sl.file_uploader("Upload Target Object Image", type=["PNG", "JPG", "JPEG"])

    if image_file is not None:
        image_path = save_image(image_file, 'target_image')
        crop(image_path, 'target_crop', 0)

        # Directory to save the selected target object
        selected_target_dir = "target_object"
        if not os.path.exists(selected_target_dir):
            os.makedirs(selected_target_dir)

        count = 1
        for i in os.listdir('target_crop'):
            col_target1, col_target2 = sl.columns(2)
            with col_target1:
                if sl.button(f"Select Object #{count}", key=f"select_{count}"):
                    selected_target_path = os.path.join(selected_target_dir, f"selected_object.jpg")
                    target_object_path = os.path.join('target_crop', i)
                    cv2.imwrite(selected_target_path, cv2.imread(target_object_path))
                    sl.session_state['target_object_path'] = selected_target_path
            with col_target2:
                sl.image(os.path.join('target_crop', i), caption='Object #' + str(count), width=100)
            count += 1


    video_file = sl.file_uploader("Upload Search Video", type=["mp4", "avi", "MOV"])

    if video_file is not None:
        temp_video_path = save_uploaded_file(video_file)
        # with col2:
        #     sl.video(temp_video_path)
        # Use this if you need to save video frames on the device itself
        # save_video_frames(temp_video_path, "video_framed")
        # Uncomment below code if you need to process frames and crop objects

        # start = ''
        # for i in os.listdir("video_framed"):
        #     for j in range(len(os.listdir('croped_video')[-1])):
        #         if j > 0:
        #             if os.listdir('croped_video')[-1][j-1] == '_':
        #                 while os.listdir('croped_video')[-1][j+1] != '.':
        #                     start += os.listdir('croped_video')[-1][j]
        #     crop("video_framed/" + i, "croped_video", int(start))


# You can access the selected target object using sl.session_state['target_object_path']
    if image_file is not None and video_file is not None:
        if temp_video_path is not None and sl.session_state['target_object_path'] is not None:
            extracted_objects = []
            MIN_DETECTION_CONFIDENCE = 0.7
            extraction_interval = 5
            is_performance_mode = True



            # """
            # Database looks like this
            # objects: obj_id (INT), cls (INT), conf (REAL), bbox (TEXT),  
            #             frame_ids = [frame_id], similarity_score (REAL)> obj_id 0 for target image
            # features: obj_id (INT), features (BLOB)
            # bboxes: obj_id INTEGER, frame_id INTEGER, bbox TEXT, PRIMARY KEY (obj_id, frame_id),
            # """

            # save some essential metadata
            metadata = {}


            # basic local helpers
            def toggle_visualization():
                global visualization_on
                visualization_on = not visualization_on


            def update_metadata(key, value):
                metadata[key] = value


            def iterate_detection_result(res, tracking_on):
                bbox = res[:4].to(int).tolist()
                conf = round(float(res[-2]), 2)
                cls = int(res[-1])
                if tracking_on:
                    obj_id = int(res[4])
                    return obj_id, cls, conf, bbox
                return cls, conf, bbox


            def should_extract_features(obj):
                # """determine whether to extract features for a given object."""
                if is_performance_mode:
                    return (obj.frame_ids[-1] - obj.frame_ids[0]) % extraction_interval == 0
                else:
                    return obj.obj_id not in extracted_objects


            # def get_time():
            #     cur_time = datetime.datetime.now()
            #     strf = "%-d-%-m-%H:%M"
            #     time_str = datetime.datetime.strftime(cur_time, strf)
            #     return time_str


            # this get_time() function version is for windows, the above one returns an error 
            def get_time():
                cur_time = datetime.datetime.now()
                strf = "%d-%m-%H-%M"
                time_str = datetime.datetime.strftime(cur_time, strf)
                return time_str



            def get_high_rank(obj_dict):
                sorted_dict_id = sorted(obj_dict, key=lambda x: obj_dict[x].similarity, reverse=True)
                return sorted_dict_id


            # basic local helpers end


            # """ Input logic here """
            input_video_path = temp_video_path
            input_img_path = "target_object/selected_object.jpg"

            update_metadata("input_video_path", input_video_path)
            update_metadata("input_img_path", input_img_path)
            # visualization_on = True

            # """ Input logic end """

            database_name = get_time()
            db = DatabaseManager(os.path.join("database", database_name))
            # create objects and features tables here
            db.create_objects_table()
            db.create_features_table()
            # todo create bboxes db

            # initialize detector, extractor, feature comparison, visualizer objects
            detector = obj_detection.ObjDetection()
            feature_extractor = feature_extract.FeatureExtract()
            f_comparison = feature_comparison.FeatureComparison(threshold=0.8)  # I know f_comparison is a shitty name
            visualizer = Visualization()

            # logic for processing the target image
            # todo move it elsewhere, or make it prettier idk
            im_result = detector.detect(input_img_path, isImage=True)
            # save cropped object image without background
            best_result = save_target_obj(im_result)

            # assign object class, confidence score to variables
            cls, conf, coord = iterate_detection_result(best_result, tracking_on=False)
            target_obj_id = 0
            target_frame_index = 0

            # extract target object's features
            target_arr = im_result[0].orig_img
            target_obj = crop_object(coord, target_arr)
            target_feature = feature_extractor.extract_vector(target_obj)
            squeezed_target = reshape_target(target_feature)

            update_metadata("target_obj_id", target_obj_id)
            update_metadata("target_feature_shape", squeezed_target.shape)

            # initialize objects dict with object instances inside
            objects = {target_obj_id: Objects(obj_id=target_obj_id,
                                            cls=cls,
                                            conf=conf,
                                            similarity=1)}
            objects[target_obj_id].add_bbox(coord)
            objects[target_obj_id].add_frame_index(target_frame_index)

            # initialize object features dict
            # this dict will contain *best* values
            obj_features = {target_obj_id: Features(target_obj_id)}
            obj_features[target_obj_id].set_feature(feature=squeezed_target)
            # logic for processing the target image END

            # Loop through the video frames
            cap = cv2.VideoCapture(input_video_path)

            while cap.isOpened():
                # Read a frame from the video
                success, frame = cap.read()

                if success:
                    # Run YOLOv8 tracking on the frame, with tracking on (persist=true)
                    results = detector.detect(frame, classes=[cls], tracker="bytetrack.yaml", persist=True)

                    current_frame = results[0].orig_img
                    current_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # frame index starts with 1

                    # iterate over one frame's results start
                    detection_data = results[0].boxes.data
                    for result in detection_data:
                        obj_id, cls, conf, bbox = iterate_detection_result(result, tracking_on=True)

                        if obj_id in objects.keys():
                            current_obj = objects[obj_id]
                            current_obj.add_frame_index(current_frame_index)
                            current_obj.add_bbox(bbox)
                        else:
                            current_obj = Objects(obj_id=obj_id, cls=cls, conf=conf)
                            current_obj.add_frame_index(current_frame_index)
                            current_obj.add_bbox(bbox)
                            objects[obj_id] = current_obj

                        # extract feature of the current object
                        if should_extract_features(current_obj):
                            cropped_object = crop_object(bbox, current_frame)
                            feature = feature_extractor.extract_vector(cropped_object)

                            # squeeze to remove dimension of 1

                            squeezed_feature = reshape_feature(feature)

                            similarity = f_comparison.cosine_similarity_matrix(squeezed_target, squeezed_feature)[0]
                            current_obj.similarity = similarity

                            # why did I do this????
                            if obj_id not in obj_features:
                                obj_features[obj_id] = Features(obj_id)
                                obj_features[obj_id].set_feature(feature)
                            elif similarity > objects[obj_id].similarity:
                                obj_features[obj_id].set_feature(feature)
                                objects[obj_id].similarity = similarity

                            if "features_shape" not in metadata:
                                features_shape = feature.shape
                                update_metadata("features_shape", features_shape)

                        if visualizer.is_on:
                            visualizer.visualize_object(frame=frame, obj=current_obj)

                    # iterate over one frame's results end
                    if visualizer.is_on:
                        visualizer.display_frame(frame=frame)



                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    # Break the loop if the end of the video is reached
                    break

            # Release the video capture object and close the display window
            cap.release()
            cv2.destroyAllWindows()

            # insert objects, features into database
            for obj_id in objects.keys():
                # print(obj_id)
                obj = objects[obj_id]
                feats = obj_features[obj_id].feature

                sim = round(float(obj.similarity), 2)

                db.insert_to_objects(obj_id=obj.obj_id,
                                    cls=obj.cls,
                                    conf=obj.conf,
                                    bbox=obj.bbox,
                                    frame_ids=obj.frame_ids,
                                    similarity=sim)

                db.insert_to_features(obj_id=obj_id, features=feats)

            write_metadata_to_file(metadata)
            video_output = Output()
            similar_obj_ids = get_high_rank(objects)
            top_k = 3
            if len(similar_obj_ids) > 0:
                for similar_obj_id in similar_obj_ids[:top_k + 1]:
                    if similar_obj_id == 0:
                        continue

                    bboxes, frames = objects[similar_obj_id].bbox, objects[similar_obj_id].frame_ids
                    with col2:
                        sl.video(video_output.extract_video_subset(metadata["input_video_path"],
                                                        [frames, bboxes], output_path=f"output_{similar_obj_id}.mp4"))
            else:
                with col2:
                    sl.write("No similar objects found.")
