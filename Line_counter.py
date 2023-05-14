import cv2
import tqdm
from ultralytics import YOLO
import supervision as sv
import numpy as np
import json

from typing import List, Optional


SOURCE_VIDEO_PATH = "/media/harolpc/Marlin Data Storage/Videos Data/THEA Tampa/Brandon Parkway and Pauls Dr/N/23/173619-180619.avi"

def load_config(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)     
    return config

def load_model(model_path):
    model = YOLO(model_path)
    return model

def create_box_annotator():
    return sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5,
        text_padding=1
    )

def create_labels(detections, class_names_dict) -> Optional[List[str]]:
    if not detections:
        return None

    labels = []
    for i, detection in enumerate(detections):
        tracker_id = detection[4]
        class_id = int((detection[3]))
        class_names = class_names_dict.get(class_id)

        label_format = f"#{tracker_id} {class_names}"
        labels.append(label_format)

    return labels

def create_line_zones(config):
    line_zones = []
    for l_zone in config["line_zones"]:
        start = sv.Point(*l_zone["start"])
        end = sv.Point(*l_zone["end"])
        line_zone = sv.LineZone(start=start, end=end)
        line_zones.append(line_zone)
    return line_zones

def Line_Zone_annotators():
    line_zone_annotator = sv.LineZoneAnnotator(thickness=4, color=sv.Color.from_hex(color_hex="#00ff00"),
                                                text_thickness=2, text_scale=0.6, text_offset=1.0, text_padding=3)
    return line_zone_annotator



def main():
    config = load_config("config2.json")
    model = load_model("yolov8s.pt")
    video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
    box_annotator = create_box_annotator()
    line_zones = create_line_zones(config)
    line_zone_annotator = Line_Zone_annotators()  # Instantiate Line Zone Annotator
    
    model = YOLO("/home/harolpc/workspace/yolov8_dev/best.pt")
    with sv.VideoSink(target_path='target_video.mp4', video_info=video_info) as s:
        for result in model.track(source=SOURCE_VIDEO_PATH, show=False, stream=True, agnostic_nms=True, iou=0.5,conf=0.5, verbose=False):
            frame = result.orig_img
            detections = sv.Detections.from_yolov8(result)


            class_names_dict = model.model.names
            if result.boxes.id is not None:
                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

            
            labels = create_labels(detections, class_names_dict)

            frame = box_annotator.annotate(scene=frame,detections=detections,labels=labels)

            for line_zone in line_zones:
                line_zone.trigger(detections=detections)
                line_zone_annotator.annotate(frame=frame, line_counter=line_zone)

            cv2.imshow("yolov8", frame)
            s.write_frame(frame)
            if cv2.waitKey(1) == 27:  # Esc key to stop
                break

if __name__ == "__main__":
    main()


