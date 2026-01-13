import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import os
from tracker import Tracker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(SCRIPT_DIR, 'of.mp4')  
MODEL_NAME = os.path.join(SCRIPT_DIR, 'yolov8s.pt') 
CONFIDENCE = 0.3     
FRAME_WIDTH = 1020
FRAME_HEIGHT = 500

LINE_ORIENTATION = 'orizontala'  #modifica aici daca vreai ca linia sa fie verticala sau orizontala
LINE_POSITION_Y = 320          
LINE_POSITION_X = 500           
OFFSET = 10                     


counts = {
    'car': set(),
    'bus': set(),
    'truck': set(),
    'motorcycle': set(),
    'bicycle': set(),
    'person': set(),
   
}



def setup_app():
    """Step 1: Initialize resources."""
    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f'Error: Could not open video file {VIDEO_PATH}')
        exit()

    coco_path = os.path.join(SCRIPT_DIR, "coco.txt")
    with open(coco_path, "r") as my_file:
        class_list = my_file.read().split("\n")

    
    trackers = {
        'car': Tracker(),
        'bus': Tracker(),
        'truck': Tracker(),
        'motorcycle': Tracker(),
        'bicycle': Tracker(),
        'person': Tracker(),
        'traffic light': Tracker()
    }
    
    return model, cap, class_list, trackers 

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        # print(point) 

def detect_objects(frame, model):
    """Step 2: Detect objects using YOLO."""
    results = model.predict(frame, conf=CONFIDENCE, verbose=False)
    detections = results[0].boxes.data
    return pd.DataFrame(detections).astype("float")

def filter_detections(df_detections, class_list):
    """Step 3: Organize detections by class."""
    classified_objects = {
        'car': [], 'bus': [], 'truck': [], 'motorcycle': [], 
        'bicycle': [], 'person': []
    }
    
    for _, row in df_detections.iterrows():
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        cls_idx = int(row[5])
        cls_name = class_list[cls_idx] if 0 <= cls_idx < len(class_list) else ''
        
        # Check if the class is one we are tracking
        for key in classified_objects:
            if key in cls_name:
                classified_objects[key].append([x1, y1, x2, y2])
                break
                
    return classified_objects

def process_trackers(frame, classified_objects, trackers):
    """Step 4 & 5: Update trackers, check line crossing, and draw."""
    
    
    if LINE_ORIENTATION == 'orizontala':
        cv2.line(frame, (0, LINE_POSITION_Y), (FRAME_WIDTH, LINE_POSITION_Y), (0, 255, 255), 5)
        cvzone.putTextRect(frame, 'Counting Line', (10, LINE_POSITION_Y - 10), 1, 1, offset=10)
    else:
        cv2.line(frame, (LINE_POSITION_X, 0), (LINE_POSITION_X, FRAME_HEIGHT), (0, 255, 255), 5)
        cvzone.putTextRect(frame, 'Counting Line', (LINE_POSITION_X - 100, 50), 1, 1, offset=10)

    for cls_name, boxes in classified_objects.items():
        tracker = trackers[cls_name]
        tracked_boxes = tracker.update(boxes)
        
        for bbox in tracked_boxes:
            x1, y1, x2, y2, obj_id = bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
           
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cvzone.putTextRect(frame, f'{cls_name} #{obj_id}', (x1, max(0, y1 - 10)), 1, 1)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)
            
        
            counted = False
            if LINE_ORIENTATION == 'orizontala':
                if (LINE_POSITION_Y - OFFSET) < cy < (LINE_POSITION_Y + OFFSET):
                    counted = True
                    
                    cv2.line(frame, (0, LINE_POSITION_Y), (FRAME_WIDTH, LINE_POSITION_Y), (0, 0, 255), 2)
            else: 
                 if (LINE_POSITION_X - OFFSET) < cx < (LINE_POSITION_X + OFFSET):
                    counted = True
                  
                    cv2.line(frame, (LINE_POSITION_X, 0), (LINE_POSITION_X, FRAME_HEIGHT), (0, 0, 255), 2)

            if counted:
               
                if obj_id not in counts[cls_name]:
                    counts[cls_name].add(obj_id)


def main():
    model, cap, class_list, trackers = setup_app()
    
    cv2.namedWindow('RGB')
    cv2.setMouseCallback('RGB', mouse_callback)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        
       if frame_idx % 2 != 0:  
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    
        detections = detect_objects(frame, model)
        
      
        classified_objects = filter_detections(detections, class_list)
        
    
        process_trackers(frame, classified_objects, trackers)

      
        y_disp = 50
        for cls, id_set in counts.items():
            if len(id_set) > 0:
                cvzone.putTextRect(frame, f'{cls.capitalize()}: {len(id_set)}', (50, y_disp), 1, 1)
                y_disp += 30

        cv2.imshow("RGB", frame)
        if cv2.waitKey(1) & 0xFF == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()
    
  
    print("\n--- Final Line Crossing Counts ---")
    for cls, id_set in counts.items():
        print(f"{cls.capitalize()}: {len(id_set)}")
    print("----------------------------------")

if __name__ == '__main__':
    main()
