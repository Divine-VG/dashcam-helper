
import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
from track import Tracker

# ------------------------------
# Optional: Mouse position debug
# ------------------------------
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        # print(point)  # Uncomment for debugging


# ------------------------------
# Setup
# ------------------------------
model = YOLO('yolov8s.pt')

cap = cv2.VideoCapture(r'dash.mp4')


if not cap.isOpened():
    print('Error: Could not open video file')
    exit()

#choice = input('Enter q to quit the program: ')
#if choice == 'q':
#    exit()

with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Use separate trackers per class for stable IDs
car_tracker = Tracker()
bus_tracker = Tracker()
truck_tracker = Tracker()
motorcycle_tracker = Tracker()
bicycle_tracker = Tracker()
person_tracker = Tracker()
traffic_light_tracker = Tracker()

# Process frames
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    # Process every 3rd frame to reduce load
    if frame_idx % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    # Run detection
    results = model.predict(frame)
    detections = results[0].boxes.data
    px = pd.DataFrame(detections).astype("float")

    # Collect detections by class
    cars, buses, trucks, motorcycles, bicycles, persons, traffic_lights = [], [], [], [], [], [], []
    for _, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        cls_idx = int(row[5])
        cls_name = class_list[cls_idx] if 0 <= cls_idx < len(class_list) else ''

        if 'car' in cls_name:
            cars.append([x1, y1, x2, y2])
        elif 'bus' in cls_name:
            buses.append([x1, y1, x2, y2])
        elif 'truck' in cls_name:
            trucks.append([x1, y1, x2, y2])
        elif 'motorcycle' in cls_name:
            motorcycles.append([x1, y1, x2, y2])
        elif 'bicycle' in cls_name:
            bicycles.append([x1, y1, x2, y2])
        elif 'person' in cls_name:
            persons.append([x1, y1, x2, y2])
        elif 'traffic light' in cls_name:
            traffic_lights.append([x1, y1, x2, y2])

    # Update trackers
    car_boxes = car_tracker.update(cars)
    bus_boxes = bus_tracker.update(buses)
    truck_boxes = truck_tracker.update(trucks)
    motorcycle_boxes = motorcycle_tracker.update(motorcycles)
    bicycle_boxes = bicycle_tracker.update(bicycles)
    person_boxes = person_tracker.update(persons)
    traffic_light_boxes = traffic_light_tracker.update(traffic_lights)

    # Helper to process a single detection list
    def process_boxes(boxes, cls_name):
        for bbox in boxes:
            x1, y1, x2, y2, obj_id = bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Draw bbox and ID label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cvzone.putTextRect(frame, f'{cls_name} #{obj_id}', (x1, max(0, y1 - 10)), 1, 1)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

    # Process detections by class
    process_boxes(car_boxes, 'car')
    process_boxes(bus_boxes, 'bus')
    process_boxes(truck_boxes, 'truck')
    process_boxes(motorcycle_boxes, 'motorcycle')
    process_boxes(bicycle_boxes, 'bicycle')
    process_boxes(person_boxes, 'person')
    process_boxes(traffic_light_boxes, 'traffic light')

    # Show frame
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
#cv2.waitKey(0)
cv2.destroyAllWindows()

