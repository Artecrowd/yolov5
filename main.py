import torch
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# YOLOv5 모델 로드
model = torch.hub.load('/home/jetson/yolov5-python3.6.9-jetson', 'custom', path='4th.pt', source='local', force_reload=True)

# 비디오 파일 경로
video_path = '/home/jetson/yolov5-python3.6.9-jetson/testVideo_AI4th.mp4'

# 비디오 파일 열기
cap = cv2.VideoCapture(video_path)

# def process_frame(frame, polygons, current_objects):
#     results = model(frame)

#     # 다각형 좌표 시각화
#     for polygon_name, polygon_coordinates in polygons.items():
#         cv2.polylines(frame, [np.array(polygon_coordinates)], True, (0, 255, 0), 2)

#     for polygon_name, polygon_coordinates in polygons.items():
#         current_objects[polygon_name] = "0"  # 초기화

#     for obj in results.xyxy[0]:
#         x1, y1, x2, y2, conf, cls = obj
#         label = f'Class: {int(cls)}, Confidence: {conf:.2f}'
#         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
#         cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#         center_x = (x1 + x2) / 2
#         center_y = (y1 + y2) / 2
        
#         for polygon_name, polygon_coordinates in polygons.items():
#             polygon_coordinates = np.array(polygon_coordinates)
#             is_inside = cv2.pointPolygonTest(polygon_coordinates, (center_x, center_y), False)

#             # 물체의 중심 좌표가 다각형 내에 있으면 내부에 있는 것으로 표시
#             if is_inside >= 0:
#                 current_objects[polygon_name] = "1"

#     return 

def process_frame(frame, polygons, current_objects):
    results = model(frame)

    # 다각형 좌표 시각화
    for polygon_name, polygon_coordinates in polygons.items():
        cv2.polylines(frame, [np.array(polygon_coordinates)], True, (0, 255, 0), 2)

    for polygon_name, polygon_coordinates in polygons.items():
        current_objects[polygon_name] = "0"  # 초기화

    for obj in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = obj
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # center_x와 center_y를 CPU로 이동
        center_x = torch.as_tensor(center_x).cpu().item()
        center_y = torch.as_tensor(center_y).cpu().item()

        detected_class = model.names[int(cls)] if hasattr(model, 'names') else model.module.names[int(cls)]
        # 클래스 이름을 화면에 출력

        label = f'Class: {detected_class}'
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        if detected_class == "ChairFull":
            closest_polygon = None
            min_distance = float('inf')
            closest_point = None

            for polygon_name, polygon_coordinates in polygons.items():
                polygon_coordinates = np.array(polygon_coordinates)
                for point in polygon_coordinates:
                    point_np = np.array(point)
                    distance = np.linalg.norm(np.array([center_x, center_y]) - point_np)  # 수정된 부분

                    if distance < min_distance:
                        min_distance = distance
                        closest_point = point_np
                        closest_polygon = polygon_name

            # 가장 가까운 꼭짓점이 속한 다각형의 값을 1로 설정
            if closest_polygon is not None:
                closest_polygon_coordinates = np.array(polygons[closest_polygon])
                is_inside = cv2.pointPolygonTest(closest_polygon_coordinates, tuple(closest_point), False)

                if is_inside >= 0:
                    current_objects[closest_polygon] = "1"
        else:
            for polygon_name, polygon_coordinates in polygons.items():
                polygon_coordinates = np.array(polygon_coordinates)
                is_inside = cv2.pointPolygonTest(polygon_coordinates, (center_x, center_y), False)

                # 물체의 중심 좌표가 다각형 내에 있으면 내부에 있는 것으로 표시
                if is_inside >= 0:
                    current_objects[polygon_name] = "1"
    return current_objects


polygons = {
    "left_3": [(590, 230), (610, 220), (630, 220), (620, 280)],
    "left_4": [(570, 190), (600, 190), (620, 220), (590, 230)],
    "left_5": [(530, 150), (560, 150), (570, 170), (550, 170)],
    "left_6": [(510, 120), (540, 120), (560, 140), (530, 150)],
    "left_7": [(480, 100), (510, 100), (520, 110), (500, 110)],
    "left_8": [(470, 80), (490, 80), (510, 100), (480, 100)],

    "center_right_1": [(220, 240), (300, 240), (300, 315), (220, 315)],
    "center_right_2": [(230, 200), (300, 200), (300, 240), (230, 240)],
    "center_right_3": [(240, 170), (300, 170), (300, 200), (240, 200)],
    "center_right_4": [(240, 130), (300, 130), (300, 170), (240, 170)],
    "center_right_5": [(250, 110), (300, 110), (300, 130), (250, 130)],
    "center_right_6": [(260, 90), (300, 90), (300, 110), (260, 110)],
    "center_right_7": [(270, 70), (300, 70), (300, 90), (260, 90)],

    "center_left_1": [(310, 240), (400, 240), (400, 310), (310, 310)],
    "center_left_2": [(310, 200), (390, 200), (390, 240), (310, 240)],
    "center_left_3": [(310, 170), (380, 170), (380, 200), (310, 200)],
    "center_left_4": [(310, 130), (380, 130), (380, 170), (310, 170)],
    "center_left_5": [(310, 110), (370, 110), (370, 130), (310, 130)],
    "center_left_6": [(310, 90), (360, 90), (360, 110), (310, 110)],
    "center_left_7": [(310, 70), (360, 70), (360, 90), (310, 90)],

    "right_3": [(10, 220), (40, 230), (10, 280), (0, 250)],
    "right_4": [(50, 170), (70, 180), (40, 230), (10, 220)],
    "right_5": [(80, 140), (100, 150), (70, 180), (50, 170)],
    "right_6": [(100, 110), (120, 120), (100, 150), (80, 140)],
    "right_7": [(110, 100), (130, 110), (110, 120), (100, 110)],
    "right_8": [(140, 80), (150, 90), (140, 110), (120, 100)],
    "right_9": [(160, 60), (170, 70), (150, 90), (140, 80)],
    "right_10": [(180, 40), (190, 50), (170, 70), (160, 60)],
}

current_objects = {key: "0" for key in polygons.keys()}  # 초기 객체 상태 설정

with torch.no_grad():
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 객체 정보 업데이트
        current_objects = process_frame(frame, polygons, current_objects)

        # 현재 프레임의 객체 정보를 파일에 저장
        result = ""
        for polygon_name, value in current_objects.items():
            result += f"{{'{polygon_name}', '{value}'}}, "

        result1 = "{'left_1', '0'}, {'left_2', '0'}, "
        result2 = "{'right_1', '0'}, {'right_2', '0'}, {'right_11', '0'}, {'right_12', '0'}"
        final_result = result1 + result + result2

        with open('yolo_result.txt', 'w') as file:
            file.write(final_result)

        cv2.imshow('YOLOv5 Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 다음 프레임으로 넘어가기 전 초기화
        current_objects = {key: "0" for key in polygons.keys()}
cap.release()
cv2.destroyAllWindows()
# with ThreadPoolExecutor() as executor:
#     while True:
#         ret, frame = cap.read()

#         if not ret:
#             break

#         future = executor.submit(process_frame, frame, polygons)
#         frame, objects_in_polygons = future.result()
#         #results = future.result()

#         result1 = "{'left_1', '0'}, {'left_2', '0'}, {'left_3', '0'}, "
#         result2 = "{'right_1', '0'}, {'right_2', '0'}, {'right_3', '0'}"
#         new_result = ""
#         for polygon_name, value in objects_in_polygons.items():
#             new_result += f"{{'{polygon_name}', '{value}'}}, "
    
#         with open('yolo_result.txt', 'w', encoding='utf-8') as file:
#             file.write(new_result)

#         cv2.imshow('YOLOv5 Object Detection', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()

# while True:
#     ret, frame = cap.read()



#     if not ret:
#         break
#     # YOLOv5를 통한 물체 탐지
#     results = model(frame)

#     # 감지된 객체의 경계 상자를 비디오에 출력
#     for obj in results.xyxy[0]:
#         x1, y1, x2, y2, conf, cls = obj
#         label = f'Class: {int(cls)}, Confidence: {conf:.2f}'
#         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
#         cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#    # objects_in_polygons = check_objects_in_area(results, polygons)
#     #print(objects_in_polygons)
#     new_result = ", ".join([f"{{'{key}', '{value}'}}" for key, value in objects_in_polygons.items()])

#     result1 = "{'left_1', '0'}, {'left_2', '0'}, {'left_3', '0'}, "
#     result2 = ", {'right_1', '0'}, {'right_2', '0'}, {'right_3', '0'}"
#     #new_result = result1 + new_result + result2
#     with open('yolo_result.txt', 'r') as file:
#         old_content = file.read()
#     with open('yolo_result.txt', 'w') as file:
#         file.write(new_result) # + '\n' + old_content)

#     cv2.imshow('YOLOv5 Object Detection', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()