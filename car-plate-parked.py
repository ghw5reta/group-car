import gdown
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from collections import defaultdict
from IPython.display import Image, display

# 파일 ID로 다운로드
file_id = '1FjfZuabUjs6XHnanPCcVUQIF0N0tOJ2j'
gdown.download(f'https://drive.google.com/uc?id={file_id}', 'video.mp4', quiet=False)

# YOLO 모델 로드
model1 = YOLO('yolov8n.pt')  # 차량 감지 모델
model2 = YOLO('/kaggle/input/yolov8_license_plate/pytorch/default/1/yolov8_license_plate.pt')  # 번호판 감지 모델

# OCR 모델 초기화
ocr = PaddleOCR(use_angle_cls=True, lang='korean')

# 비디오 파일 경로
video_path = '/kaggle/working/video.mp4'
cap = cv2.VideoCapture(video_path)

frame_rate = cap.get(cv2.CAP_PROP_FPS)  # 프레임 속도 가져오기
process_interval = int(frame_rate)  # 1초마다 처리

# 차량 클래스 ID 목록
vehicle_classes = [2, 3, 7]
frame_count = 0

# 차량 탐색 영역 정의 (위쪽 1/3을 제외한 나머지)
search_area = (0, 360, 1920, 1080)  # (x1, y1, x2, y2)

# 고정된 감지 위치 리스트의 좌표 (x1, y1, x2, y2)
fixed_detection_boxes = [
    (300, 600, 330, 630),
    (620, 600, 650, 630),
    (920, 600, 950, 630),
    (1220, 600, 1250, 630),
    (1620, 600, 1650, 630)
]

# 주차 감지를 위한 초기화
detection_count = defaultdict(int)

# 포함 여부 계산 함수
def calculate_inclusion(main_box, check_box):
    xA = max(main_box[0], check_box[0])
    yA = max(main_box[1], check_box[1])
    xB = min(main_box[2], check_box[2])
    yB = min(main_box[3], check_box[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    checkBoxArea = (check_box[2] - check_box[0]) * (check_box[3] - check_box[1])
    
    inclusion_ratio = interArea / float(checkBoxArea) if checkBoxArea > 0 else 0
    return inclusion_ratio

# 비디오 프레임 처리 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_count % process_interval == 0:  # 1초에 한 프레임 처리하기
        # 차량 탐지
        vehicle_results = model1(frame)
        
        # 고정된 감지 위치 처리를 위한 기본 상태 설정
        box_status = {box: (0, 255, 255) for box in fixed_detection_boxes}

        for vehicle_result in vehicle_results:
            boxes = vehicle_result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls in vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    vehicle_box = (x1, y1, x2, y2)

                    # 차량이 탐색 영역 내에 있는지 확인
                    if x1 >= search_area[0] and y1 >= search_area[1] and x2 <= search_area[2] and y2 <= search_area[3]:
                        # 차량 바운딩 박스 그리기 (초록색)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # 차량 영역 크롭
                        cropped_image = frame[y1:y2, x1:x2]

                        # 번호판 탐지
                        if cropped_image.size > 0:  # 유효한 크기 체크
                            license_plate_results = model2(cropped_image)

                            for license_plate_result in license_plate_results:
                                license_plate_boxes = license_plate_result.boxes
                                for license_plate_box in license_plate_boxes:
                                    x1_c, y1_c, x2_c, y2_c = map(int, license_plate_box.xyxy[0])
                                    
                                    # 번호판 영역 표시
                                    if (0 <= y1_c < cropped_image.shape[0]) and (0 <= x1_c < cropped_image.shape[1]):
                                        cv2.rectangle(cropped_image, (x1_c, y1_c), (x2_c, y2_c), (255, 0, 0), 2)

                                        # OCR 수행
                                        license_plate_image = cropped_image[y1_c:y2_c, x1_c:x2_c]
                                        if license_plate_image.size > 0:  # 유효한 크기 체크
                                            ocr_result = ocr.ocr(license_plate_image, cls=True)

                                            # OCR 결과 출력 및 이미지에 표시
                                            if ocr_result:
                                                text = ""
                                                best_accuracy = 0
                                                for line in ocr_result:
                                                    if line:
                                                        for word_info in line:
                                                            text += word_info[1][0] + " "
                                                            best_accuracy = max(best_accuracy, word_info[1][1])

                                                # 번호판 텍스트를 이미지에 표시
                                                cv2.putText(cropped_image, text.strip(), (x1_c, y1_c - 10),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                                                # 콘솔에 번호판 텍스트 출력
                                                print(f'License Plate: {text.strip()}, Accuracy: {best_accuracy*100:.2f}%')

                    # 고정된 박스와 비교하여 95% 이상 포함하는지를 확인하고 색상 변경
                    for fixed_box in fixed_detection_boxes:
                        inclusion = calculate_inclusion(vehicle_box, fixed_box)
                        if inclusion >= 0.95:
                            detection_count[fixed_box] += 1
                            if detection_count[fixed_box] >= 5:
                                box_status[fixed_box] = (0, 0, 255)  # 주차 완료 상태: 빨강
                                cv2.putText(frame, "Parked", (fixed_box[0], fixed_box[1] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            else:
                                box_status[fixed_box] = (0, 128, 255)  # 감지 중 상태: 주황색

        # 결정된 색상으로 고정된 박스를 그리기
        for fixed_box, color in box_status.items():
            cv2.rectangle(frame, (fixed_box[0], fixed_box[1]), (fixed_box[2], fixed_box[3]), color, 2)

        # 결과를 이미지 파일로 저장 및 표시
        output_path = f'/kaggle/working/output_frame_{frame_count}.jpg'
        cv2.imwrite(output_path, frame)
        display(Image(filename=output_path))
    
    frame_count += 1

# 자원 해제
cap.release()
