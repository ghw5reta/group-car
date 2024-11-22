import os
import time
import numpy as np
import pandas as pd
import torch
import streamlit as st
import cv2
from ultralytics import YOLO
import plotly.graph_objects as go
from paddleocr import PaddleOCR
from datetime import datetime
from collections import defaultdict
from difflib import SequenceMatcher
import csv

# 환경 설정
if "TORCH_HOME" in os.environ:
    del os.environ["TORCH_HOME"]
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# nvidia cuda gpu 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vehicle_model = YOLO('yolov8n.pt').to(device)
plate_model = YOLO('yolov8_plate.pt').to(device)
ocr = PaddleOCR(use_angle_cls=True, lang='korean')

# 차량 감지 박스 설정 (1번 자리부터 5번 자리까지 설정)
fixed_detection_boxes = [
    (220, 470, 250, 500),  # 1번 자리
    (630, 470, 660, 500),  # 2번 자리
    (970, 470, 1000, 500),  # 3번 자리
    (1300, 470, 1330, 500),  # 4번 자리
    (1700, 470, 1730, 500)   # 5번 자리
]


def read_defined_plates(file_path):
    
    with open(file_path, 'r', encoding='utf-8') as file:
        defined_plates = [line.strip() for line in file.readlines()]
    return defined_plates

plates_file_path = 'plates.txt'
defined_plates = read_defined_plates(plates_file_path)


# OCR 텍스트와 사전 정의된 번호판 간 유사도를 비교해 가장 유사한 번호판을 반환하는 함수
def match_defined_plate(ocr_text, defined_plates):
    """
    OCR로 인식된 텍스트와 사전 정의된 번호판 목록을 비교하여
    가장 유사한 번호판 텍스트를 반환하는 함수.
    """
    highest_similarity = 0
    best_match = ocr_text  # 기본적으로 원본 OCR 결과 반환

    # 사전 정의된 번호판 목록과 비교
    for plate in defined_plates:
        # 문자열 유사도를 계산
        similarity = SequenceMatcher(None, ocr_text, plate).ratio()

        # 가장 높은 유사도를 가진 번호판 선택
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = plate

    # 유사도가 0.3 이상일 때만 교정된 번호판을 반환
    if highest_similarity >= 0.3:
        return best_match
    else:
        return ocr_text  # 유사도가 낮으면 원본 OCR 텍스트 반환
    
    
# 주차 상태와 로그 초기화
detection_count = defaultdict(int)
non_detection_count = defaultdict(int)
log_file_path = "log.csv"

# Streamlit 페이지 설정
st.set_page_config(layout="wide", page_title="주차장 관리 시스템")
st.title("주차장 관리 시스템")
left_column, right_column = st.columns([1.7, 2])

with left_column:
    log_table = st.empty()

# 그래프와 테이블 위치 설정
with right_column:
    st.write("주차 로그 및 실시간 통계")
    parking_count_data = pd.DataFrame(columns=["시간", "주차 대수"])  # 시간과 주차 대수 컬럼 생성
    parking_count_chart = st.empty()  # 차트 공간

# 포함 여부 계산 함수
def calculate_inclusion(main_box, check_box):
    xA = max(main_box[0], check_box[0])
    yA = max(main_box[1], check_box[1])
    xB = min(main_box[2], check_box[2])
    yB = min(main_box[3], check_box[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    checkBoxArea = (check_box[2] - check_box[0]) * (check_box[3] - check_box[1])
    inclusion_ratio = interArea / float(checkBoxArea)
    return inclusion_ratio

# 실시간 주차 대수 기록 함수 (Plotly 사용)
def update_parking_count_chart(occupied_spots):
    global parking_count_data  # parking_count_data를 전역 변수로 선언
    
    # 실시간 시간 데이터 생성 (현재 시간)
    current_time = datetime.now().strftime("%H:%M:%S") + '.' + str(datetime.now().microsecond)[:2]  # 마이크로초를 잘라서 소수점 둘째자리까지만 사용
    
    # 새로운 데이터 추가 (시간과 주차 대수)
    new_data = pd.DataFrame({"시간": [current_time], "주차 대수": [occupied_spots]})
    parking_count_data = pd.concat([parking_count_data, new_data], ignore_index=True).tail(10)  # 최근 10개 데이터만 유지
    
    # Plotly 차트 생성
    fig = go.Figure()

    # 라인 그래프 추가
    fig.add_trace(go.Scatter(x=parking_count_data["시간"], y=parking_count_data["주차 대수"],
                             mode='lines+markers', name="주차 대수"))

    # x축과 y축 포맷 설정
    fig.update_layout(
        title="실시간 주차 대수",
        xaxis_title="시간",
        yaxis_title="주차 대수",
        xaxis=dict(
            type="category",  # x축을 카테고리형으로 설정하여 시간 순서대로 표시
            showticklabels=True,
            tickangle=45  # 시간을 더 잘 보이도록 회전
        ),
        yaxis=dict(
            tickmode="linear", 
            tick0=0, 
            dtick=1,  # y축을 정수 간격으로 설정
            showticklabels=True
        ),
        plot_bgcolor='rgba(0,0,0,0)',  # 배경 투명
        paper_bgcolor='rgba(0,0,0,0)'  # 차트의 배경도 투명으로 설정
    )

    # Streamlit에 차트 업데이트
    parking_count_chart.plotly_chart(fig)



# 실시간 주차 로그 테이블 업데이트 함수 (최신 정보만 반영)
def update_parking_log_table():
    df = pd.read_csv(log_file_path)
    
    current_time = datetime.now()
    if not df.empty:
        for idx, _ in enumerate(fixed_detection_boxes):
            parking_spot = idx + 1
            filtered_df = df[df['자리번호'] == parking_spot]
            
            if not filtered_df.empty:
                filtered_df['시간'] = pd.to_datetime(filtered_df['시간'], errors = 'coerce')
                latest_row = filtered_df.sort_values(by = '시간', ascending = False).iloc[0]

                if latest_row['출차여부'] == 'no' and (current_time - latest_row['시간']).total_seconds() >= 10:
                    df.at[latest_row.name, '출차여부'] = 'yes'
                    
                    second_latest_yes = filtered_df[filtered_df['출차여부'] == 'yes'].sort_values(by = '시간', ascending = False).iloc[0]

                    if not second_latest_yes.empty:
                        first_no_row = filtered_df[(filtered_df['출차여부'] == 'no') & (filtered_df['시간'] > second_latest_yes['시간'])].sort_values(by='시간', ascending=True).iloc[0]
                        time_diff = (current_time - first_no_row['시간']).total_seconds()
                        charge = (time_diff // 60) * 100
                        df.at[latest_row.name, '요금'] = charge

    df.to_csv(log_file_path, index = False, encoding = 'utf-8')

    latest_logs = df[df['출차여부'] == "no"].sort_values(by='시간', ascending=False).drop_duplicates(subset=['자리번호'], keep='first')
        
    log_table.empty()  # 이전 표 삭제
    log_table.table(latest_logs[['자리번호', '차량번호', '시간', '출차여부']].reset_index(drop=True))

    

# 왼쪽 비디오 스트림 처리 함수
def left():
    with left_column:
        frame_window = st.empty()

        video_path = "1080p-30m(3).mp4"
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = 0
        parking_log = []
        
        while cap.isOpened():
            
            occupied_spots = 0  # 감지된 주차 대수 초기화
            ret, frame = cap.read()
            is_detected = defaultdict(lambda : False)
            
            if not ret:
                break
            
            if frame_count % fps == 0:
                start_time = time.time()  # 프레임 처리 시작 시간
                vehicle_results = vehicle_model(frame)
                box_status = {box: (0, 255, 255) for box in fixed_detection_boxes}
                
                for vehicle_result in vehicle_results:
                    boxes = vehicle_result.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        if cls in [2, 3, 7]:  # 차량 클래스만 감지
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            vehicle_box = (x1, y1, x2, y2)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            cropped_vehicle = frame[y1:y2, x1:x2]
                            plate_results = plate_model(cropped_vehicle)

                            for plate_result in plate_results:
                                # 번호판을 인식하여 텍스트로 변환
                                for plate_box in plate_result.boxes:
                                    x_plate1, y_plate1, x_plate2, y_plate2 = map(int, plate_box.xyxy[0])
                                    cropped_plate = cropped_vehicle[y_plate1:y_plate2, x_plate1:x_plate2]
                                    ocr_results = ocr.ocr(cropped_plate, cls=True)

                                    for line in ocr_results:
                                        if line:  # None 체크 추가
                                            for res in line:
                                                ocr_text = res[1][0]  # OCR 결과에서 텍스트 추출
                                                
                                                # 유사도 함수 호출하여 최종 텍스트 교정
                                                plate_text = match_defined_plate(ocr_text, defined_plates)

                                                if plate_text:
                                                    for fixed_box in fixed_detection_boxes:
                                                        if calculate_inclusion(vehicle_box, fixed_box) >= 0.95:
                                                            is_detected[fixed_box] = True
                                                            if detection_count[fixed_box] >= 30:
                                                                box_status[fixed_box] = (0, 0, 255)
                                                                  # 감지된 주차 대수 증가
                                                                parking_log.append({
                                                                    "자리번호": fixed_detection_boxes.index(fixed_box) + 1,
                                                                    "차량번호": plate_text.strip(),
                                                                    "시간": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                                    "출차여부": "no"
                                                                })
                                            
                                                            else:
                                                                box_status[fixed_box] = (0, 128, 255)

                if parking_log:  # parking_log에 데이터가 있을 경우
                    with open(log_file_path, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=parking_log[0].keys())
                        if f.tell() == 0:
                            writer.writeheader()
                        writer.writerows(parking_log)  # 리스트의 모든 로그를 한 번에 기록
                    parking_log.clear()  # 기록 후 리스트 초기화


                for fixed_box, color in box_status.items():
                    if is_detected[fixed_box]:
                        detection_count[fixed_box] += 1
                    if color == (0, 255, 255):
                        non_detection_count[fixed_box] += 1
                        if non_detection_count[fixed_box] >= 10:
                                detection_count[fixed_box] = 0
                                non_detection_count[fixed_box] = 0
                    if color == (0, 0, 255):
                        occupied_spots += 1
                        non_detection_count[fixed_box] = 0
                    cv2.rectangle(frame, (fixed_box[0], fixed_box[1]), (fixed_box[2], fixed_box[3]), color, 2)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_window.image(frame, use_container_width=True)

                # 실시간 주차 대수 그래프 업데이트
                update_parking_count_chart(occupied_spots)

                # 실시간 주차 로그 테이블 업데이트
                update_parking_log_table()

                elapsed_time = time.time() - start_time  # 경과 시간 계산
                remaining_time = 0.9 - elapsed_time  # 남은 시간 계산

                if remaining_time > 0:
                    time.sleep(remaining_time)  # 남은 시간만큼 대기

            frame_count += 1

            


if __name__ == "__main__":
    left()