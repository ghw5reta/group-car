# Streamlit 불러오기(웹 애플리케이션을 쉽게 생성할 수 있는 라이브러리)
import streamlit as st

# OS 모듈을 불러와 환경 변수 설정하기
import os
# 영상 스트리밍 관련 오류를 해결하기 위한 설정
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# OpenCV 불러오기(영상 처리를 위한 라이브러리)
import cv2
# Numpy와 Pandas 불러오기(데이터 처리 및 관리용)
import numpy as np
import pandas as pd
# PIL를 통해 이미지를 다루기(이미지 그리기 및 텍스트 추가용)
from PIL import Image, ImageDraw, ImageFont
# YOLO(객체 탐지 모델) 불러오기
from ultralytics import YOLO
# PaddleOCR 불러오기(한국어 OCR 인식을 위해 사용)
from paddleocr import PaddleOCR
# 지연 시간을 위한 time 라이브러리
import time

# YOLO 모델을 불러오고, GPU 사용 여부를 확인하여 최적화하기
import torch
vehicle_model = YOLO('yolov8n.pt')  # 차량 감지용 모델 불러오기
plate_model = YOLO('yolov8_license_plate.pt')  # 번호판 감지용 모델 불러오기

# PaddleOCR 모델 초기화(번호판 텍스트 인식, 한국어 설정)
ocr = PaddleOCR(use_angle_cls=True, lang='korean') 

# Streamlit 페이지 설정하기(레이아웃과 제목을 설정)
st.set_page_config(layout="wide", page_title="주차장 관리 시스템")
# 페이지 상단에 제목 표시하기
st.title("주차장 관리 시스템")

# 화면을 좌우로 나누기(왼쪽은 비디오 스트림, 오른쪽은 통계 및 데이터 영역으로 구성됨)
left_column, right_column = st.columns([1.7, 2])  # 왼쪽이 조금 더 좁게, 오른쪽이 더 넓게 설정하기

# 왼쪽 비디오 스트림 영역을 정의하는 함수
def left():
    # 왼쪽 영역에 Streamlit 컨테이너를 생성하여 웹캠 비디오 스트림 표시하기
    with left_column:
        # 비디오 프레임을 업데이트할 빈 플레이스홀더 생성하기
        frame_window = st.empty()

        # 비디오 파일 경로 설정하기
        video_path = "20241024_111930.mp4" # 동영상 필요
        cap = cv2.VideoCapture(video_path)  # 비디오 파일을 읽어오는 VideoCapture 객체 생성

        # 비디오 파일의 FPS 가져오기
        fps = cap.get(cv2.CAP_PROP_FPS)
                
        # 캡처가 성공적으로 시작된 경우에만 루프 실행하기
        while cap.isOpened():
            ret, frame = cap.read()  # 프레임 읽기
            if not ret:  # 읽기 실패 시 루프 종료
                break

            # BGR을 RGB로 변환하여 Streamlit과 호환
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 차량 감지 모델로 차량 탐지 수행
            vehicle_results = vehicle_model(frame)

            # PIL 이미지를 생성하고, 그리기 객체를 초기화
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)

            # 차량 감지 결과 순회
            for vehicle_result in vehicle_results[0].boxes:
                # 차량 클래스인지 확인 (클래스 ID로 구분)
                if int(vehicle_result.cls) in [2, 3, 7]:  # 차량 클래스 ID 목록
                    x1, y1, x2, y2 = map(int, vehicle_result.xyxy[0])  # 좌표 추출
                    vehicle_box = (x1, y1, x2, y2)
                    
                    # 번호판 감지 모델 사용하여 번호판 추출
                    cropped_vehicle = frame[y1:y2, x1:x2]  # 차량 부분만 자름
                    plate_results = plate_model(cropped_vehicle)  # 차량 내 번호판 탐지

                    for plate_result in plate_results[0].boxes:
                        x1_p, y1_p, x2_p, y2_p = map(int, plate_result.xyxy[0])
                        license_plate_img = cropped_vehicle[y1_p:y2_p, x1_p:x2_p]  # 번호판 부분만 자름
                        
                        # OCR 수행
                        ocr_result = ocr.ocr(license_plate_img, cls=True)  # OCR 수행하여 텍스트 인식
                        plate_text = ''
                        if ocr_result:
                            for line in ocr_result:
                                for word_info in line:
                                    plate_text += word_info[1][0] + " "  # 인식된 텍스트 추출
                        
                        # OCR 결과 텍스트와 차량, 번호판 박스를 그리기
                        draw.rectangle(vehicle_box, outline='green', width=2)  # 차량 박스
                        draw.text((x1, y1 - 20), f"Plate: {plate_text.strip()}", fill='blue')  # 번호판 텍스트
                        draw.rectangle([x1 + x1_p, y1 + y1_p, x1 + x2_p, y1 + y2_p], outline='red', width=2)  # 번호판 박스
            
            # Streamlit을 통해 업데이트된 이미지 표시
            frame_window.image(img, use_column_width=True)

            # UI 업데이트 속도 조절을 위해 약간의 지연을 줌
            time.sleep(0.3)  # 약 3fps
            
        # 비디오 캡처 해제
        cap.release()

# 오른쪽 통계 및 데이터 영역을 정의하는 함수
def right():
    # 오른쪽 영역에 Streamlit 컨테이너를 생성하여 차트와 테이블 배치하기
    with right_column:
        # 오른쪽 상단 영역에 두 개의 차트를 배치할 컨테이너 생성하기
        with st.container():
            # 상단을 두 개의 열로 나누어 각 차트를 표시할 공간 마련하기
            upper_left, upper_right = st.columns(2)

            # 왼쪽 차트 공간
            with upper_left:
                st.write("샘플차트")  # 차트 제목
                # 샘플 데이터를 생성하여 바 차트로 표시하기
                chart_data = pd.DataFrame(
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    columns=['A', 'B', 'C']
                )
                st.bar_chart(chart_data)  # 바 차트 표시

            # 오른쪽 차트 공간
            with upper_right:
                st.write("샘플차트 2")  # 차트 제목
                # 샘플 데이터를 생성하여 라인 차트로 표시하기
                chart_data_2 = pd.DataFrame(
                    [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
                    columns=['X', 'Y', 'Z']
                )
                st.line_chart(chart_data_2)  # 라인 차트 표시

            # 하단 영역 - 테이블 표시
            with st.container():
                st.write("Table")  # 테이블 제목
                # CSV 파일로부터 데이터를 읽어 테이블 형태로 표시하기
                df = pd.read_csv('log.ipynb') # py로 변경 필요
                # 데이터 테이블 표시 및 높이 설정
                st.dataframe(df, height=180)

# 프로그램의 시작점
if __name__ == "__main__":
    right()  # 오른쪽 통계 및 데이터 영역 호출하기
    left()   # 왼쪽 비디오 스트림 영역 호출하기

