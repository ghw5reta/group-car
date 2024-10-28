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
# 지연 시간을 위한 time 라이브러리
import time

# YOLO 모델을 불러오고, GPU 사용 여부를 확인하여 최적화하기
import torch
model = YOLO('yolov8n.pt')  # YOLOv8 모델의 경량 버전(nano) 모델 불러오기
if torch.cuda.is_available():
    model.cuda()  # GPU 사용 가능 시 GPU로 모델 이동시키기

# PIL에서 기본 폰트 로드하기(텍스트 추가에 사용)
font = ImageFont.load_default(40)

# Streamlit 페이지 설정하기(레이아웃과 제목을 설정)
st.set_page_config(layout="wide", page_title="주차장 관리 시스템")
# 페이지 상단에 제목 표시하기
st.title("주차장 관리 시스템")

# 화면을 좌우로 나누기(왼쪽은 비디오 스트림, 오른쪽은 통계 및 데이터 영역으로 구성됨)
left_column, right_column = st.columns([1.7, 2])  # 왼쪽이 조금 더 좁게, 오른쪽이 더 넓게 설정하기

def left():
    # 왼쪽 영역에 Streamlit 컨테이너를 생성하여 웹캠 비디오 스트림 표시하기
    with left_column:
        # 컨테이너 설정: 높이를 420px로 설정하고 비디오 플레이스홀더 준비하기
        with st.container(border=True, height=420):
            # 비디오 프레임을 업데이트할 빈 플레이스홀더 생성하기
            frame_window = st.empty()

            # 비디오 파일 경로 설정하기
            video_path = r"C:\Users\user\Desktop\group-car\juyoung1213\20241024_111930.mp4"
            cap = cv2.VideoCapture(video_path)

            # 비디오 파일의 FPS를 가져오기
            fps = cap.get(cv2.CAP_PROP_FPS)
                
            # 캡처가 성공적으로 시작된 경우에만 루프 실행하기
            while cap.isOpened():
                ret, frame = cap.read()  # 프레임 읽기
                if not ret:  # 읽기 실패 시 계속 루프
                    continue

                # OpenCV는 BGR 형식으로 이미지를 다루므로 RGB로 변환하여 이미지 호환성 맞추기
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # YOLO 모델로 영상 내 객체 예측 수행하기
                results = model(frame)
                # 이미지 처리를 위해 배열을 PIL 이미지로 변환하기
                img = Image.fromarray(frame)

                # 결과를 이미지에 표시하기 위해 그리기 도구 생성하기
                draw = ImageDraw.Draw(img)
                # YOLO 모델의 탐지 결과 순회하기
                for i in range(results[0].boxes.shape[0]):
                    # 'person' 클래스만 필터링하여 표시하기
                    if model.names[int(results[0].boxes.cls[i])] == 'car': 
                        # 객체 위치에 맞춰 녹색 사각형(박스) 그리기
                        draw.rectangle(results[0].boxes.xyxy.cpu()[i].numpy(), outline='red', width=2) 
                        # 박스 상단에 'person' 텍스트 표시하기
                        draw.text((results[0].boxes.xyxy.cpu()[i].numpy()[0], results[0].boxes.xyxy.cpu()[i].numpy()[1] - 50), 'car', fill='red', font=font)  

                # 업데이트된 이미지를 스트림에 표시하기
                frame_window.image(img, use_column_width=True)

                # UI 업데이트 속도를 조절하기 위해 짧은 대기 시간 설정하기
                time.sleep(0.3)  # 약 3fps (프레임 속도를 낮춰 시스템 부하를 줄임)
            # 루프 종료 시 캡처 장치 해제하기
            cap.release()

def right():
    # 오른쪽 영역에 Streamlit 컨테이너를 생성하여 차트와 테이블 배치하기
    with right_column:
        # 오른쪽 상단 영역에 두 개의 차트를 배치할 컨테이너 생성하기
        with st.container(border=True, height=420):
            # 상단을 두 개의 열로 나누어 각 차트를 표시할 공간 마련하기
            upper_left, upper_right = st.columns(2)

            # 왼쪽 차트 공간
            with upper_left:
                # 샘플 바 차트를 위한 컨테이너 설정하기(높이 240px)
                with st.container(border=True, height=240):
                    st.write("샘플차트")  # 차트 제목
                    # 샘플 데이터를 생성하여 바 차트로 표시하기
                    chart_data = pd.DataFrame(
                        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                        columns=['A', 'B', 'C']
                    )
                    st.bar_chart(chart_data)

            # 오른쪽 차트 공간
            with upper_right:
                # 샘플 라인 차트를 위한 컨테이너 설정하기(높이 240px)
                with st.container(border=True, height=240):
                    st.write("샘플차트 2")  # 차트 제목
                    # 샘플 데이터를 생성하여 라인 차트로 표시하기
                    chart_data_2 = pd.DataFrame(
                        [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
                        columns=['X', 'Y', 'Z']
                    )
                    st.line_chart(chart_data_2)

            # 하단 영역 - 테이블 표시
            with st.container(border=True):
                st.write("Table")  # 테이블 제목
                # 데이터 테이블 플레이스홀더 생성하기
                frame_table = st.empty()
                # CSV 파일로부터 데이터를 읽어 테이블 형태로 표시하기
                df = pd.read_csv(r'C:\Users\user\Desktop\group-car\juyoung1213\log.csv')

                # 테이블의 높이를 조정하여 UI의 일관성을 유지
                frame_table.dataframe(df, height=180)  # 테이블 높이 설정
                
# 프로그램의 시작점
if __name__ == "__main__":
    right()  # 오른쪽 통계 및 데이터 영역 호출하기
    left()   # 왼쪽 비디오 스트림 영역 호출하기

