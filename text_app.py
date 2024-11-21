import os
import numpy as np
import pandas as pd
import streamlit as st
import csv
from datetime import datetime

st.set_page_config(layout="wide", page_title="주차장 검색 시스템")
st.title("주차장 검색 시스템")
# 차량 정보를 검색하여 자리번호를 반환하는 함수
def find_parking_spot_by_car_number(car_number, log_file_path='log.csv'):
    try:
        with open(log_file_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            for row in reversed(rows):
                if row['차량번호'] == car_number:
                    return row['자리번호']
        return "차량 번호를 찾을 수 없습니다."
    except FileNotFoundError:
        return "로그 파일을 찾을 수 없습니다."
    except Exception as e:
        return f"오류 발생: {e}"
    
    

def find_parking_cost_by_car_number_and_parking_spot(car_number, space_number, log_file_path='log.csv'):
    try:
        # log.csv 파일 읽기
        df = pd.read_csv(log_file_path, encoding='utf-8')
        
        is_or_not = df[(df['차량번호'] == car_number) & (df['자리번호'] == int(space_number))].copy()
        
        if not is_or_not.empty:
        # 차량번호와 자리번호가 일치하는 행들 필터링
            filtered_df = df[df['자리번호'] == int(space_number)].copy()
            filtered_df['시간'] = pd.to_datetime(filtered_df['시간'], errors='coerce')  # 시간 컬럼을 datetime 형식으로 변환
            latest_yes_row = filtered_df[filtered_df['출차여부'] == 'yes'].sort_values(by='시간', ascending=False).iloc[0]
            first_no_row = filtered_df[(filtered_df['출차여부'] == 'no') & (filtered_df['시간'] > latest_yes_row['시간'])].sort_values(by='시간', ascending=True).iloc[0]
            time_diff = (datetime.now() - first_no_row['시간']).total_seconds() # 분 단위 계산
            charge = (time_diff // 60) * 100
            
            return f"{charge} 원 (차량번호: {car_number}, 자리번호: {space_number})"
    
        else:
            return "해당 차량 번호와 자리 번호에 대한 기록이 없습니다."

    except FileNotFoundError:
        return "로그 파일을 찾을 수 없습니다."
    except Exception as e:
        return f"오류 발생: {e}"
    
with st.form(key="number"):
    car_number = st.text_input(
        label="차 번호",
        placeholder="12가3456(공백 없이)" # 기본 값 설정 (필요 시 수정 가능)
    )
        
    # 폼 제출 버튼 생성
    submit_button = st.form_submit_button(label="자리번호 조회")
        
# 제출 버튼이 눌렸을 때 실행
if submit_button:
    result = find_parking_spot_by_car_number(car_number)
    st.write(f"자리 번호: {result}")
    
with st.form(key="cost"):
    col1, col2 = st.columns(2)
    with col1:
        car_number = st.text_input(
            label="차 번호",
            placeholder="12가3456(공백 없이)" # 기본 값 설정 (필요 시 수정 가능)
    )
    with col2:
        space_number = st.text_input(
            label="자리 번호",
            placeholder="(단일 숫자)" # 기본 값 설정 (필요 시 수정 가능)
        )
        
    # 폼 제출 버튼 생성
    submit_button = st.form_submit_button(label="주차요금 조회")
        
# 제출 버튼이 눌렸을 때 실행
if submit_button:
    result = find_parking_cost_by_car_number_and_parking_spot(car_number, space_number)
    st.write(f"요금 : {result}")