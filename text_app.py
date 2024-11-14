import os
import numpy as np
import pandas as pd
import streamlit as st
import csv

st.set_page_config(layout="wide", page_title="주차장 검색 시스템")
st.title("주차장 검색 시스템")
# 차량 정보를 검색하여 자리번호를 반환하는 함수
def find_parking_spot_by_car_number(car_number, log_file_path='log.csv'):
    try:
        with open(log_file_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['차량번호'] == car_number:
                    return row['자리번호']
        return "차량 번호를 찾을 수 없습니다."
    except FileNotFoundError:
        return "로그 파일을 찾을 수 없습니다."
    except Exception as e:
        return f"오류 발생: {e}"
    
    

def find_parking_cost_by_car_number_and_parking_spot(car_number, space_number, log_file_path='log.csv'):
    try:
        with open(log_file_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            latest_cost = None  # 가장 최신 요금을 저장할 변수 초기화
            latest_time = None  # 가장 최신 시간을 저장할 변수 초기화
            
            for row in reader:
                # 차량 번호와 자리 번호가 일치하는 경우
                if row['차량번호'] == car_number and row['자리번호'] == space_number:
                    # 요금 컬럼이 존재하고 비어 있지 않은지 확인
                    if '요금' in row and row['요금']:
                        try:
                            current_time = row['시간']
                            current_cost = float(row['요금'])
                            
                            # 가장 최근의 시간과 요금을 업데이트
                            if latest_time is None or current_time > latest_time:
                                latest_time = current_time
                                latest_cost = current_cost
                        except ValueError:
                            continue  # 요금이 변환할 수 없는 경우 무시
            
            if latest_cost is not None:  # 요금이 있는 경우
                return f"{int(latest_cost)} 원 (최신 시간: {latest_time})"
            else:
                return "해당 차량 번호와 주차칸에 대한 요금 정보가 없습니다."
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