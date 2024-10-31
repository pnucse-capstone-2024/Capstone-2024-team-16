# AIccaso
부산대학교 졸업과제 프로젝트

## 1. 프로젝트 소개
생성형 모델을 활용한 AI 헤어스타일러

## 2. 팀 소개

* 한지훈: 모델 실험 및 서비스 API 개발
  - HairFastGAN 모델 실험 및 결과 분석
  - 이미지 전처리 여부에 따른 성능 비교
  - 백엔드 서버 구축 및 모델 사전 로딩, API 구현
 
* 박시형: AI 모델 비용 효율화 방안 연구
  - 기존 pth 모델과 onnx 변환 모델 성능 비교
  - QAT 및 PTQ 방법으로 모델 경량화 적용
  - 각 방안에 대해 성능 비교 분석
 
* 홍진욱: AI 모델 경량화 및 분석
  - Streamlit 개발 및 프론트엔드 구축
  - Resnet9 모델 경량화 및 분석
  - GPU/CPU 비용 비교 및 분석

## 3. 시스템 구성도

![image](https://github.com/user-attachments/assets/26d6094a-9f9c-42e2-b644-e9b355b0bab3)

## 4. 소개 및 시연 영상

[![AICasso 소개 영상](https://img.youtube.com/vi/bHhdxl2hH-g/0.jpg)](https://www.youtube.com/watch?v=bHhdxl2hH-g)    

## 5. 설치 및 사용법

1) Clone GitHub Repository (HairFastGAN)
2) Follow "README" from above github link and install the "HairFastGAN"
3) Install some packages from "requirements.txt"

```
pip install -r requirements.txt
```
4) Run Front & Backend server

```
cd ./web

streamlit run front.py
uvicorn backend:app --reload --host 0.0.0.0 --port 8000 # Port number can be changed
```
