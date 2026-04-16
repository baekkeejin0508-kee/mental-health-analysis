import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.font_manager as fm

# 스트림릿 클라우드용 한글 폰트 세팅
font_dirs = ['/usr/share/fonts/truetype/nanum']
font_files = fm.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    fm.fontManager.addfont(font_file)
plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# 페이지 기본 설정 (가장 넓게 쓰기)
st.set_page_config(page_title="청소년 정신건강 분석", layout="wide", page_icon="🏥")

# ----------------------------------------------------
# 1. 데이터 로딩 및 전처리 (로지스틱용 변수 추가)
# ----------------------------------------------------
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('청소년_정신건강_실습용.csv')
    df_clean = df.dropna().copy()
    df_clean['성별'] = df_clean['성별'].map({'남자': 0, '여자': 1}).astype(int)
    
    # ★ 로지스틱 회귀분석을 위한 '위험군(0 또는 1)' 종속변수 생성
    # 자살경향 평균이 1.0 이상이면 위험군(1), 미만이면 일반군(0)으로 분류
    df_clean['자살위험군'] = (df_clean['자살경향'] >= 1.0).astype(int)
    return df_clean

df_clean = load_and_preprocess_data()

# ----------------------------------------------------
# 2. 사이드바(Sidebar) 메뉴 구성 (사진과 동일한 방식)
# ----------------------------------------------------
with st.sidebar:
    st.title("🏥 보건/의료 데이터 분석")
    st.info("청소년 정신건강 실태조사 데이터를 활용한 회귀분석 웹 보고서입니다.")
    
    # 메뉴 선택 (라디오 버튼)
    menu = st.radio(
        "📂 분석 메뉴를 선택하세요",
        ("소개 및 데이터 탐색", "다중선형회귀분석", "로지스틱 회귀분석")
    )
    
    st.divider()
    st.write("👨‍🏫 **연구자:** 000 보건교사")

# ----------------------------------------------------
# 3. 메뉴별 화면 구성
# ----------------------------------------------------

# [메뉴 1] 소개 및 데이터 탐색
if menu == "소개 및 데이터 탐색":
    st.title("📊 청소년 정신건강 데이터 탐색")
    st.subheader("기본 데이터 구조 및 상관관계 확인")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### 📝 데이터 미리보기")
        st.dataframe(df_clean.head(10))
    
    with col2:
        st.write("### 🌡️ 변수 간 상관관계 (히트맵)")
        fig, ax = plt.subplots(figsize=(6, 4))
        numeric_cols = ['자살경향', 'ADHD', '신체화', '불안']
        sns.heatmap(df_clean[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

# [메뉴 2] 다중선형회귀분석
elif menu == "다중선형회귀분석":
    st.title("📈 다중선형회귀분석")
    st.success("**연구주제:** ADHD, 신체화, 불안 점수가 자살경향성 '점수'에 미치는 영향")
    
    st.write("### 💡 회귀분석 결과표")
    model_ols = smf.ols(formula='자살경향 ~ ADHD + 신체화 + 불안', data=df_clean).fit()
    st.write(model_ols.summary().as_html(), unsafe_allow_html=True)

# [메뉴 3] 로지스틱 회귀분석
elif menu == "로지스틱 회귀분석":
    st.title("📉 로지스틱 회귀분석")
    st.error("**연구주제:** ADHD, 신체화, 불안 요인이 자살 '위험군(Cut-off 1.0 이상)'으로 분류될 확률에 미치는 영향")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("### ⚠️ 위험군 분포")
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        df_clean['자살위험군'].value_counts().plot(kind='bar', color=['#99ccff', '#ff9999'], ax=ax2)
        ax2.set_xticklabels(['일반군(0)', '위험군(1)'], rotation=0)
        st.pyplot(fig2)
        st.caption("자살경향 1.0 이상을 위험군으로 정의함")
        
    with col2:
        st.write("### 💡 로지스틱 분석 결과표")
        # smf.logit을 사용하여 로지스틱 회귀분석 수행
        model_logit = smf.logit(formula='자살위험군 ~ ADHD + 신체화 + 불안', data=df_clean).fit()
        st.write(model_logit.summary().as_html(), unsafe_allow_html=True)
