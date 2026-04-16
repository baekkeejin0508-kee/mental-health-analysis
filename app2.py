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

st.set_page_config(page_title="청소년 정신건강 분석", layout="wide", page_icon="🏥")

@st.cache_data
def load_and_preprocess_data():
    df_raw = pd.read_csv('청소년_정신건강_실습용.csv')
    missing_info = df_raw.isnull().sum() # 결측치 정보 저장
    
    df_clean = df_raw.dropna().copy()
    df_clean['성별'] = df_clean['성별'].map({'남자': 0, '여자': 1}).astype(int)
    df_clean['자살위험군'] = (df_clean['자살경향'] >= 1.0).astype(int) # 로지스틱용 파생변수
    return df_raw, df_clean, missing_info

df_raw, df_clean, missing_info = load_and_preprocess_data()

# --- 사이드바 ---
with st.sidebar:
    st.title("🏥 보건/의료 데이터 분석")
    st.info("청소년 정신건강 실태조사 데이터를 활용한 웹 보고서입니다.")
    
    menu = st.radio(
        "📂 분석 메뉴를 선택하세요",
        ("1. 데이터 탐색 (통계 및 결측치)", "2. 다중선형회귀분석", "3. 로지스틱 회귀분석")
    )
    st.divider()
    st.write("👨‍🏫 **연구자:** 000 보건교사")

# --- 메인 화면 ---
if menu == "1. 데이터 탐색 (통계 및 결측치)":
    st.title("📊 데이터 탐색 (EDA)")
    st.write("본격적인 분석 전, 데이터의 기초 통계량과 결측치를 확인합니다.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### 📌 기초 통계량 (결측치 제거 전)")
        st.dataframe(df_raw.describe())
    with col2:
        st.write("### ❓ 결측치 확인")
        st.dataframe(pd.DataFrame(missing_info, columns=['결측치 개수']))
        st.info("💡 분석의 정확성을 위해 결측치가 포함된 행은 모두 제거 후 회귀분석을 진행합니다.")
        
    st.write("### 🌡️ 변수 간 상관관계 (히트맵)")
    fig, ax = plt.subplots(figsize=(6, 4))
    numeric_cols = ['자살경향', 'ADHD', '신체화', '불안']
    sns.heatmap(df_clean[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

elif menu == "2. 다중선형회귀분석":
    st.title("📈 다중선형회귀분석")
    st.success("**연구주제:** ADHD, 신체화, 불안 점수가 자살경향성 '점수'에 미치는 영향")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("### ⚖️ 다중공선성 (VIF) 확인")
        X = df_clean[['ADHD', '신체화', '불안']]
        vif = pd.DataFrame()
        vif["VIF 지수"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif["변수명"] = X.columns
        st.dataframe(vif)
        st.caption("💡 VIF가 10 미만이면 독립변수 간 중복 문제가 없다고 판단합니다. (로지스틱 회귀분석에도 동일하게 적용됨)")
        
    with col2:
        st.write("### 💡 회귀분석 결과표")
        model_ols = smf.ols(formula='자살경향 ~ ADHD + 신체화 + 불안', data=df_clean).fit()
        st.write(model_ols.summary().as_html(), unsafe_allow_html=True)

elif menu == "3. 로지스틱 회귀분석":
    st.title("📉 로지스틱 회귀분석")
    st.error("**연구주제:** ADHD, 신체화, 불안 요인이 자살 '위험군(Cut-off 1.0 이상)'으로 분류될 확률에 미치는 영향")
    st.info("💡 독립변수들의 다중공선성(VIF)은 다중선형회귀분석 탭의 결과와 동일하게 문제없음을 확인했습니다.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("### ⚠️ 자살 위험군 분포")
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        df_clean['자살위험군'].value_counts().plot(kind='bar', color=['#99ccff', '#ff9999'], ax=ax2)
        ax2.set_xticklabels(['일반군(0)', '위험군(1)'], rotation=0)
        st.pyplot(fig2)
        st.caption("자살경향 1.0 이상을 위험군으로 정의함")
        
    with col2:
        st.write("### 💡 로지스틱 분석 결과표")
        model_logit = smf.logit(formula='자살위험군 ~ ADHD + 신체화 + 불안', data=df_clean).fit()
        st.write(model_logit.summary().as_html(), unsafe_allow_html=True)
