import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.font_manager as fm

# 한글 폰트 설정 (윈도우: Malgun Gothic, 맥: AppleGothic)
!apt-get -y install fonts-nanum
import matplotlib as mpl
import matplotlib.font_manager as fm

font_list = fm.findSystemFonts(fontpaths=["/usr/share/fonts/truetype/nanum"])
for font_path in font_list:
    fm.fontManager.addfont(font_path)

mpl.rc('font', family='NanumGothic')
mpl.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="청소년 정신건강 회귀분석 실습", layout="wide")

st.title("📊 청소년 정신건강 데이터 분석 프로젝트")
st.subheader("연구주제: 청소년의 ADHD, 신체화, 불안이 자살경향에 미치는 영향")

# 데이터 로딩 및 전처리 파이프라인
@st.cache_data
def load_and_preprocess_data():
    # 1. 데이터 불러오기
    df = pd.read_csv('청소년_정신건강_실습용.csv')
    
    # 2. 결측치 현황 저장 (보여주기 위함)
    missing_info = df.isnull().sum()
    
    # 3. 결측치 제거
    df_clean = df.dropna().copy()
    
    # 4. 범주형 데이터 인코딩 (남자=0, 여자=1)
    df_clean['성별'] = df_clean['성별'].map({'남자': 0, '여자': 1}).astype(int)
    
    return df, df_clean, missing_info

df_raw, df_clean, missing_info = load_and_preprocess_data()

# 탭 구성 (선생님의 코드 흐름 적용)
tab1, tab2, tab3, tab4 = st.tabs(["1. 기초 통계 & 결측치", "2. EDA (히트맵)", "3. 다중공선성(VIF)", "4. 다중선형회귀분석"])

with tab1:
    st.write("### 📌 기초 통계량 (결측치 제거 전)")
    st.dataframe(df_raw.describe())
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### ❓ 결측치 확인 (`df.isnull().sum()`)")
        st.dataframe(pd.DataFrame(missing_info, columns=['결측치 개수']))
        st.info("결측치가 포함된 행을 모두 제거(dropna)하여 분석을 진행합니다.")
        
    with col2:
        st.write("### 📝 범주형 변수(성별) 분포")
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        df_raw['성별'].value_counts().plot(kind='bar', color=['skyblue', 'pink'], ax=ax1)
        plt.xticks(rotation=0)
        st.pyplot(fig1)

with tab2:
    st.write("### 🌡️ 수치형 변수 상관관계 히트맵")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    numeric_cols = df_clean.select_dtypes(include=['number']).columns
    sns.heatmap(df_clean[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
    st.pyplot(fig2)

with tab3:
    st.write("### ⚖️ 독립변수 다중공선성 (VIF) 확인")
    st.write("VIF(Variance Inflation Factor)가 10을 넘으면 변수 간 중복 설명이 크다는 뜻입니다.")
    
    X = df_clean[['ADHD', '신체화', '불안']] # 회귀분석에 쓸 핵심 변수만 추출
    vif = pd.DataFrame()
    vif["VIF 지수"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["변수명"] = X.columns
    st.dataframe(vif)

with tab4:
    st.write("### 📈 최종 다중선형회귀분석 결과")
    st.write("종속변수: `자살경향` / 독립변수: `ADHD`, `신체화`, `불안`")
    
    model = smf.ols(formula='자살경향 ~ ADHD + 신체화 + 불안', data=df_clean).fit()
    st.write(model.summary().as_html(), unsafe_allow_html=True)
    
    st.success("""
    **💡 발표 해석 가이드:**
    * **R-squared (설명력):** 이 모델이 자살경향을 몇 % 설명하는가?
    * **P>|t|:** 각 독립변수의 p-value가 0.05 미만으로 통계적으로 유의미한가?
    * **coef (회귀계수):** ADHD, 신체화, 불안 중 어떤 변수가 자살경향을 가장 가파르게 상승시키는가?
    """)