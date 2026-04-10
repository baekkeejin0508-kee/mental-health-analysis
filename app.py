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

st.set_page_config(page_title="청소년 정신건강 회귀분석 실습", layout="wide")

st.title("📊 청소년 정신건강 데이터 분석 프로젝트")
st.subheader("연구주제: 청소년의 ADHD, 신체화, 불안이 자살경향에 미치는 영향")

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('청소년_정신건강_실습용.csv')
    missing_info = df.isnull().sum()
    df_clean = df.dropna().copy()
    df_clean['성별'] = df_clean['성별'].map({'남자': 0, '여자': 1}).astype(int)
    return df, df_clean, missing_info

df_raw, df_clean, missing_info = load_and_preprocess_data()

tab1, tab2, tab3, tab4 = st.tabs(["1. 기초 통계 & 결측치", "2. EDA (히트맵)", "3. 다중공선성(VIF)", "4. 다중선형회귀분석"])

with tab1:
    st.write("### 📌 기초 통계량 (결측치 제거 전)")
    st.dataframe(df_raw.describe())
    col1, col2 = st.columns(2)
    with col1:
        st.write("### ❓ 결측치 확인")
        st.dataframe(pd.DataFrame(missing_info, columns=['결측치 개수']))
    with col2:
        st.write("### 📝 성별 분포")
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
    X = df_clean[['ADHD', '신체화', '불안']]
    vif = pd.DataFrame()
    vif["VIF 지수"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["변수명"] = X.columns
    st.dataframe(vif)

with tab4:
    st.write("### 📈 최종 다중선형회귀분석 결과")
    model = smf.ols(formula='자살경향 ~ ADHD + 신체화 + 불안', data=df_clean).fit()
    st.write(model.summary().as_html(), unsafe_allow_html=True)
