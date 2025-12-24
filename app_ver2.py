import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import date, datetime, timedelta

# --- 1. 配置與 API 設定 ---
WEATHER_API_KEY = "27f1a1defe7b4cb1a4b124857252312" 

st.set_page_config(page_title="馬拉松完賽預測 - 國際氣象版", layout="wide")

st.title("🏃‍♂️ 國際氣象版：馬拉松完賽時間預測系統")
st.markdown("""
本系統結合 **WeatherAPI 即時預報**、**個人屬性**與 **Random Forest 模型**。
輸入前 10 公里 (Sector 1 & 2) 的表現及性別組別，系統將預測最終完賽時間。
""")

def format_time(seconds):
    if seconds < 0: return "00:00:00"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# --- 2. 氣象抓取邏輯 ---
@st.cache_data(ttl=3600)
def get_global_weather(target_date):
    try:
        url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q=Tainan&days=10&aqi=no&alerts=no"
        res = requests.get(url)
        data = res.json()
        
        target_str = target_date.strftime("%Y-%m-%d")
        hours_to_get = [5, 6, 7, 8]
        results = {"temp": {}, "wind": {}, "hum": {}}

        for day in data['forecast']['forecastday']:
            if day['date'] == target_str:
                for hour_data in day['hour']:
                    time_obj = datetime.strptime(hour_data['time'], '%Y-%m-%d %H:%M')
                    if time_obj.hour in hours_to_get:
                        h = f"{time_obj.hour:02}am"
                        results["temp"][h] = hour_data['temp_c']
                        results["wind"][h] = hour_data['wind_kph'] / 3.6
                        results["hum"][h] = hour_data['humidity']
        
        if not results["temp"]: return None, False
        return results, True
    except:
        return None, False

# --- 3. 載入資料 ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("final_data.csv")
        return df
    except:
        return None

df = load_data()

if df is not None:
    # --- 側邊欄輸入 ---
    st.sidebar.header("📋 比賽資訊輸入")
    run_date = st.sidebar.date_input("比賽日期", date.today())
    
    # 性別與年齡組別選擇
    gender_age_cols = [col for col in df.columns if 'gender_age_interaction_' in col]
    gender_age_options = [c.replace('gender_age_interaction_', '') for c in gender_age_cols]
    selected_ga = st.sidebar.selectbox("性別年齡組別", gender_age_options)

    weather_info, success = get_global_weather(run_date)
    
    final_weather_inputs = {}
    with st.sidebar.expander("☁️ 氣象數據 (自動同步預報)", expanded=True):
        if success:
            st.success("✅ 成功同步台南預報數據")
        else:
            st.warning("⚠️ 無法獲取預報，請手動確認數值")

        for h_int in [5, 6, 7, 8]:
            h_str = f"{h_int:02}am"
            st.markdown(f"**時段: {h_int}:00 AM**")
            c1, c2, c3 = st.columns(3)
            with c1:
                val_t = st.number_input(f"溫度 (°C)", value=weather_info['temp'].get(h_str, 22.0) if success else 22.0, key=f"t{h_str}")
                final_weather_inputs[f'temperature{h_str}'] = val_t
            with c2:
                val_w = st.number_input(f"風速 (m/s)", value=weather_info['wind'].get(h_str, 1.5) if success else 1.5, key=f"w{h_str}")
                final_weather_inputs[f'windSpeed{h_str}'] = val_w
            with c3:
                val_h = st.number_input(f"濕度 (%)", value=weather_info['hum'].get(h_str, 75.0) if success else 75.0, key=f"h{h_str}")
                final_weather_inputs[f'humidity{h_str}'] = val_h

    st.sidebar.subheader("🏃‍♂️ 配速表現")
    s1_total = st.sidebar.number_input("Sector 1 (5K) 總秒數", value=1500)
    s2_total = st.sidebar.number_input("Sector 2 (10K) 總秒數", value=1560)

    # --- 4. 訓練與模型效能分析 ---
    @st.cache_resource
    def train_model_and_get_metrics():
        X = df.drop(columns=['gradeWithSec', 'sector3', 'sector4', 'sector5'], errors='ignore')
        y = df['gradeWithSec']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, max_depth=30, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            "R2": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred)
        }
        return model, X.columns, metrics

    rf_model, model_features, model_metrics = train_model_and_get_metrics()

    # --- 5. 執行預測與極端值檢查 ---
    if st.sidebar.button("🚀 開始預測"):
        # 極端值提示檢查邏輯
        warnings = []
        if s1_total < 750 or s2_total < 750: # 快於 2:30/km
            warnings.append("🚨 **警告**：輸入配速接近或超越世界紀錄，預測結果可能失真。")
        if s1_total > 3600 or s2_total > 3600: # 慢於 12:00/km
            warnings.append("ℹ️ **提示**：配速較接近步行速度，模型可能無法精確捕捉長跑動能變化。")
        if abs(s2_total - s1_total) > 300: # 兩段 5K 差超過 5 分鐘
            warnings.append("⚠️ **提示**：前兩段配速波動較大（超過 5 分鐘），建議檢查輸入數值是否正確。")

        input_row = pd.DataFrame(0, index=[0], columns=model_features)
        for k, v in final_weather_inputs.items():
            if k in input_row.columns: input_row[k] = v
        
        if 'sector1' in input_row.columns: input_row['sector1'] = s1_total
        if 'sector2' in input_row.columns: input_row['sector2'] = s2_total
        
        ga_col = f'gender_age_interaction_{selected_ga}'
        if ga_col in input_row.columns: input_row[ga_col] = 1
        
        pred = rf_model.predict(input_row)[0]
        all_tree_preds = np.array([tree.predict(input_row.values) for tree in rf_model.estimators_])
        std_dev = np.std(all_tree_preds)

        st.balloons()
        
        # 顯示極端值警示
        for msg in warnings:
            st.warning(msg)

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.success("### 預測完賽時間")
            st.metric("預估時間", format_time(pred))
            st.write(f"95% 信心區間：**{format_time(pred - 1.96*std_dev)}** ~ **{format_time(pred + 1.96*std_dev)}**")
        
        with res_col2:
            st.info("### 配速數據摘要")
            st.write(f"Sector 1 (5K): {format_time(s1_total)}")
            st.write(f"Sector 2 (10K): {format_time(s2_total)}")
            st.write(f"兩段落秒差: {s2_total - s1_total} 秒")

    # --- 6. 模型效能展示 ---
    st.divider()
    st.subheader("📊 模型預測準確度評估")
    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score (模型解釋力)", f"{model_metrics['R2']:.4f}")
    m2.metric("平均絕對誤差 (MAE)", f"{model_metrics['MAE']:.2f} 秒")
    m3.metric("均方誤差 (MSE)", f"{model_metrics['MSE']:.1f}")

else:
    st.warning("請確保資料夾中存有 'final_data.csv' 檔案。")