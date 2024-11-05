import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from datetime import datetime, timedelta
import plotly.express as px

# Set page config
st.set_page_config(page_title="SmartGrid AI", page_icon="⚡", layout="wide")

# Custom CSS to improve the look
st.markdown("""
    <style>
    /* Main app styling */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #262730;
        padding: 2rem 1rem;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #ffffff;
    }
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] .stDateInput label {
        color: #ffffff !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #ffffff !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        color: #b0b0b0 !important;
    }
    
    /* Cards for metrics */
    div[data-testid="column"] > div[data-testid="stMetricValue"] {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Text elements */
    .stMarkdown {
        color: #ffffff;
    }
    p {
        color: #ffffff;
    }
    
    /* Plotly charts background */
    .js-plotly-plot {
        background-color: #262730 !important;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
    }
    
    /* Divider */
    hr {
        border-color: #4a4a4a;
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def generate_mock_data(days=30):
    date_rng = pd.date_range(start='2024-01-01', end='2024-01-30', freq='H')
    df = pd.DataFrame(date_rng, columns=['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['temperature'] = np.random.normal(20, 5, size=len(df))
    df['energy_consumption'] = 10 + 2*df['hour'] + 5*np.sin(df['hour']*np.pi/12) + 3*df['temperature'] + np.random.normal(0, 2, size=len(df))
    return df

@st.cache_resource
def train_model(df):
    X = df[['hour', 'day_of_week', 'temperature']]
    y = df['energy_consumption']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_consumption(model, hour, day_of_week, temperature):
    return model.predict([[hour, day_of_week, temperature]])[0]

def calculate_cost(consumption, rate):
    return consumption * rate / 100  # Assuming rate is in cents per kWh

def main():
    st.title("SmartGrid AI: Energy Demand Response System")
    st.write("Optimize your energy usage with AI-driven insights and recommendations.")

    # Generate mock data and train model
    df = generate_mock_data()
    model = train_model(df)

    # Create two columns for layout
    col1, col2 = st.columns([2, 1])

    with col2:
        st.sidebar.header("User Inputs")
        current_time = datetime.now()
        selected_date = st.sidebar.date_input("Select Date", current_time)
        selected_hour = st.sidebar.slider("Select Hour", 0, 23, current_time.hour)
        temperature = st.sidebar.slider("Temperature (°C)", -10, 40, 20)
        energy_rate = st.sidebar.number_input("Energy Rate (cents per kWh)", min_value=1.0, max_value=50.0, value=15.0)

        st.sidebar.markdown("---")
        st.sidebar.subheader("Appliance Control")
        ac_on = st.sidebar.checkbox("Air Conditioning", value=True)
        ev_charging = st.sidebar.checkbox("EV Charging")
        smart_thermostat = st.sidebar.slider("Smart Thermostat Setting (°C)", 18, 28, 22)

    with col1:
        # Current Prediction
        st.subheader("Current Energy Snapshot")
        day_of_week = selected_date.weekday()
        predicted_consumption = predict_consumption(model, selected_hour, day_of_week, temperature)
        cost = calculate_cost(predicted_consumption, energy_rate)

        col_pred1, col_pred2, col_pred3 = st.columns(3)
        col_pred1.metric("Predicted Consumption", f"{predicted_consumption:.2f} kWh")
        col_pred2.metric("Estimated Cost", f"${cost:.2f}")
        col_pred3.metric("Grid Load", "Medium" if 10 <= predicted_consumption <= 20 else ("High" if predicted_consumption > 20 else "Low"))

        # Optimization suggestions
        st.subheader("Smart Optimization Suggestions")
        off_peak_hour = (selected_hour + 6) % 24
        off_peak_consumption = predict_consumption(model, off_peak_hour, day_of_week, temperature)
        off_peak_cost = calculate_cost(off_peak_consumption, energy_rate)
        savings = cost - off_peak_cost

        st.write(f"🕒 Suggested off-peak usage time: {off_peak_hour:02d}:00")
        st.write(f"💡 Estimated off-peak consumption: {off_peak_consumption:.2f} kWh")
        st.write(f"💰 Potential savings: ${savings:.2f}")

        if ev_charging:
            st.write("🚗 Consider delaying EV charging to off-peak hours for maximum savings.")
        if ac_on and temperature > 25:
            st.write(f"❄️ Adjusting your AC from {smart_thermostat}°C to {smart_thermostat + 2}°C could save up to 10% on cooling costs.")

        # 24-Hour Energy Consumption Forecast
        st.subheader("24-Hour Energy Consumption Forecast")
        hours = list(range(24))
        consumptions = [predict_consumption(model, h, day_of_week, temperature) for h in hours]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hours, y=consumptions, mode='lines+markers', name='Predicted Consumption',
                                 line=dict(color='#1f618d', width=2), marker=dict(size=6)))
        fig.update_layout(
            title='Hourly Energy Consumption Prediction',
            xaxis_title='Hour of Day',
            yaxis_title='Energy Consumption (kWh)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
            xaxis=dict(gridcolor='rgba(0,0,0,0.1)')
        )
        st.plotly_chart(fig, use_container_width=True)

        # Historical Data Visualization
        st.subheader("Historical Energy Consumption")
        fig_historical = px.line(df, x='timestamp', y='energy_consumption', title='Past 30 Days Energy Consumption')
        fig_historical.update_layout(xaxis_title="Date", yaxis_title="Energy Consumption (kWh)")
        st.plotly_chart(fig_historical, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("📊 SmartGrid AI - Empowering you with intelligent energy management.")
    st.markdown("⚠️ Note: This is a demo using simulated data. For real-world applications, please consult with energy professionals.")

if __name__ == "__main__":
    main()
