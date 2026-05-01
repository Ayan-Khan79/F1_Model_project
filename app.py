import streamlit as st
import pandas as pd
import joblib
import datetime
import plotly.express as px
import os
import warnings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent
import requests

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
warnings.filterwarnings("ignore")

st.set_page_config(page_title="F1 Timing Predictor & AI Assistant", page_icon="🏎️", layout="wide")

# 1. LOAD ASSETS (No training happens here)
@st.cache_resource
def load_assets():
    model = joblib.load('data/f1_predictor.pkl')
    df = pd.read_csv('data/sessions.csv')
    # Minimal processing needed for UI display only
    df['date_start'] = pd.to_datetime(df['date_start'])
    df['date_end'] = pd.to_datetime(df['date_end'])
    df['duration_mins'] = (df['date_end'] - df['date_start']).dt.total_seconds() / 60
    return model, df

model, df = load_assets()

import requests

def fetch_live_f1_data(endpoint, params=None):
    """
    Fetches real-time data from OpenF1 API.
    Common endpoints: 'weather', 'sessions', 'intervals', 'laps'
    """
    base_url = "https://api.openf1.org/v1/"
    try:
        # We use 'session_key=latest' to always get the most recent data
        url = f"{base_url}{endpoint}"
        query_params = {"session_key": "latest"}
        if params:
            query_params.update(params)
            
        response = requests.get(url, params=query_params, timeout=5)
        if response.status_code == 200:
            return pd.DataFrame(response.json())
    except Exception as e:
        return None
    return None

# 2. SIDEBAR (User Inputs)
st.sidebar.header("🏎️ Race Control")
circuit = st.sidebar.selectbox("Select Circuit", sorted(df['circuit_short_name'].unique()))
s_type = st.sidebar.radio("Session Type", ["Practice", "Qualifying", "Race"])
start_time = st.sidebar.time_input("Expected Start Time (UTC)", datetime.time(14, 0))
# Add this in your SIDEBAR section
st.sidebar.divider()
st.sidebar.subheader("📡 Live Track Status")

if st.sidebar.toggle("Connect to OpenF1 Live"):
    weather_df = fetch_live_f1_data('weather')
    if weather_df is not None and not weather_df.empty:
        latest = weather_df.iloc[-1]
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Air Temp", f"{latest['air_temperature']}°C")
        col2.metric("Track Temp", f"{latest['track_temperature']}°C")
        st.sidebar.caption(f"Last Updated: {latest['date'][11:19]} UTC")
    else:
        st.sidebar.warning("No live session active.")

# 3. UI TABS
tab1, tab2, tab3, tab4 = st.tabs(["📊 Predictor", "🌍 Map", "📈 Analytics", "🤖 AI Pit Wall"])

with tab1:
    st.subheader("🏁 Timing Prediction")
    # Quick mapping for prediction input
    is_p, is_q, is_r = (1,0,0) if s_type == "Practice" else (0,1,0) if s_type == "Qualifying" else (0,0,1)
    input_df = pd.DataFrame([[start_time.hour, 6, is_p, is_q, is_r]], 
                            columns=['start_hour', 'day_of_week', 'session_type_Practice', 'session_type_Qualifying', 'session_type_Race'])
    
    prediction = model.predict(input_df)[0]
    st.metric(label="Estimated Duration", value=f"{int(prediction)} Minutes")
    st.success(f"**Predicted End Time:** {(datetime.datetime.combine(datetime.date.today(), start_time) + datetime.timedelta(minutes=prediction)).strftime('%H:%M')} UTC")

with tab2:
    st.subheader("🌍 F1 World Tour 2023")
    map_df = df.drop_duplicates(subset=['circuit_short_name'])
    
    fig_map = px.scatter_geo(
        map_df, 
        locations="country_name", 
        locationmode='country names',
        hover_name="circuit_short_name", 
        color="country_name", 
        template="plotly_dark",
        projection="natural earth", # Added for a more professional look
        height=700 # Increased height to make the map larger
    )
    
    # Update layout to remove unnecessary margins
    fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig_map, width='stretch')

with tab3:
    st.subheader("📈 Advanced Analytics & Model Insights")
    
    # --- ROW 1: FEATURE IMPORTANCE ---
    st.markdown("### 🤖 Why does the model predict these timings?")
    features = ['start_hour', 'day_of_week', 'session_type_Practice', 'session_type_Qualifying', 'session_type_Race']
    importances = model.feature_importances_
    
    feat_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    fig_importance = px.bar(
        feat_df, x='Importance', y='Feature', orientation='h',
        title="Model Decision Drivers",
        labels={'Importance': 'Impact Score', 'Feature': 'Data Point'},
        template="plotly_dark",
        color='Importance',
        color_continuous_scale='Viridis',
        height=400
    )
    st.plotly_chart(fig_importance, width='stretch')
    
    st.divider()

    # --- ROW 2: SESSION DURATION DISTRIBUTION ---
    st.markdown("### ⏱️ Session Duration Variance")
    # A box plot shows the spread of data - great for an Analyst portfolio
    fig_box = px.box(
        df, 
        x="session_type", 
        y="duration_mins", 
        color="session_type",
        template="plotly_dark",
        title="Duration Spreads by Session Type",
        labels={'duration_mins': 'Minutes', 'session_type': 'Session'}
    )
    st.plotly_chart(fig_box, width='stretch')
    
    st.info("💡 **Analyst Note:** The dots above the boxes represent outliers—these are likely races with significant delays or Red Flags.")

    st.divider()

    # --- ROW 3: CIRCUIT COMPARISON ---
    st.markdown("### 🏁 Circuit Comparison Tool")
    selected_circs = st.multiselect(
        "Select Circuits to Compare", 
        sorted(df['circuit_short_name'].unique()), 
        default=sorted(df['circuit_short_name'].unique())[:3]
    )
    
    if selected_circs:
        comp_df = df[df['circuit_short_name'].isin(selected_circs)]
        fig_bar = px.bar(
            comp_df.groupby(['circuit_short_name', 'session_type'])['duration_mins'].mean().reset_index(), 
            x="circuit_short_name", y="duration_mins", color="session_type", 
            barmode="group", template="plotly_dark",
            labels={'duration_mins': 'Avg Minutes', 'circuit_short_name': 'Circuit'}
        )
        st.plotly_chart(fig_bar, width='stretch')
        
with tab4:
    st.subheader("🤖 AI Race Engineer")
    st.caption("F1 Expert System | Llama 3.3 & LangChain")

    if not GROQ_API_KEY:
        st.error("Missing GROQ_API_KEY in .env file.")
    else:
        # 1. Initialize Agent
        llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)
        
        custom_prefix = (
            "You are a strict F1 Race Engineer. Your knowledge is EXCLUSIVELY limited to Formula 1. "
            "1. For schedule, locations, and timing, use the provided dataframe. "
            "2. For drivers, teams, and F1 history, use your internal knowledge. "
            "3. If LIVE TELEMETRY is provided in the prompt, prioritize it for questions about current track conditions. "
            "4. CRITICAL: If a user asks about anything NOT related to Formula 1, politely refuse. "
            "5. Always maintain a professional, analytical tone."
        )

        agent = create_pandas_dataframe_agent(
            llm, df, prefix=custom_prefix, verbose=False, 
            allow_dangerous_code=True, handle_parsing_errors=True
        )

        # 2. Setup Chat History Container
        chat_container = st.container()

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 3. Display Chat History in the container
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # 4. Chat Input Box
        if prompt := st.chat_input("Ask about the 2023 season or live data..."):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing telemetry..."):
                        # --- LIVE DATA INTEGRATION ---
                        # Attempt to fetch live data to give the agent context
                        live_weather = fetch_live_f1_data('weather')
                        live_intervals = fetch_live_f1_data('intervals')
                        
                        context_str = ""
                        if live_weather is not None and not live_weather.empty:
                            w = live_weather.iloc[-1]
                            context_str += f"\n[LIVE WEATHER: Air {w['air_temperature']}°C, Track {w['track_temperature']}°C]"
                        
                        if live_intervals is not None and not live_intervals.empty:
                            # Get the 3 most recent timing intervals
                            gaps = live_intervals.tail(3)[['driver_number', 'gap_to_leader']].to_string(index=False)
                            context_str += f"\n[LIVE GAPS: \n{gaps}]"
                        
                        # Combine original prompt with live context
                        full_query = f"{context_str}\n\nUser Question: {prompt}" if context_str else prompt
                        
                        # Run the agent
                        response = agent.run(full_query)
                        st.markdown(response)
            
            # Save assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
st.divider()
st.caption("Mohammad Ayan | MERN Specialist & AI Analyst")