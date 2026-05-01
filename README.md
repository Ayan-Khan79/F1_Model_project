# 🏎️ F1 AI Pit Wall: Predictive Analytics & Agentic Engineering

An end-to-end AI dashboard that combines Machine Learning predictions with real-time F1 telemetry and an LLM-powered Race Engineer.

## 🌟 Core Features
- **📊 Predictive Timing Engine:** A Random Forest model (R²: 0.98) that estimates session durations based on circuit, session type, and start time.
- **📡 Live Telemetry Integration:** Real-time track weather and driver intervals fetched via the **OpenF1 API**.
- **🤖 AI Race Engineer:** A specialized agent (Llama 3.3) equipped with domain-specific guardrails, capable of analyzing both historical dataframes and live telemetry.
- **📈 Advanced Analytics:** Interactive Plotly visualizations for circuit comparisons, feature importance, and duration distributions.

## 🛠️ Technical Stack
- **Languages:** Python
- **AI/ML:** Scikit-Learn, LangChain, Groq (Llama 3.3-70b)
- **Data:** Pandas, OpenF1 API
- **Frontend:** Streamlit, Plotly
- **Environment:** Dotenv for API security

## 🏗️ Architecture
The system uses a **ReAct Agent** framework. When a user asks a question, the agent:
1. Checks the **historical dataframe** for schedule data.
2. Injects **Live Telemetry** if the OpenF1 API is active.
3. Uses its **Internal Knowledge** for historical or driver-specific queries.
4. Operates under strict **Domain Guardrails** to maintain focus on F1.

## 🚀 Installation & Setup
1. Clone the repo: `git clone https://github.com/yourusername/f1-ai-pitwall.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file and add your `GROQ_API_KEY`.
4. Run the app: `streamlit run app.py`

---
**Developed by Mohammad Ayan**  
*MERN Specialist | Final Year CSE Student | Analyst Trainee at Capgemini*