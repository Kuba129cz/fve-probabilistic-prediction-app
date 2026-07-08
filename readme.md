```markdown
# ☀️ PV Probabilistic Prediction App

An interactive Streamlit web application designed for **probabilistic forecasting of photovoltaic (PV) power production**. The app utilizes a Deep Learning model with an attention mechanism and leverages a **5-model ensemble** to deliver robust quantile predictions (10%, 30%, 50%, 70%, 90% confidence bands) along with historical data and weather forecast integration.

🚀 **Live Demo:** [Run the application here](https://jha-fve-probabilistic-prediction-app.streamlit.app/)

## 📊 Features

- **Probabilistic Forecasting:** Generates multiple quantiles to visualize prediction uncertainty, helping to better assess solar power availability.
- **Model Ensembling:** Averages predictions from 5 distinct trained PyTorch models to reduce noise and eliminate anomalies.
- **Interactive Visualization:** Powered by Plotly, offering unified hover modes and custom confidence intervals (10%-90% and 30%-70% bands) mapped against actual production data.
- **Automated Data Pipeline:** Fetches historical weather/production data and future forecasts through an API, automatically managing feature scaling.

---

## 📁 Project Structure

Based on the repository architecture, the project is organized as follows:

```text
FVE-PROBABILISTIC-PREDICTION-APP/
│
├── artifacts/
│   └── model_weights/          # Pre-trained PyTorch weights (Folds 1-5)
│       ├── best_model_probability_preds_1.pth
│       └── ...
│
├── scalers/                    # Serialized data scalers (scikit-learn)
│   ├── feature_scaler.pkl
│   ├── scaled_features_list.pkl
│   └── target_scaler.pkl
│
├── src/                        # Core backend codebase
│   ├── models/
│   │   └── model_attention.py  # PyTorch Neural Network architecture (Attention-based)
│   ├── data_processing.py      # API data fetching and processing
│   ├── dataset.py              # PyTorch Dataset definitions
│   ├── meteo.py                # Meteorological data helpers
│   ├── predict.py              # Ensemble inference pipeline
│   └── preprocessing.py        # Data cleaning and scaling logic
│
├── .env                        # Local environment variables (API credentials & GPS)
├── .gitignore
├── app.py                      # Main Streamlit user interface frontend
└── requirements.txt            # Python dependencies

```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/Kuba129cz/fve-probabilistic-prediction-app.git](https://github.com/Kuba129cz/fve-probabilistic-prediction-app.git)
cd fve-probabilistic-prediction-app

```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Configure Environment Variables

Create a `.env` file in the root directory to store your API credentials and solar plant location:

```env
USERNAME=your_api_username
PASSWORD=your_api_password
LATITUDE=50.0755
LONGITUDE=14.4378

```

---

## 🚀 Running the App

Start the Streamlit dashboard by running the following command:

```bash
streamlit run app.py

```

Open your browser and navigate to the local URL provided in the terminal (usually `http://localhost:8501`).

---

## 🧠 How It Works

1. **User Input:** The user selects a target prediction date in the Streamlit UI form.
2. **Data Ingestion (`src/data_processing.py`):** The app fetches weather forecasts for the target day and meteorological/production history for the previous day.
3. **Preprocessing (`src/preprocessing.py`):** Features are transformed using saved configurations inside the `scalers/` directory.
4. **Ensemble Inference (`src/predict.py`):** - 5 PyTorch models with an Attention layer are loaded.
* Predictions are stacked into a 3D tensor `(5, 24, 5)` and averaged via `torch.mean(..., dim=0)`.


5. **Dashboard Rendering (`app.py`):** The final NumPy array is wrapped into a Pandas DataFrame and rendered as an interactive chart.