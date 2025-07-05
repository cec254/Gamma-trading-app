
# 📈 FX Options Dashboard

An interactive dashboard built using **Dash** and **Plotly** to visualize European-style call and put options on foreign exchange (FX) rates. 

## 🔧 Features
- Live implied volatility vs. strike (volatility smile)
- Simulated PnL chart for average strike options
- Bid-Ask spread histogram
- Auto-refreshes CSV data every 5 minutes

---

## 🚀 Deployment (Render.com)

### 1. Upload Project
- Create a new Web Service at [Render.com](https://render.com)
- Upload your project files or connect to your GitHub repo

### 2. Render Settings
- **Build Command**: *(Leave blank)*
- **Start Command**:
  ```
  gunicorn fx_options_dashboard:app.server
  ```
- **Python Version**: 3.10+
- **Runtime**: Python 3
- Add a `requirements.txt` file with:
  ```
  dash
  plotly
  pandas
  numpy
  gunicorn
  ```

### 3. CSV File
- Upload your `forex_options_latest.csv` to the `output/` folder
- Make sure your app uses this filename (or update `csv_path` in the script)

---

## 📁 File Structure
```
fx_options_dashboard/
├── fx_options_dashboard.py
├── requirements.txt
└── output/
    └── forex_options_latest.csv  # <== Your FX options data here
```

---

## 📡 Running Locally
```bash
pip install -r requirements.txt
python fx_options_dashboard.py
```
Then open your browser to `http://127.0.0.1:8050`

---

Created by ChatGPT 🤖
