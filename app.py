import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import requests
import ftplib
from io import BytesIO
import hashlib
import hmac
import time
from datetime import datetime, timedelta
import logging
import os  # For environment variables

# ========================
# 1. Security and Authentication
# ========================
def check_password():
    """Password protection with hashed credentials"""
    def password_entered():
        entered_hash = hashlib.sha256(st.session_state["password"].encode()).hexdigest()
        stored_hash = os.getenv("PASSWORD_HASH", "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8")  # Default: 'password'
        if hmac.compare_digest(entered_hash, stored_hash):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
    
    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("Please enter the password to access this tool")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True

if not check_password():
    st.stop()

# ========================
# 2. API Configuration
# ========================
# Use environment variables for API keys
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "KGBNPDAXPSGF22FJ")
FRED_API_KEY = os.getenv("FRED_API_KEY", "9f39150c0d0e3eb25f9df72bf04e05e1")

# ========================
# 3. Free Data Integrations
# ========================
def get_cme_fx_options(currency_pair="EUR/USD"):
    """Get CME FX options data via FTP"""
    try:
        # Map currency pairs to CME symbols
        cme_symbols = {
            "EUR/USD": "euro_fx",
            "USD/JPY": "jpy_fx",
            "GBP/USD": "gbp_fx",
            "USD/CAD": "cad_fx"
        }
        symbol = cme_symbols.get(currency_pair, "euro_fx")
        
        # Connect to CME FTP
        ftp = ftplib.FTP("ftp.cmegroup.com")
        ftp.login()  # Anonymous login
        
        # Navigate to settlement directory
        ftp.cwd(f"settle/{symbol}")
        
        # Find latest file (format: euro_fx_YYMMDD.csv)
        files = [f for f in ftp.nlst() if f.startswith(symbol) and f.endswith(".csv")]
        if not files:
            return None
            
        latest_file = sorted(files)[-1]  # Get most recent
        
        # Download file
        bio = BytesIO()
        ftp.retrbinary(f"RETR {latest_file}", bio.write)
        bio.seek(0)
        
        # Parse CSV - CME format is messy so we need to handle it carefully
        df = pd.read_csv(bio, skiprows=1)
        
        # Clean up column names
        df.columns = [col.strip() for col in df.columns]
        
        # Preprocess columns
        rename_map = {
            "Strike": "strike",
            "Expiration Date": "expiry",
            "Call Volume": "call_volume",
            "Put Volume": "put_volume",
            "Call Open Interest": "call_oi",
            "Put Open Interest": "put_oi",
            "Settlement": "settle"
        }
        
        # Keep only columns we need
        df = df.rename(columns=rename_map)
        keep_cols = list(rename_map.values())
        df = df[[col for col in keep_cols if col in df.columns]]
        
        # Convert expiry to datetime
        df['expiry'] = pd.to_datetime(df['expiry'])
        df['expiry_days'] = (df['expiry'] - datetime.now()).dt.days
        
        # Melt to long format
        calls = df[['strike', 'expiry_days', 'call_oi', 'settle']].copy()
        calls['type'] = 'call'
        calls = calls.rename(columns={'call_oi': 'open_interest', 'settle': 'iv'})
        
        puts = df[['strike', 'expiry_days', 'put_oi', 'settle']].copy()
        puts['type'] = 'put'
        puts = puts.rename(columns={'put_oi': 'open_interest', 'settle': 'iv'})
        
        return pd.concat([calls, puts], ignore_index=True).dropna()
    
    except Exception as e:
        logging.error(f"CME data error: {str(e)}")
        st.error(f"CME data error: {str(e)}")
        return None

def get_alpha_vantage_forex(from_currency, to_currency):
    """Get real-time FX rates from Alpha Vantage"""
    url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={from_currency}&to_currency={to_currency}&apikey={ALPHA_VANTAGE_API_KEY}"
    
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if "Realtime Currency Exchange Rate" in data:
            rate = float(data['Realtime Currency Exchange Rate']['5. Exchange Rate'])
            return rate
        return None
    except Exception as e:
        logging.error(f"Alpha Vantage API error: {str(e)}")
        return None

def get_fred_interest_rate(currency):
    """Get risk-free rates from FRED"""
    series_map = {
        "USD": "DFF",  # Fed Funds Rate
        "EUR": "ECBESTRVOLWGTTRMDKN",  # Euro Short-Term Rate
        "JPY": "IRSTCI01JPM156N",  # Japan Interest Rate
        "GBP": "IUDSOIA",  # UK Bank Rate
        "CAD": "IRSTCI01CAM156N"   # Canada Interest Rate
    }
    
    series_id = series_map.get(currency, "DFF")
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json&limit=1&sort_order=desc"
    
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        # Get latest rate
        if "observations" in data and len(data["observations"]) > 0:
            latest_rate = float(data['observations'][0]['value'])
            return latest_rate / 100  # Convert to decimal
        return None
    except Exception as e:
        logging.error(f"FRED API error: {str(e)}")
        return None

def get_historical_volatility(currency_pair, days=30):
    """Calculate historical volatility using Alpha Vantage"""
    from_curr, to_curr = currency_pair.split("/")
    
    url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={from_curr}&to_symbol={to_curr}&apikey={ALPHA_VANTAGE_API_KEY}&outputsize=compact"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if "Time Series FX (Daily)" in data:
            df = pd.DataFrame(data["Time Series FX (Daily)"]).T
            df = df.rename(columns={"4. close": "close"})
            df["close"] = df["close"].astype(float)
            df = df.iloc[:days]  # Get last N days
            
            # Calculate daily returns and volatility
            returns = np.log(df['close'] / df['close'].shift(1))
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            return volatility
        return None
    except Exception as e:
        logging.error(f"Alpha Vantage historical volatility error: {str(e)}")
        return None

# ========================
# 4. Core Gamma Calculations
# ========================
def fx_option_gamma(S, K, T, r_d, r_f, sigma, option_type):
    """Calculate gamma for FX options using Garman-Kohlhagen model"""
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) * np.exp(-r_f * T) / (S * sigma * np.sqrt(T))
    return gamma

def calculate_gex(spot_price, options_df, domestic_rate, foreign_rate):
    """Calculate Gamma Exposure (GEX)"""
    if options_df.empty:
        return 0, pd.DataFrame()
        
    options_df = options_df.copy()
    
    # Calculate gamma for each option
    options_df['gamma'] = options_df.apply(
        lambda x: fx_option_gamma(
            spot_price, 
            x['strike'], 
            x['expiry_days']/365, 
            domestic_rate, 
            foreign_rate, 
            x.get('iv', 0.1),  # Default IV if missing
            x['type']
        ), axis=1
    )
    
    # Calculate GEX contribution
    options_df['gex_contribution'] = (spot_price ** 2) * 0.01 * options_df['gamma'] * options_df['open_interest']
    total_gex = options_df['gex_contribution'].sum()
    
    return total_gex, options_df

def detect_gamma_walls(options_df, percentile=0.85):
    """Identify significant gamma concentration levels"""
    if options_df.empty:
        return pd.Series(dtype=float)
    
    # Cluster nearby strikes
    options_df['strike_group'] = (options_df['strike'] // 0.005) * 0.005
    gamma_walls = options_df.groupby('strike_group')['gex_contribution'].sum().abs()
    
    # Identify significant walls
    threshold = gamma_walls.quantile(percentile)
    significant_walls = gamma_walls[gamma_walls > threshold].sort_values(ascending=False)
    
    return significant_walls

def generate_entry_signals(spot, gamma_walls, options_df, iv_threshold=0.8):
    """Generate trading signals based on gamma positioning"""
    signals = []
    
    if options_df.empty:
        return pd.DataFrame(signals)
    
    # 1. Gamma Flip Zones
    for strike, gex_strength in gamma_walls.items():
        distance = abs(spot - strike)
        if distance < 0.005:  # Very near wall
            signal_type = "Breakout" if gex_strength < 0 else "Reversal"
            strength = "🔥 High" if distance < 0.002 else "⚠️ Medium"
            signals.append({
                'strike': strike,
                'signal': f"{signal_type} Likely",
                'type': signal_type,
                'strength': strength,
                'distance': distance,
                'reason': f"Gamma Wall at {strike}"
            })
    
    # 2. Gamma-VOL Confluence
    if 'iv' in options_df:
        iv_threshold_val = options_df['iv'].quantile(iv_threshold)
        gamma_threshold_val = options_df['gex_contribution'].abs().quantile(0.9)
        
        high_iv_gamma = options_df[
            (options_df['iv'] > iv_threshold_val) & 
            (options_df['gex_contribution'].abs() > gamma_threshold_val)
        ]
        
        for _, row in high_iv_gamma.iterrows():
            signals.append({
                'strike': row['strike'],
                'signal': "Long Opportunity" if row['gex_contribution'] > 0 else "Short Opportunity",
                'type': "Premium" if row['gex_contribution'] > 0 else "Discount",
                'strength': "💎 High Value",
                'distance': abs(spot - row['strike']),
                'reason': f"High IV ({row['iv']*100:.1f}%) + Gamma"
            })
    
    return pd.DataFrame(signals)

def profit_maximization(spot, strike, option_type, gex, iv, risk_amount):
    """Calculate optimal position sizing and targets"""
    # Calculate gamma-based position sizing
    gamma_weight = min(1.0, abs(gex) / 1000000)
    position_size = risk_amount * gamma_weight * 10000
    
    # Calculate volatility-adjusted targets
    atr = iv * spot * np.sqrt(1/252)
    if option_type == "call":
        profit_target = strike + (3 * atr)
        stop_loss = strike - (1.5 * atr)
    else:
        profit_target = strike - (3 * atr)
        stop_loss = strike + (1.5 * atr)
    
    # Time decay adjustment
    theta_factor = max(0.7, 1 - (iv * 0.1))
    
    rr_ratio = abs(profit_target - strike) / abs(strike - stop_loss)
    
    return {
        "position_size": int(position_size),
        "profit_target": round(profit_target, 4),
        "stop_loss": round(stop_loss, 4),
        "risk_reward": round(rr_ratio, 1),
        "theta_adjustment": theta_factor
    }

def create_heatmap(options_df):
    """Create gamma exposure heatmap across strikes and expiries"""
    if options_df.empty or 'expiry_days' not in options_df:
        return None
        
    pivot = options_df.pivot_table(index='strike', 
                                  columns='expiry_days', 
                                  values='gex_contribution', 
                                  aggfunc='sum')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, cmap="vlag", center=0, annot=False, fmt=".0f")
    plt.title("Gamma Exposure Heatmap")
    plt.xlabel("Days to Expiry")
    plt.ylabel("Strike Price")
    return plt

# ========================
# 5. Streamlit App Setup
# ========================
st.set_page_config(
    page_title="FX Gamma Exposure Analyzer", 
    layout="wide",
    page_icon="📊"
)

st.title("📊 FX Gamma Exposure Analyzer")
st.markdown("""
    **Professional gamma exposure analysis using free data sources**  
    *Powered by CME Group, Alpha Vantage, and FRED data*
""")

# ========================
# 6. Sidebar Configuration
# ========================
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Currency pair selection
    currency_pair = st.selectbox("Currency Pair", 
                               ["EUR/USD", "USD/JPY", "GBP/USD", "USD/CAD"],
                               index=0)
    
    # Risk parameters
    risk_amount = st.number_input("Risk Amount ($)", 
                                 min_value=100, 
                                 value=1000, 
                                 step=100)
    
    # Gamma analysis settings
    gamma_threshold = st.slider("Gamma Wall Threshold", 
                               min_value=70, 
                               max_value=95, 
                               value=85, 
                               step=5,
                               help="Percentile for significant gamma walls")
    
    iv_percentile = st.slider("IV Percentile Filter", 
                             min_value=0.65, 
                             max_value=0.95, 
                             value=0.80, 
                             step=0.05,
                             help="Filter for high IV opportunities")
    
    # Data refresh
    if st.button("🔄 Refresh Data"):
        st.experimental_rerun()

# ========================
# 7. Data Loading
# ========================
def load_market_data(currency_pair):
    """Load all required market data"""
    # Split currency pair
    from_curr, to_curr = currency_pair.split("/")
    
    # Get spot price
    spot_price = get_alpha_vantage_forex(from_curr, to_curr)
    if spot_price is None:
        # Fallback to manual input if API fails
        spot_price = 1.0800 if currency_pair == "EUR/USD" else 110.00
        st.warning("Using fallback spot price - Alpha Vantage API limit may be reached")
    
    # Get interest rates
    domestic_rate = get_fred_interest_rate(to_curr)
    foreign_rate = get_fred_interest_rate(from_curr)
    
    if domestic_rate is None:
        domestic_rate = 0.05
        st.warning("Using fallback domestic interest rate")
    
    if foreign_rate is None:
        foreign_rate = 0.03
        st.warning("Using fallback foreign interest rate")
    
    # Get options data
    options_df = get_cme_fx_options(currency_pair)
    
    # If CME data fails, create mock data
    if options_df is None or options_df.empty:
        st.warning("Using simulated options data - CME connection failed")
        options_df = pd.DataFrame([
            {'strike': spot_price * 0.98, 'expiry_days': 30, 'iv': 0.08, 'type': 'call', 'open_interest': 1000},
            {'strike': spot_price * 0.99, 'expiry_days': 30, 'iv': 0.085, 'type': 'call', 'open_interest': 800},
            {'strike': spot_price, 'expiry_days': 30, 'iv': 0.09, 'type': 'put', 'open_interest': 1500},
            {'strike': spot_price * 1.01, 'expiry_days': 30, 'iv': 0.085, 'type': 'put', 'open_interest': 1200},
            {'strike': spot_price * 1.02, 'expiry_days': 30, 'iv': 0.10, 'type': 'call', 'open_interest': 800},
        ])
    
    # Calculate historical volatility if needed
    hist_vol = get_historical_volatility(currency_pair)
    if hist_vol is not None and 'iv' in options_df:
        options_df['iv'] = options_df['iv'].fillna(hist_vol)
    
    return spot_price, domestic_rate, foreign_rate, options_df

# Load market data with progress
with st.spinner("Loading market data from free sources..."):
    spot_price, domestic_rate, foreign_rate, options_df = load_market_data(currency_pair)

# ========================
# 8. Gamma Exposure Calculation
# ========================
with st.spinner("Calculating gamma exposure..."):
    total_gex, options_df = calculate_gex(spot_price, options_df, domestic_rate, foreign_rate)
    gamma_walls = detect_gamma_walls(options_df, gamma_threshold/100)
    entry_signals = generate_entry_signals(spot_price, gamma_walls, options_df, iv_percentile)

# ========================
# 9. Dashboard Layout
# ========================
# Market Overview
col1, col2, col3 = st.columns(3)
col1.metric("Spot Price", f"{spot_price:.4f}")
col2.metric("Gamma Exposure", f"{total_gex:,.0f}", 
           "Long Gamma (Stabilizing)" if total_gex > 0 else "Short Gamma (Volatile)")
col3.metric("Market Status", "Normal", 
           "High Volatility" if options_df['iv'].mean() > 0.1 else "Low Volatility")

# Gamma Walls
st.subheader("🚧 Key Gamma Walls")
if not gamma_walls.empty:
    walls_df = gamma_walls.reset_index()
    walls_df.columns = ['Strike Zone', 'Gamma Strength']
    walls_df['Distance to Spot'] = abs(walls_df['Strike Zone'] - spot_price)
    walls_df['Impact'] = walls_df['Gamma Strength'].apply(
        lambda x: "High" if x > gamma_walls.quantile(0.9) else "Medium")
    
    # Sort by distance to spot
    walls_df = walls_df.sort_values('Distance to Spot')
    st.dataframe(walls_df.style.format({'Gamma Strength': '{:,.0f}', 'Distance to Spot': '{:.4f}'}))
else:
    st.warning("No significant gamma walls detected")

# Trading Signals
st.subheader("💡 Trading Signals")
if not entry_signals.empty:
    entry_signals = entry_signals.sort_values('distance')
    
    # Color coding for signals
    def color_signal(signal):
        if "Long" in signal or "Reversal" in signal:
            return 'background-color: #4CAF50; color: white'
        elif "Short" in signal or "Breakout" in signal:
            return 'background-color: #F44336; color: white'
        return ''
    
    st.dataframe(
        entry_signals.style.applymap(color_signal, subset=['signal']),
        hide_index=True
    )
else:
    st.info("No strong trading signals detected. Try adjusting parameters.")

# Gamma Visualization Tabs
tab1, tab2, tab3 = st.tabs(["Gamma by Strike", "Gamma Heatmap", "Sensitivity Analysis"])

with tab1:
    st.subheader("Gamma Exposure by Strike")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if not options_df.empty:
        calls = options_df[options_df['type'] == 'call']
        puts = options_df[options_df['type'] == 'put']
        
        ax.bar(calls['strike'], calls['gex_contribution'], width=0.001, color='green', label='Calls')
        ax.bar(puts['strike'], puts['gex_contribution'], width=0.001, color='red', label='Puts')
        
        # Add gamma walls
        for wall in gamma_walls.index:
            ax.axvline(x=wall, color='blue', linestyle='--', alpha=0.3)
            ax.text(wall, ax.get_ylim()[1]*0.9, f'Gamma Wall', rotation=90, 
                    verticalalignment='top', fontsize=8, color='blue')
        
        # Add spot line
        ax.axvline(x=spot_price, color='black', linestyle='-', linewidth=2)
        ax.text(spot_price, ax.get_ylim()[1]*0.95, 'Current Spot', 
                horizontalalignment='center', fontsize=9, backgroundcolor='white')
        
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel("Strike Price")
        ax.set_ylabel("GEX Contribution")
        ax.set_title(f"{currency_pair} Gamma Exposure")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("No options data available for visualization")

with tab2:
    st.subheader("Gamma Exposure Heatmap")
    if not options_df.empty and 'expiry_days' in options_df:
        heatmap_fig = create_heatmap(options_df)
        if heatmap_fig:
            st.pyplot(heatmap_fig)
        else:
            st.warning("Could not generate heatmap - insufficient data")
    else:
        st.warning("No options data available for heatmap")

with tab3:
    st.subheader("Sensitivity Analysis")
    
    if not options_df.empty:
        # Spot sensitivity
        spot_range = np.linspace(spot_price * 0.98, spot_price * 1.02, 20)
        gex_spot = [calculate_gex(S, options_df, domestic_rate, foreign_rate)[0] for S in spot_range]
        
        # IV sensitivity
        iv_range = np.linspace(-0.5, 0.5, 20)
        gex_iv = []
        for iv_change in iv_range:
            temp_df = options_df.copy()
            if 'iv' in temp_df:
                temp_df['iv'] = temp_df['iv'] * (1 + iv_change)
            gex_iv.append(calculate_gex(spot_price, temp_df, domestic_rate, foreign_rate)[0])
        
        # Plot
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(spot_range, gex_spot, marker='o')
        ax1.axvline(spot_price, color='r', linestyle='--')
        ax1.set_title("GEX Sensitivity to Spot Price")
        ax1.set_xlabel("Spot Price")
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        ax2.plot(iv_range, gex_iv, marker='o', color='orange')
        ax2.axvline(0, color='r', linestyle='--')
        ax2.set_title("GEX Sensitivity to Implied Volatility")
        ax2.set_xlabel("IV Change (%)")
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        st.pyplot(fig2)
    else:
        st.warning("No options data available for sensitivity analysis")

# Profit Maximization
st.subheader("💰 Profit Maximization Planner")
if not entry_signals.empty:
    selected_signal = st.selectbox("Select Trading Signal", 
                                  entry_signals['reason'] + " @ " + entry_signals['strike'].astype(str))
    
    if selected_signal:
        selected_row = entry_signals[
            entry_signals['reason'] + " @ " + entry_signals['strike'].astype(str) == selected_signal
        ].iloc[0]
        
        # Get gamma value for selected strike
        strike_gamma = options_df[options_df['strike'] == selected_row['strike']]
        if not strike_gamma.empty:
            strike_gamma = strike_gamma['gamma'].mean()
        else:
            strike_gamma = 0.1  # Default if not found
        
        # Calculate profit parameters
        trade_params = profit_maximization(
            spot_price,
            selected_row['strike'],
            "call" if "Long" in selected_row['signal'] else "put",
            strike_gamma * 1000000,  # Scale gamma
            options_df['iv'].mean() if 'iv' in options_df else 0.1,
            risk_amount
        )
        
        # Display trade plan
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Position Size", f"${trade_params['position_size']:,}")
        col2.metric("Profit Target", trade_params['profit_target'])
        col3.metric("Stop Loss", trade_params['stop_loss'])
        col4.metric("Risk/Reward", f"1:{trade_params['risk_reward']}")
        
        st.info(f"**Time Decay Factor:** {trade_params['theta_adjustment']:.2f} - " +
                ("Higher IV requires larger moves" if trade_params['theta_adjustment'] < 0.9 else "Normal time decay"))

# Strategy Guide
with st.expander("📚 Gamma Trading Strategy Guide"):
    st.markdown("""
    ### How to Trade Gamma Exposure
    
    **Gamma Walls Strategy**:
    - Prices tend to reverse near positive gamma walls (blue lines)
    - Prices tend to break through negative gamma walls
    - Trade direction: Buy pullbacks to positive walls, sell rallies to negative walls
    
    **High IV Gamma Opportunities**:
    - Look for high IV (>80th percentile) with positive gamma
    - These setups offer premium selling opportunities with gamma protection
    - Position sizing: Use 50-70% of normal size due to volatility
    
    **Profit Maximization Rules**:
    1. Always use volatility-adjusted stops (1.5x ATR)
    2. Take profits at 3x ATR from entry
    3. Adjust position size based on gamma strength
    4. Reduce trade duration when IV > 15%
    
    **Data Limitations**:
    - CME data is delayed by 10-15 minutes
    - Alpha Vantage has 5 requests/minute, 500 requests/day limit
    - Use signals as guidance, not trade recommendations
    """)

# Footer
st.divider()
st.caption("ℹ️ Data Sources: CME Group (delayed options data), Alpha Vantage (FX rates), FRED (interest rates)")
st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption(f"Alpha Vantage API Key: {ALPHA_VANTAGE_API_KEY[:4]}...{ALPHA_VANTAGE_API_KEY[-4:]}")
st.caption(f"FRED API Key: {FRED_API_KEY[:4]}...{FRED_API_KEY[-4:]}")

# ========================
# 10. Deployment Notes
# ========================
'''
## Deployment Instructions:

1. Create a new GitHub repository with:
   - This file named `app.py`
   - A `requirements.txt` file containing:
        streamlit
        numpy
        pandas
        matplotlib
        seaborn
        scipy
        requests

2. Go to Streamlit Community Cloud (https://share.streamlit.io/)
3. Connect your GitHub repository
4. Set main file path to `app.py`
5. Deploy!

## Password Note:
- Default password is 'password'
- To change password:
  1. Generate new hash: 
     ```python
     import hashlib
     hashlib.sha256("your-new-password".encode()).hexdigest()
     ```
  2. Replace hash in the `check_password()` function
'''