import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import math
from datetime import datetime, timedelta
import io
import numpy as np

# ---------------------------------------------------------
# 1. 페이지 설정 및 초기화
# ---------------------------------------------------------
st.set_page_config(page_title="codex 실험용 앱", layout="wide")

if 'search_ticker' not in st.session_state:
    st.session_state['search_ticker'] = 'TQQQ'

def get_db_connection():
    return sqlite3.connect('portfolio.db', timeout=30)

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS holdings
                 (ticker TEXT PRIMARY KEY, shares REAL, avg_price REAL, sort_order INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS cash
                 (currency TEXT PRIMARY KEY, amount REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS trade_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date TEXT, ticker TEXT, action TEXT, shares REAL, price REAL, note TEXT, realized_pnl REAL)''')
    
    try: c.execute("SELECT sort_order FROM holdings LIMIT 1")
    except sqlite3.OperationalError: c.execute("ALTER TABLE holdings ADD COLUMN sort_order INTEGER DEFAULT 99")
    try: c.execute("SELECT realized_pnl FROM trade_logs LIMIT 1")
    except sqlite3.OperationalError: c.execute("ALTER TABLE trade_logs ADD COLUMN realized_pnl REAL DEFAULT 0.0")

    conn.commit()
    conn.close()

def get_portfolio():
    conn = get_db_connection()
    try:
        df_holdings = pd.read_sql("SELECT * FROM holdings ORDER BY sort_order ASC, ticker ASC", conn)
        df_cash = pd.read_sql("SELECT * FROM cash", conn)
    except:
        df_holdings = pd.DataFrame(); df_cash = pd.DataFrame()
    conn.close()
    return df_holdings, df_cash

def add_log(ticker, action, shares, price, note="", pnl=0.0):
    conn = get_db_connection()
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        c.execute("INSERT INTO trade_logs (date, ticker, action, shares, price, note, realized_pnl) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                  (now, ticker, action, shares, price, note, pnl))
    except sqlite3.OperationalError:
        c.execute('''CREATE TABLE IF NOT EXISTS trade_logs
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      date TEXT, ticker TEXT, action TEXT, shares REAL, price REAL, note TEXT, realized_pnl REAL)''')
        c.execute("INSERT INTO trade_logs (date, ticker, action, shares, price, note, realized_pnl) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                  (now, ticker, action, shares, price, note, pnl))
    conn.commit()
    conn.close()

def get_logs():
    conn = get_db_connection()
    try: df = pd.read_sql("SELECT * FROM trade_logs ORDER BY id DESC", conn)
    except: df = pd.DataFrame(columns=['id', 'date', 'ticker', 'action', 'shares', 'price', 'note', 'realized_pnl'])
    conn.close()
    return df

def delete_log(log_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM trade_logs WHERE id=?", (log_id,))
    conn.commit()
    conn.close()
    st.toast(f"✅ 로그 삭제 완료 (ID: {log_id})")

def update_cash(amount):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO cash VALUES (?, ?)", ('USD', amount))
    conn.commit()
    conn.close()

def get_current_cash(conn):
    c = conn.cursor()
    c.execute("SELECT amount FROM cash WHERE currency='USD'")
    row = c.fetchone()
    return row[0] if row else 0.0

# [신규] 데이터 백업 (CSV 다운로드)
def convert_df_to_csv():
    conn = get_db_connection()
    holdings = pd.read_sql("SELECT * FROM holdings", conn)
    cash = pd.read_sql("SELECT * FROM cash", conn)
    conn.close()
    
    # 두 테이블을 하나의 CSV로 합치기 위해 식별자 추가
    holdings['type'] = 'stock'
    cash['type'] = 'cash'
    
    # 컬럼 통일 (cash 테이블엔 ticker, shares 등이 없으므로 조정)
    cash = cash.rename(columns={'currency': 'ticker', 'amount': 'avg_price'}) # amount를 잠시 avg_price 컬럼에 태움
    cash['shares'] = 0
    cash['sort_order'] = 0
    
    merged = pd.concat([holdings, cash], ignore_index=True)
    return merged.to_csv(index=False).encode('utf-8')

# [신규] 데이터 복구 (CSV 업로드)
def restore_from_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        conn = get_db_connection()
        c = conn.cursor()
        
        # 기존 데이터 초기화
        c.execute("DELETE FROM holdings")
        c.execute("DELETE FROM cash")
        
        # 데이터 복원
        for _, row in df.iterrows():
            if row['type'] == 'stock':
                c.execute("INSERT INTO holdings (ticker, shares, avg_price, sort_order) VALUES (?, ?, ?, ?)",
                          (row['ticker'], row['shares'], row['avg_price'], row['sort_order']))
            elif row['type'] == 'cash':
                # 위에서 avg_price에 태웠던 amount 복구
                c.execute("INSERT INTO cash (currency, amount) VALUES (?, ?)", 
                          (row['ticker'], row['avg_price']))
        
        conn.commit()
        conn.close()
        st.success("✅ 데이터 복구 성공! (새로고침 됩니다)")
        st.rerun()
    except Exception as e:
        st.error(f"복구 실패: {e}")

# [매수 Logic]
def add_stock(ticker, new_shares, new_price):
    conn = get_db_connection()
    c = conn.cursor()
    
    current_cash = get_current_cash(conn)
    cost = new_shares * new_price
    new_cash_balance = current_cash - cost
    c.execute("INSERT OR REPLACE INTO cash VALUES (?, ?)", ('USD', new_cash_balance))
    
    c.execute("SELECT shares, avg_price FROM holdings WHERE ticker=?", (ticker,))
    row = c.fetchone()
    
    if row:
        old_shares, old_avg = row
        total_shares = old_shares + new_shares
        total_cost_stock = (old_shares * old_avg) + (new_shares * new_price)
        new_avg = total_cost_stock / total_shares if total_shares > 0 else 0.0
        c.execute("UPDATE holdings SET shares=?, avg_price=? WHERE ticker=?", (total_shares, new_avg, ticker))
        conn.commit(); conn.close()
        add_log(ticker, "추가 매수", new_shares, new_price, f"현금차감: -${cost:,.2f}", 0.0)
        st.toast(f"➕ 매수 완료! 현금 -${cost:,.2f}")
    else:
        c.execute("SELECT MAX(sort_order) FROM holdings")
        res = c.fetchone(); max_order = res[0] if res and res[0] else 0; next_order = max_order + 1
        c.execute("INSERT INTO holdings VALUES (?, ?, ?, ?)", (ticker, new_shares, new_price, next_order))
        conn.commit(); conn.close()
        add_log(ticker, "신규 매수", new_shares, new_price, f"현금차감: -${cost:,.2f}", 0.0)
        st.toast(f"🆕 신규 매수! 현금 -${cost:,.2f}")

# [매도 Logic]
def sell_stock(ticker, sell_shares, sell_price):
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute("SELECT shares, avg_price FROM holdings WHERE ticker=?", (ticker,))
    row = c.fetchone()
    
    if row:
        old_shares, old_avg = row
        if sell_shares > old_shares + 0.000001:
            st.error(f"❌ 매도 불가: 보유 수량({old_shares}주)보다 많이 팔 수 없습니다.")
            conn.close(); return

        current_cash = get_current_cash(conn)
        revenue = sell_shares * sell_price
        new_cash_balance = current_cash + revenue
        c.execute("INSERT OR REPLACE INTO cash VALUES (?, ?)", ('USD', new_cash_balance))

        new_shares = old_shares - sell_shares
        realized_pnl = (sell_price - old_avg) * sell_shares 
        
        if new_shares < 0.000001:
            c.execute("DELETE FROM holdings WHERE ticker=?", (ticker,))
            conn.commit(); conn.close()
            add_log(ticker, "전량 매도", sell_shares, sell_price, f"현금입금: +${revenue:,.2f}", realized_pnl)
            st.toast(f"📉 전량 매도! 현금 +${revenue:,.2f} (이익 ${realized_pnl:,.2f})")
        else:
            c.execute("UPDATE holdings SET shares=? WHERE ticker=?", (new_shares, ticker))
            conn.commit(); conn.close()
            add_log(ticker, "부분 매도", sell_shares, sell_price, f"현금입금: +${revenue:,.2f}", realized_pnl)
            st.toast(f"📉 부분 매도! 현금 +${revenue:,.2f} (이익 ${realized_pnl:,.2f})")
    else:
        conn.close(); st.error(f"❌ 매도 불가: 보유하지 않은 종목입니다 ({ticker})")

def overwrite_stock(ticker, shares, price):
    conn = get_db_connection()
    c = conn.cursor()
    # ????? ??? ??? ?? ??? ?? ?? ??? ??
    cash_before = get_current_cash(conn)
    c.execute("SELECT sort_order FROM holdings WHERE ticker=?", (ticker,))
    row = c.fetchone(); order = row[0] if row else 99
    c.execute("INSERT OR REPLACE INTO holdings VALUES (?, ?, ?, ?)", (ticker, shares, price, order))
    if cash_before is not None:
        c.execute("INSERT OR REPLACE INTO cash VALUES (?, ?)", ('USD', cash_before))
    conn.commit(); conn.close()
    add_log(ticker, "?? ??", shares, price, "?? ?? ?? (????)", 0.0)
    st.toast(f"? ?? ??: {ticker}")
    st.toast(f"✏️ 수정 완료: {ticker}")

def delete_stock(ticker):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM holdings WHERE ticker=?", (ticker,))
    conn.commit(); conn.close()
    add_log(ticker, "종목 삭제", 0, 0, "관리자 삭제", 0.0)
    st.toast(f"🗑️ 삭제 완료: {ticker}")

def update_sort_orders(df_edited):
    conn = get_db_connection()
    c = conn.cursor()
    existing_tickers = df_edited['ticker'].tolist()
    if existing_tickers:
        placeholders = ','.join(['?'] * len(existing_tickers))
        c.execute(f"DELETE FROM holdings WHERE ticker NOT IN ({placeholders})", existing_tickers)
    else: c.execute("DELETE FROM holdings")
    for index, row in df_edited.iterrows():
        c.execute("UPDATE holdings SET sort_order=? WHERE ticker=?", (row['sort_order'], row['ticker']))
    conn.commit(); conn.close()

def set_ticker(ticker): st.session_state['search_ticker'] = ticker

init_db()

# ---------------------------------------------------------
# 상단 대시보드
# ---------------------------------------------------------
def display_global_dashboard():
    tickers = {'NQ=F': '나스닥 100 선물', 'KRW=X': '원/달러 환율', '^VIX': 'VIX 지수 (공포)'}
    cols = st.columns(3)
    for i, (ticker, name) in enumerate(tickers.items()):
        with cols[i]:
            try:
                data = yf.Ticker(ticker).history(period="5d")
                if len(data) >= 2:
                    curr = data['Close'].iloc[-1]; prev = data['Close'].iloc[-2]
                    delta = curr - prev; pct = (delta / prev) * 100
                    val_str = f"{curr:,.2f}"
                    if ticker == 'KRW=X': val_str = f"{curr:,.0f}원"
                    st.metric(label=name, value=val_str, delta=f"{delta:.2f} ({pct:.2f}%)")
                else: st.metric(label=name, value="-", delta="-")
            except: st.metric(label=name, value="Loading...", delta=None)
    st.divider()

display_global_dashboard()

# ---------------------------------------------------------
# 단타(초단기) 신호 계산
# ---------------------------------------------------------
def _pct_from_n(series, n):
    if series is None or len(series) < n or n <= 0:
        return None
    base = series.iloc[-n]
    return None if base == 0 else (series.iloc[-1] / base - 1) * 100

def _clamp(val, lo=-1.0, hi=1.0):
    return max(lo, min(hi, val))

# --- ML Feature Engineering ---
def add_technical_features(df):
    df = df.copy()
    # 데이터가 짧을 때는 긴 이평/밴드 계산을 완화
    if len(df) < 60:
        df['ma5'] = df['Close'].rolling(window=5, min_periods=3).mean()
        df['ma20'] = df['Close'].rolling(window=20, min_periods=5).mean()
        df['ma60'] = df['ma20']  # 대체값으로 사용
    else:
        df['ma5'] = df['Close'].rolling(window=5).mean()
        df['ma20'] = df['Close'].rolling(window=20).mean()
        df['ma60'] = df['Close'].rolling(window=60).mean()
    df['disparity_20'] = (df['Close'] / df['ma20']) - 1

    std20 = df['Close'].rolling(window=20).std()
    df['upper_band'] = df['ma20'] + (std20 * 2)
    df['lower_band'] = df['ma20'] - (std20 * 2)
    df['bb_position'] = (df['Close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_feat'] = 100 - (100 / (1 + rs))

    df['vol_change'] = df['Volume'].pct_change()
    df['return_1d'] = df['Close'].pct_change()
    df['return_2d'] = df['Close'].pct_change(2)
    df = df.dropna()
    return df

def train_prediction_model(df):
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as e:
        return None, None, None, None, "scikit-learn 미설치"

    df_feat = add_technical_features(df)
    df_feat['Target'] = (df_feat['Close'].shift(-1) > df_feat['Close']).astype(int)
    if len(df_feat) < 30:
        return None, None, "데이터가 부족합니다 (30행 이상 필요)"
    data_for_training = df_feat.iloc[:-1]
    last_row_features = df_feat.iloc[[-1]]
    feature_cols = ['disparity_20', 'bb_position', 'rsi_feat', 'vol_change', 'return_1d', 'return_2d']
    X = data_for_training[feature_cols]
    y = data_for_training['Target']

    model = RandomForestClassifier(n_estimators=120, min_samples_split=8, max_depth=6, random_state=42)
    model.fit(X, y)
    prob_up = model.predict_proba(last_row_features[feature_cols])[0][1]
    importances = dict(zip(feature_cols, model.feature_importances_))
    return prob_up, importances, model, feature_cols, None

@st.cache_resource
def get_cached_model(ticker, period, interval):
    try:
        hist = yf.Ticker(ticker).history(period=period, interval=interval)
        prob_up, importances, model, feature_cols, err = train_prediction_model(hist)
        return prob_up, importances, model, feature_cols, err
    except Exception as e:
        return None, None, None, None, str(e)

def compute_short_term_signal(df):
    if df is None or df.empty or len(df) < 20:
        return None

    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume'] if 'Volume' in df else pd.Series(dtype=float)
    price = close.iloc[-1]

    # RSI(14) 계산
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = (100 - (100 / (1 + rs))).iloc[-1]

    # 볼린저 밴드 하단 터치 여부
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_lower = (sma20 - 2 * std20).iloc[-1] if len(sma20.dropna()) > 0 else None
    hit_bb_lower = bb_lower is not None and price <= bb_lower

    mom_5 = _pct_from_n(close, 5)
    mom_15 = _pct_from_n(close, 15)
    mom_30 = _pct_from_n(close, 30)

    vol_ratio = None
    if len(volume.dropna()) >= 20:
        recent_vol = volume.iloc[-1]
        base_vol = volume.iloc[-20:].median()
        vol_ratio = None if base_vol == 0 else recent_vol / base_vol

    vwap = None
    if len(volume.dropna()) >= 10:
        pv = (close * volume).rolling(30, min_periods=10).sum()
        vwap_den = volume.rolling(30, min_periods=10).sum()
        vwap = pv.iloc[-1] / vwap_den.iloc[-1] if vwap_den.iloc[-1] else None

    tr = (high - low)
    atr14 = tr.rolling(14, min_periods=5).mean().iloc[-1]
    vol_30 = close.pct_change().rolling(30).std().iloc[-1] * 100 if len(close) >= 30 else None

    score = 0.0
    weight = 0.0
    for val, scale, w in [
        (mom_5, 0.4, 0.3),
        (mom_15, 0.8, 0.25),
        (mom_30, 1.2, 0.15),
        ((vol_ratio - 1) * 100 if vol_ratio else None, 20, 0.1),
        ((price - vwap) / vwap * 100 if vwap else None, 0.5, 0.1),
        ((1 - (rsi / 100)) * 100 if rsi is not None and not np.isnan(rsi) else None, 50, 0.08),  # 역추세: RSI 낮을수록 가점
    ]:
        if val is None or np.isnan(val):
            continue
        score += _clamp(val / scale) * w
        weight += w

    # 볼린저 하단 밴드 터치 시 추가 가점
    if hit_bb_lower:
        score += 0.1
        weight += 0.1

    prob_up = (score / weight + 1) / 2 if weight > 0 else 0.5
    target_pct = max(0.15, (vol_30 or 0.3) * 1.2)
    stop_pct = max(0.08, (vol_30 or 0.2) * 0.7)

    return {
        "price": price,
        "prob_up": prob_up,
        "mom_5": mom_5,
        "mom_15": mom_15,
        "mom_30": mom_30,
        "vol_ratio": vol_ratio,
        "vwap": vwap,
        "atr14": atr14,
        "vol_30": vol_30,
        "target_pct": target_pct,
        "stop_pct": stop_pct,
        "recent_high": high.tail(50).max(),
        "recent_low": low.tail(50).min(),
        "rsi": rsi,
        "bb_lower": bb_lower,
        "hit_bb_lower": hit_bb_lower,
    }

# S&P500 주요 티커 (상위 시총 중심)
SP500_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","META","AVGO","LLY","JPM","GOOGL","GOOG",
    "XOM","UNH","JNJ","V","HD","PG","MA","COST","ABBV","BAC",
    "MRK","CVX","ADBE","WMT","PEP","KO","ORCL","NFLX","CRM","AMD",
    "INTC","CSCO","TXN","LIN","MCD","DIS","TMO","ABT","AMGN","HON",
    "PM","ACN","CAT","IBM","GE","LOW","UPS","SPGI","VRTX","BKNG"
]

@st.cache_data(ttl=1800)
def get_sp500_short_signals(limit=30, interval="1d", period="1mo"):
    tickers = SP500_TICKERS[:limit]
    rows = []
    for t in tickers:
        try:
            df = yf.Ticker(t).history(period=period, interval=interval)
            sig = compute_short_term_signal(df)
            if sig:
                rows.append({
                    "종목": t,
                    "상승확률(%)": round(sig["prob_up"] * 100, 2),
                    "현재가": round(sig["price"], 2),
                    "타겟(%)": round(sig["target_pct"], 2),
                    "손절(%)": round(sig["stop_pct"], 2),
                    "모멘텀5(%)": None if sig["mom_5"] is None else round(sig["mom_5"], 2),
                    "거래량배율(x)": None if sig["vol_ratio"] is None else round(sig["vol_ratio"], 2),
                })
        except Exception:
            continue
    rows = sorted(rows, key=lambda x: x["상승확률(%)"], reverse=True)
    return pd.DataFrame(rows)

# ---------------------------------------------------------
# 2. 핵심 로직 (백테스팅)
# ---------------------------------------------------------
def run_backtest(df, initial_cash, mode, target_weight, trigger_up, sell_pct, trigger_down, buy_pct, base_mode="TOTAL", use_ai=False, ai_threshold=0.6, ai_period="6mo", ai_interval="1d"):
    # base_mode: 매매 비율 산정 기준 (TOTAL: 현금+주식, STOCK: 주식가치)
    # use_ai: True면 AI 상승확률(ai_threshold 이상)일 때만 매수
    cash = initial_cash
    start_price = df.iloc[0]['Close']
    initial_invest = (initial_cash * (target_weight / 100))
    shares = initial_invest / start_price 
    cash -= shares * start_price
    last_rebal_price = start_price 
    history = []; trade_log = []
    for date, row in df.iterrows():
        price = row['Close']
        stock_val = shares * price
        total_val = cash + stock_val
        action_taken = False; should_sell = False
        # 상승시 매도 (가격 밴드 기준)
        if shares > 0 and price >= last_rebal_price * (1 + trigger_up/100):
            should_sell = True
        if should_sell:
            base_val = total_val if base_mode == "TOTAL" else stock_val
            target_sell_val = base_val * (sell_pct / 100)
            sell_qty = min(stock_val, target_sell_val) / price if price > 0 else 0
            if sell_qty > 0:
                shares -= sell_qty; cash += sell_qty * price
                pct_diff = (price - last_rebal_price)/last_rebal_price*100
                trade_log.append({"date": date, "type": "🔴 매도", "price": price, "qty": sell_qty, "cause": f"상승 밴드 (+{pct_diff:.1f}%)"})
                last_rebal_price = price; action_taken = True
        if not action_taken:
            should_buy = False
            if price <= last_rebal_price * (1 - trigger_down/100) or (shares == 0 and cash > price):
                should_buy = True
            # AI 필터 적용
            if use_ai and should_buy:
                prob_ai, _, _, _, err = get_cached_model(str(df.name) if hasattr(df, "name") else "", ai_period, ai_interval)
                if err or prob_ai is None or prob_ai < ai_threshold:
                    should_buy = False
            if should_buy:
                base_val = total_val if base_mode == "TOTAL" else stock_val
                invest_amt = base_val * (buy_pct / 100)
                invest_amt = min(invest_amt, cash)
                buy_qty = invest_amt / price 
                if buy_qty > 0:
                    shares += buy_qty; cash -= buy_qty * price
                    pct_diff = (price - last_rebal_price)/last_rebal_price*100
                    trade_log.append({"date": date, "type": "🔵 매수", "price": price, "qty": buy_qty, "cause": f"하락 밴드 ({pct_diff:.1f}%)"})
                    last_rebal_price = price
        history.append(cash + (shares * price))
    df['Strategy_Asset'] = history
    final_return = ((history[-1] - initial_cash) / initial_cash) * 100
    buy_hold_return = ((df.iloc[-1]['Close'] - df.iloc[0]['Close']) / df.iloc[0]['Close']) * 100
    return df, trade_log, final_return, buy_hold_return

def optimize_params(df, fixed_b, fixed_d, target_w):
    if len(df) < 10: st.toast("❌ 데이터 부족"); return
    best_ret = -9999
    best_params = (st.session_state.get('mode', 'VALUE'), st.session_state.get('up_a', 10.0), st.session_state.get('down_c', 10.0))
    search_ranges = [3.0, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0]
    st.toast("🤖 시뮬레이션 중...")
    for a_val in search_ranges:
        for c_val in search_ranges:
            _, _, ret, _ = run_backtest(df.copy(), 10000, 'VALUE', target_w, a_val, fixed_b, c_val, fixed_d, st.session_state.get('base_mode', 'TOTAL'))
            if ret > best_ret: best_ret = ret; best_params = ('VALUE', a_val, c_val)
    st.session_state['mode'] = best_params[0]; st.session_state['up_a'] = best_params[1]; st.session_state['down_c'] = best_params[2]
    st.toast(f"✅ 최적 전략: {best_params[0]} / +{best_params[1]}% / -{best_params[2]}%")

def walk_forward_analysis(df, initial_cash, train_days=252, test_days=63):
    df = df.sort_index()
    results = []
    start = 0
    while True:
        train_end = start + train_days
        test_end = train_end + test_days
        if test_end > len(df):
            break
        train_df = df.iloc[start:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()
        if len(train_df) < train_days * 0.5 or len(test_df) < test_days * 0.5:
            break
        _, _, ret_train, bh_train = run_backtest(
            train_df.copy(), initial_cash, st.session_state['mode'], st.session_state['target_w'],
            st.session_state['up_a'], st.session_state['sell_b'], st.session_state['down_c'], st.session_state['buy_d'],
            st.session_state.get('base_mode', 'TOTAL'),
            st.session_state.get('use_ai_filter', False),
            st.session_state.get('ai_threshold', 0.6),
            st.session_state.get('ai_period', '6mo'),
            st.session_state.get('ai_interval', '1d'),
        )
        _, _, ret_test, bh_test = run_backtest(
            test_df.copy(), initial_cash, st.session_state['mode'], st.session_state['target_w'],
            st.session_state['up_a'], st.session_state['sell_b'], st.session_state['down_c'], st.session_state['buy_d'],
            st.session_state.get('base_mode', 'TOTAL'),
            st.session_state.get('use_ai_filter', False),
            st.session_state.get('ai_threshold', 0.6),
            st.session_state.get('ai_period', '6mo'),
            st.session_state.get('ai_interval', '1d'),
        )
        results.append({
            "훈련기간": f"{train_df.index[0].date()} ~ {train_df.index[-1].date()}",
            "검증기간": f"{test_df.index[0].date()} ~ {test_df.index[-1].date()}",
            "훈련수익률(%)": round(ret_train, 2),
            "검증수익률(%)": round(ret_test, 2),
            "훈련_단순보유(%)": round(bh_train, 2),
            "검증_단순보유(%)": round(bh_test, 2),
        })
        start += test_days
    return pd.DataFrame(results)

# ---------------------------------------------------------
# 3. UI 레이아웃
# ---------------------------------------------------------
col_main, col_side = st.columns([3, 1])

# --- [우측 패널] ---
with col_side:
    my_stocks, my_cash = get_portfolio()
    current_cash = my_cash.iloc[0]['amount'] if not my_cash.empty else 0.0
    
    total_stock_val = 0.0; daily_pnl_sum = 0.0; total_invested = 0.0
    stock_display_list = []
    
    if not my_stocks.empty:
        for index, row in my_stocks.iterrows():
            ticker = row['ticker']; shares = row['shares']; avg_price = row['avg_price']
            try:
                stock_data = yf.Ticker(ticker).history(period="5d")
                if len(stock_data) >= 2:
                    cur_price = stock_data['Close'].iloc[-1]; prev_close = stock_data['Close'].iloc[-2]
                    val = cur_price * shares
                    invested = avg_price * shares
                    total_stock_val += val; total_invested += invested
                    daily_pnl_sum += (cur_price - prev_close) * shares
                    profit_pct = (cur_price - avg_price) / avg_price * 100 if avg_price > 0 else 0.0
                    stock_display_list.append({'ticker': ticker, 'shares': shares, 'val': val, 'profit_pct': profit_pct})
            except: pass

    # 총 손익 계산
    total_pnl_val = total_stock_val - total_invested
    total_pnl_pct = (total_pnl_val / total_invested * 100) if total_invested > 0 else 0.0
    pnl_color = "red" if total_pnl_val >= 0 else "blue"
    pnl_icon = "🔺" if total_pnl_val >= 0 else "▼"
    
    st.markdown(f"<h3 style='display:inline;'>내 투자</h3><span style='color:{pnl_color}; font-size:1rem; margin-left:10px;'>{pnl_icon} {total_pnl_pct:.2f}% (${total_pnl_val:,.2f})</span>", unsafe_allow_html=True)
    
    for item in stock_display_list:
        with st.container(border=True):
            c1, c2 = st.columns([1.2, 1])
            if c1.button(f"{item['ticker']}", key=f"btn_{item['ticker']}", use_container_width=True, on_click=set_ticker, args=(item['ticker'],)): pass
            c1.caption(f"{item['shares']:g}주")
            color = "red" if item['profit_pct'] > 0 else "blue"
            c2.markdown(f"${item['val']:,.0f}")
            c2.markdown(f":{color}[{item['profit_pct']:.1f}%]")

    total_value = total_stock_val + current_cash
    st.metric(label="총 자산 (USD)", value=f"${total_value:,.2f}", delta=f"${daily_pnl_sum:,.2f} (오늘)")
    st.caption(f"📊 주식 ${total_stock_val:,.2f} + 💵 현금 ${current_cash:,.2f}")
    # VIX
    try:
        vix = yf.Ticker("^VIX").history(period="5d")['Close']
        if len(vix) >= 2:
            vix_now = vix.iloc[-1]; vix_prev = vix.iloc[-2]
            vix_delta = vix_now - vix_prev
            st.metric("VIX(공포지수)", f"{vix_now:.2f}", delta=f"{vix_delta:+.2f}")
            if vix_now >= 30:
                st.warning("VIX>30: 변동성 고조, 현금 비중 확대/분할매수만 고려")
            elif vix_now <= 15:
                st.info("VIX<15: 변동성 낮음, 과열 여부 확인 후 비중 조절")
    except Exception:
        pass
    
    sim_col1, sim_col2, sim_col3 = st.columns([1, 1, 1])
    sim_period = sim_col1.selectbox("시뮬레이션 기간", ["6mo", "1y", "2y", "5y"], index=1)
    view_mode = sim_col2.selectbox("표시 방식", ["자산(USD)", "정규화(=100)", "수익률(%)"], index=0)
    bench_ticker = sim_col3.selectbox("벤치마크", ["SPY", "QQQ", "VT", "IWV", "없음"], index=0)

    if st.button("📈 자산 추이 (Simulation)", use_container_width=True):
        if not my_stocks.empty:
            with st.spinner("계산 중..."):
                tickers = my_stocks['ticker'].tolist()
                data = yf.download(tickers, period=sim_period)['Close']
                if isinstance(data, pd.Series):
                    data = data.to_frame(name=tickers[0])
                data = data.dropna(axis=0, how='all')
                if data.empty:
                    st.warning("가격 데이터를 불러오지 못했습니다.")
                else:
                    port_index = data.index
                    if hasattr(port_index, "tz"): port_index = port_index.tz_localize(None)
                    portfolio_hist = pd.Series(current_cash, index=port_index)
                    latest_prices = {}
                    for _, row in my_stocks.iterrows():
                        t = row['ticker']
                        if t in data.columns:
                            series_t = data[t]
                            if hasattr(series_t.index, "tz"): series_t = series_t.tz_localize(None)
                            series_t = series_t.reindex(portfolio_hist.index).ffill()
                            portfolio_hist += series_t * row['shares']
                            latest_prices[t] = series_t.iloc[-1]

                    bench_series = None
                    if bench_ticker != "없음":
                        try:
                            bench_data = yf.download(bench_ticker, period=sim_period)['Close']
                            if hasattr(bench_data.index, "tz"): bench_data = bench_data.tz_localize(None)
                            if isinstance(bench_data, pd.DataFrame):
                                bench_series = bench_data.iloc[:, 0]
                            else:
                                bench_series = bench_data
                        except Exception:
                            bench_series = None

                    if len(portfolio_hist) == 0:
                        st.warning("계산할 자산 곡선이 없습니다.")
                    else:
                        bench_plot = None
                        bench_series_aligned = None
                        if bench_series is not None and len(bench_series) > 0:
                            bench_series_aligned = bench_series.reindex(portfolio_hist.index).ffill()
                            if bench_series_aligned.dropna().empty:
                                bench_series_aligned = None

                        base_port = portfolio_hist.iloc[0] if portfolio_hist.iloc[0] != 0 else 1
                        if view_mode == "정규화(=100)":
                            plot_series = portfolio_hist / base_port * 100
                            y_label = "지수화(=100)"
                            if bench_series_aligned is not None:
                                bench_start = bench_series_aligned.iloc[0]
                                if isinstance(bench_start, pd.Series):
                                    bench_start = bench_start.iloc[0]
                                bench_base = bench_start if bench_start != 0 else 1
                                bench_plot = bench_series_aligned / bench_base * 100
                        elif view_mode == "수익률(%)":
                            plot_series = (portfolio_hist / base_port - 1) * 100
                            y_label = "수익률(%)"
                            if bench_series_aligned is not None:
                                bench_start = bench_series_aligned.iloc[0]
                                if isinstance(bench_start, pd.Series):
                                    bench_start = bench_start.iloc[0]
                                bench_base = bench_start if bench_start != 0 else 1
                                bench_plot = (bench_series_aligned / bench_base - 1) * 100
                        else:
                            plot_series = portfolio_hist
                            y_label = "총 자산 (USD)"
                            if bench_series_aligned is not None:
                                bench_start = bench_series_aligned.iloc[0]
                                if isinstance(bench_start, pd.Series):
                                    bench_start = bench_start.iloc[0]
                                bench_base = bench_start if bench_start != 0 else 1
                                bench_plot = bench_series_aligned / bench_base * plot_series.iloc[0]

                        roll_max = portfolio_hist.cummax()
                        drawdown = (portfolio_hist / roll_max - 1) * 100

                        fig_total = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
                        fig_total.add_trace(go.Scatter(x=plot_series.index, y=plot_series, fill='tozeroy', line=dict(color='#8b5cf6', width=2), name=y_label), row=1, col=1)
                        if bench_plot is not None:
                            fig_total.add_trace(go.Scatter(x=bench_plot.index, y=bench_plot, line=dict(color='#f59e0b', width=2), name=f"{bench_ticker}"), row=1, col=1)
                        fig_total.add_trace(go.Scatter(x=drawdown.index, y=drawdown, fill='tozeroy', line=dict(color='#ef4444'), name='드로우다운(%)'), row=2, col=1)
                        fig_total.update_layout(margin=dict(t=20, b=10, l=10, r=10), height=480, showlegend=True, legend=dict(orientation="h", y=1.08))
                        fig_total.update_yaxes(title_text=y_label, row=1, col=1)
                        fig_total.update_yaxes(title_text="드로우다운(%)", row=2, col=1)
                        st.plotly_chart(fig_total, use_container_width=True)

                    if latest_prices:
                        total_port = portfolio_hist.iloc[-1]
                        n = len(latest_prices)
                        target_eq = 100 / n if n > 0 else 0
                        guide_rows = []
                        for _, row in my_stocks.iterrows():
                            t = row['ticker']
                            if t not in latest_prices:
                                continue
                            price_now = latest_prices[t]
                            val_now = price_now * row['shares']
                            weight = val_now / total_port * 100 if total_port else 0
                            drift = weight - target_eq
                            diff_val = (target_eq/100 * total_port) - val_now
                            sug_shares = diff_val / price_now if price_now else 0
                            # 비중이 목표보다 높으면 매도, 낮으면 매수, 근접하면 유지
                            action = "매도" if drift > 2 else "매수" if drift < -2 else "유지"
                            guide_rows.append({
                                "종목": t,
                                "현재가": price_now,
                                "보유가치": val_now,
                                "비중(%)": round(weight, 2),
                                "목표=균등(%)": round(target_eq, 2),
                                "괴리(%)": round(drift, 2),
                                "제안 수량": round(sug_shares, 4),
                                "가이드": action,
                            })
                        if guide_rows:
                            st.markdown("#### 매매 가이드")
                            st.dataframe(pd.DataFrame(guide_rows), use_container_width=True)

    st.divider()
    
    tab_edit1, tab_edit2, tab_edit3, tab_log = st.tabs(["💵 현금", "⚖️ 매매", "🛠️ 관리", "📝 일지"])
    
    with tab_edit1:
        new_cash = st.number_input("보유 현금 ($)", value=float(current_cash), step=100.0)
        if st.button("현금 업데이트"): update_cash(new_cash); st.rerun()
            
    with tab_edit2:
        st.caption("티커 입력 후 매수/매도 선택")
        input_ticker = st.text_input("티커 (예: TQQQ)").upper()
        c_sh, c_pr = st.columns(2)
        input_shares = c_sh.number_input("수량", min_value=0.000001, step=0.01, format="%.6f")
        input_avg = c_pr.number_input("단가 ($)", min_value=0.0)
        is_overwrite = st.checkbox("단순 정보 수정 (덮어쓰기)")
        col_buy, col_sell = st.columns(2)
        if col_buy.button("🔵 매수", use_container_width=True):
            if input_ticker:
                if is_overwrite: overwrite_stock(input_ticker, input_shares, input_avg)
                else: add_stock(input_ticker, input_shares, input_avg)
                st.rerun()
            else: st.toast("티커 입력 필요")
        if col_sell.button("🔴 매도", use_container_width=True):
            if input_ticker:
                if is_overwrite: st.error("수정 모드 매도 불가")
                else: sell_stock(input_ticker, input_shares, input_avg); st.rerun()
            else: st.toast("티커 입력 필요")

    # [신규] 백업/복구 기능이 추가된 관리 탭
    with tab_edit3:
        st.write("**데이터 백업/복구**")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            csv_data = convert_df_to_csv()
            st.download_button(
                label="💾 백업 (다운로드)",
                data=csv_data,
                file_name="portfolio_backup.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_b2:
            uploaded_file = st.file_uploader("📂 복구 (업로드)", type=['csv'], label_visibility="collapsed")
            if uploaded_file is not None:
                if st.button("복구 시작", type="primary", use_container_width=True):
                    restore_from_csv(uploaded_file)

        st.divider()
        
        if not my_stocks.empty:
            st.write("**종목 순서 / 삭제**")
            edited_df = st.data_editor(
                my_stocks[['ticker', 'sort_order']], 
                column_config={"ticker": st.column_config.TextColumn("종목", disabled=True), "sort_order": st.column_config.NumberColumn("순서", min_value=1, step=1)},
                num_rows="dynamic", hide_index=True, use_container_width=True
            )
            if st.button("변경사항 저장"): update_sort_orders(edited_df); st.rerun()
        else: st.info("보유 종목 없음")

    with tab_log:
        logs = get_logs()
        if not logs.empty:
            total_pnl = logs['realized_pnl'].sum()
            total_trades = len(logs[logs['action'].str.contains('매도')])
            win_trades = len(logs[logs['realized_pnl'] > 0])
            win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0.0
            k1, k2, k3 = st.columns(3)
            k1.metric("누적 실현 손익", f"${total_pnl:,.2f}", delta=f"{total_pnl:,.2f}")
            k2.metric("총 매도 횟수", f"{total_trades}회")
            k3.metric("매도 승률", f"{win_rate:.1f}%")
            st.divider()
            log_to_del = st.selectbox("삭제할 로그", options=logs['id'], format_func=lambda x: f"#{x}: {logs[logs['id']==x].iloc[0]['action']} ({logs[logs['id']==x].iloc[0]['ticker']})")
            if st.button("선택 로그 삭제"): delete_log(log_to_del); st.rerun()
            st.dataframe(logs[['date', 'ticker', 'action', 'shares', 'price', 'realized_pnl', 'note']], column_config={"price": st.column_config.NumberColumn("단가", format="$%.2f"), "realized_pnl": st.column_config.NumberColumn("손익($)", format="$%.2f"), "shares": st.column_config.NumberColumn("수량", format="%g")}, hide_index=True, use_container_width=True)
        else: st.info("기록 없음")

# --- [좌측 패널] 메인 차트 ---
with col_main:
    c_search, c_int, c_refresh = st.columns([2, 1, 0.5])
    with c_search: search_ticker = st.text_input("종목 검색", key='search_ticker').upper()
    with c_int:
        interval_map = {'1m': '1분', '5m': '5분', '1d': '일봉', '1wk': '주봉', '1mo': '월봉'}
        sel_interval = st.selectbox("주기", options=list(interval_map.keys()), format_func=lambda x: interval_map[x], index=2)
    with c_refresh:
        st.write(""); st.write("")
        if st.button("🔄"): st.rerun()

    stock = yf.Ticker(search_ticker)
    period_map = {'1m': '5d', '5m': '1mo', '1d': '2y', '1wk': '5y', '1mo': '10y'}
    try: hist_chart = stock.history(period=period_map[sel_interval], interval=sel_interval)
    except: hist_chart = pd.DataFrame()
    
    if hist_chart.empty: st.error("데이터 없음")
    else:
        last_price = hist_chart['Close'].iloc[-1]
        change = last_price - hist_chart['Close'].iloc[-2]
        pct_change = (change / hist_chart['Close'].iloc[-2]) * 100
        st.markdown(f"## {search_ticker} ${last_price:.2f} <span style='color:{'red' if change>0 else 'blue'}'>({pct_change:.2f}%)</span>", unsafe_allow_html=True)

        if sel_interval in ['1m', '5m']:
            cols = st.columns([1,1,1,1,6])
            if cols[0].button("1H"): st.session_state['chart_zoom'] = 1
            if cols[1].button("2H"): st.session_state['chart_zoom'] = 2
            if cols[2].button("4H"): st.session_state['chart_zoom'] = 4
            if cols[3].button("ALL"): st.session_state['chart_zoom'] = 0
            if 'chart_zoom' not in st.session_state: st.session_state['chart_zoom'] = 4

        fig = go.Figure(data=[go.Candlestick(x=hist_chart.index, open=hist_chart['Open'], high=hist_chart['High'], low=hist_chart['Low'], close=hist_chart['Close'])])
        if sel_interval in ['1m', '5m'] and st.session_state.get('chart_zoom', 0) > 0:
            fig.update_xaxes(range=[hist_chart.index[-1] - timedelta(hours=st.session_state['chart_zoom']), hist_chart.index[-1]])
        fig.update_layout(xaxis_rangeslider_visible=False, height=400, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        tab1, tab2, tab3, tab4 = st.tabs(["🔄 전략 시뮬레이터", "📢 매매 신호", "📈 추세 예측", "📊 S&P500 랭킹"])
        
        with tab1:
            st.markdown("### 🛠️ 과거 데이터 검증")
            c1, c2 = st.columns(2)
            start_date = c1.date_input("시작일", value=datetime.now() - timedelta(days=365))
            end_date = c2.date_input("종료일", value=datetime.now())
            hist_back = stock.history(start=start_date, end=end_date, interval="1d")
            st.divider()
            ci, cr = st.columns([1, 2])
            with ci:
                if 'mode' not in st.session_state:
                    st.session_state.update({
                        'mode':'VALUE','target_w':50,'up_a':10.0,'sell_b':50,'down_c':10.0,'buy_d':50,
                        'base_mode':'TOTAL','use_ai_filter': False,'ai_threshold':0.6,'ai_threshold_pct':60.0,'ai_period':'6mo','ai_interval':'1d'
                    })
                with st.container(border=True):
                    st.radio("매매 비율 기준", ['TOTAL','STOCK'], key='base_mode', format_func=lambda x: "현금+주식" if x=="TOTAL" else "주식 평가액")
                    st.slider("초기 투자 비중(%)", 10, 90, key='target_w', step=10)
                    st.markdown("**매도**")
                    st.slider("상승폭 트리거(%)", 1.0, 30.0, key='up_a', step=0.5)
                    st.slider("매도 비율(%)", 10, 100, key='sell_b', step=10)
                    st.markdown("**매수**")
                    st.slider("하락폭 트리거(%)", 1.0, 30.0, key='down_c', step=0.5)
                    st.slider("매수 비율(%)", 10, 100, key='buy_d', step=10)
                    st.markdown("**AI 필터**")
                    st.checkbox("AI 상승확률 필터 사용", key='use_ai_filter')
                    if 'ai_threshold_pct' not in st.session_state: st.session_state['ai_threshold_pct'] = 60.0
                    ai_pct_val = st.slider("AI 임계값(%)", 50.0, 80.0, key='ai_threshold_pct', value=st.session_state.get('ai_threshold_pct', 60.0), step=1.0)
                    st.session_state['ai_threshold'] = ai_pct_val / 100.0
                    st.selectbox("AI 학습 기간", ["3mo","6mo","1y"], key='ai_period')
                    st.selectbox("AI 데이터 인터벌", ["1d","1h"], key='ai_interval')
                st.button("✨ 최적 파라미터", on_click=optimize_params, args=(hist_back, st.session_state['sell_b'], st.session_state['buy_d'], st.session_state['target_w']))
            with cr:
                if len(hist_back) > 0:
                    df_r, logs_sim, ret, b_ret = run_backtest(
                        hist_back.copy(), 10000, st.session_state['mode'], st.session_state['target_w'],
                        st.session_state['up_a'], st.session_state['sell_b'], st.session_state['down_c'], st.session_state['buy_d'],
                        st.session_state.get('base_mode','TOTAL'),
                        st.session_state.get('use_ai_filter', False),
                        st.session_state.get('ai_threshold', 0.6),
                        st.session_state.get('ai_period', '6mo'),
                        st.session_state.get('ai_interval', '1d'),
                    )
                    c1, c2, c3 = st.columns(3)
                    c1.metric("전략 수익", f"{ret:.2f}%", delta=f"{ret-b_ret:.2f}%p")
                    c2.metric("단순 보유", f"{b_ret:.2f}%")
                    c3.metric("매매 횟수", f"{len(logs_sim)}회")
                    f_b = go.Figure()
                    f_b.add_trace(go.Scatter(x=df_r.index, y=df_r['Strategy_Asset'], name='전략', line=dict(color='#ef4444', width=2)))
                    f_b.add_trace(go.Scatter(x=df_r.index, y=df_r['Close']*(10000/df_r['Close'].iloc[0]), name='보유', line=dict(color='#e5e7eb', dash='dot')))
                    b_p = df_r.loc[[x['date'] for x in logs_sim if '매수' in x['type']]]
                    s_p = df_r.loc[[x['date'] for x in logs_sim if '매도' in x['type']]]
                    f_b.add_trace(go.Scatter(x=b_p.index, y=b_p['Strategy_Asset'], mode='markers', name='매수', marker=dict(color='blue', symbol='triangle-up', size=8)))
                    f_b.add_trace(go.Scatter(x=s_p.index, y=s_p['Strategy_Asset'], mode='markers', name='매도', marker=dict(color='red', symbol='triangle-down', size=8)))
                    f_b.update_layout(margin=dict(t=30, b=0, l=0, r=0), legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
                    st.plotly_chart(f_b, use_container_width=True)
                    with st.expander("기록"): st.dataframe(pd.DataFrame(logs_sim), use_container_width=True)

                    with st.expander("전진 분석 (Walk-forward)"):
                        wf_col1, wf_col2, wf_col3 = st.columns([1,1,1])
                        train_days = wf_col1.slider("훈련 기간(일)", 60, 400, 252, step=30)
                        test_days = wf_col2.slider("검증 기간(일)", 20, 180, 63, step=7)
                        initial_cash = wf_col3.number_input("초기자본", value=10000, step=1000)
                        if st.button("전진 분석 실행"):
                            if len(hist_back) < train_days + test_days:
                                st.warning("데이터가 부족합니다. 기간을 줄이세요.")
                            else:
                                with st.spinner("전진 분석 중..."):
                                    wf_df = walk_forward_analysis(hist_back.copy(), initial_cash, train_days=train_days, test_days=test_days)
                                if wf_df.empty:
                                    st.warning("분석할 윈도우가 없습니다. 기간을 조정하세요.")
                                else:
                                    st.dataframe(wf_df, use_container_width=True)
                                    c_w1, c_w2, c_w3, c_w4 = st.columns(4)
                                    c_w1.metric("평균 훈련 수익률", f"{wf_df['훈련수익률(%)'].mean():.2f}%")
                                    c_w2.metric("평균 검증 수익률", f"{wf_df['검증수익률(%)'].mean():.2f}%")
                                    c_w3.metric("훈련 BH", f"{wf_df['훈련_단순보유(%)'].mean():.2f}%")
                                    c_w4.metric("검증 BH", f"{wf_df['검증_단순보유(%)'].mean():.2f}%")

        with tab2:
            min_len = 60
            if len(hist_chart) >= min_len:
                close = hist_chart['Close'].dropna()
                price = close.iloc[-1]

                d = close.diff()
                gain = d.where(d > 0, 0).rolling(14).mean()
                loss = -d.where(d < 0, 0).rolling(14).mean()
                rs = gain / loss.replace(0, pd.NA)
                rsi = (100 - (100 / (1 + rs))).iloc[-1]

                sma20 = close.rolling(20).mean()
                sma60 = close.rolling(60).mean()
                ma20 = sma20.iloc[-1]; ma60 = sma60.iloc[-1]

                std20 = close.rolling(20).std()
                bb_upper = (sma20 + 2 * std20).iloc[-1]
                bb_lower = (sma20 - 2 * std20).iloc[-1]

                macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
                macd_signal = macd_line.ewm(span=9, adjust=False).mean()
                macd_hist = macd_line - macd_signal
                macd_val = macd_line.iloc[-1]
                macd_sig = macd_signal.iloc[-1]
                macd_h = macd_hist.iloc[-1]

                m1, m2, m3 = st.columns(3)
                m1.metric("RSI(14)", f"{rsi:.1f}", delta="과매도" if rsi < 30 else "과매수" if rsi > 70 else "중립")
                m2.metric("20/60 SMA", f"{ma20:.2f}", delta=f"P:{price:.2f} | 60:{ma60:.2f}")
                m3.metric("MACD", f"{macd_val:.2f}", delta=f"Signal {macd_sig:.2f} / Hist {macd_h:.2f}")

                def render_signal(title, status, note, color="gray"):
                    with st.container(border=True):
                        st.markdown(f"**{title}**")
                        st.markdown(f":{color}[{status}]")
                        st.caption(note)

                col_a, col_b = st.columns(2)
                with col_a:
                    if price > ma20 and ma20 > ma60:
                        render_signal("추세", "상승 추세", "20/60 상향 정렬. 비중 유지, 과열 시 분할매도 검토.", "red")
                    elif price > ma60:
                        render_signal("추세", "단기 반등", "60일선 돌파 확인 전까지 관망/소량 매수.", "gray")
                    else:
                        render_signal("추세", "중기 약세", "60일선 아래. 분할매수만, 추세 전환 대기.", "blue")

                    if price <= bb_lower:
                        render_signal("볼린저 밴드", "하단 이탈 (매수 관심)", "과매도 구간. 분할매수/저가 매수 검토.", "red")
                    elif price >= bb_upper:
                        render_signal("볼린저 밴드", "상단 이탈 (매도 경계)", "과매수/과열 구간. 익절·분할매도 검토.", "blue")
                    else:
                        render_signal("볼린저 밴드", "밴드 중앙", "중립. 추세 지표와 함께 방향 확인.", "gray")

                with col_b:
                    if rsi < 30:
                        render_signal("RSI(14)", "과매도", "모멘텀 과도 하락. 분할매수 우위.", "red")
                    elif rsi > 70:
                        render_signal("RSI(14)", "과매수", "단기 과열. 분할매도/비중 축소 검토.", "blue")
                    else:
                        render_signal("RSI(14)", "중립", "30~70 사이. 추세/밴드와 함께 판단.", "gray")

                    if macd_val > macd_sig and macd_h > 0:
                        render_signal("MACD", "상승 모멘텀", "MACD>Signal, 히스토그램 양수. 추세 우상향.", "red")
                    elif macd_val < macd_sig and macd_h < 0:
                        render_signal("MACD", "하락 모멘텀", "MACD<Signal, 히스토그램 음수. 약세 모멘텀.", "blue")
                    else:
                        render_signal("MACD", "혼조", "교차 대기 구간. 추가 확인 필요.", "gray")
            else: st.warning(f"데이터가 부족합니다 (최소 {min_len}봉 필요)")
        
        with tab3:
            intr_interval = '1m' if sel_interval in ['1m', '5m'] else '5m'
            intr_period = '5d' if intr_interval == '1m' else '1mo'
            st.caption(f"{intr_interval} 데이터 기반 초단기 신호 (타임프레임: {intr_period})")
            try:
                short_df = stock.history(period=intr_period, interval=intr_interval)
            except Exception:
                short_df = pd.DataFrame()

            if short_df.empty or len(short_df) < 20:
                st.warning("단타 신호를 계산할 데이터가 부족합니다.")
            else:
                sig = compute_short_term_signal(short_df)
                if not sig:
                    st.warning("신호 계산에 실패했습니다.")
                else:
                    prob = sig['prob_up'] * 100
                    tgt_px = sig['price'] * (1 + sig['target_pct'] / 100)
                    stop_px = sig['price'] * (1 - sig['stop_pct'] / 100)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("상승 확률", f"{prob:.1f}%", delta=">60%에서만 진입 권장")
                    c2.metric("타겟/손절", f"+{sig['target_pct']:.2f}% / -{sig['stop_pct']:.2f}%", delta=f"${tgt_px:.2f} / ${stop_px:.2f}")
                    c3.metric("단기 변동성", f"{(sig['vol_30'] or 0):.2f}%", delta=f"ATR {sig['atr14']:.2f}")

                    st.progress(prob / 100, text="매수 모멘텀" if prob >= 50 else "중립/매도 대기")
                    st.markdown(
                        f"- 현재가: ${sig['price']:.2f} | VWAP: {('N/A' if sig['vwap'] is None else f'${sig['vwap']:.2f}')}"
                        f"\n- 최근 고점/저점(50캔들): ${sig['recent_high']:.2f} / ${sig['recent_low']:.2f}"
                        f"\n- 거래량 스파이크: {('N/A' if sig['vol_ratio'] is None else f'{sig['vol_ratio']:.2f}x')}"
                    )

                    feat_rows = [
                        {"지표": "모멘텀 5", "값(%)": None if sig['mom_5'] is None else round(sig['mom_5'], 2)},
                        {"지표": "모멘텀 15", "값(%)": None if sig['mom_15'] is None else round(sig['mom_15'], 2)},
                        {"지표": "모멘텀 30", "값(%)": None if sig['mom_30'] is None else round(sig['mom_30'], 2)},
                        {"지표": "VWAP 괴리", "값(%)": None if sig['vwap'] is None else round((sig['price'] - sig['vwap']) / sig['vwap'] * 100, 3)},
                        {"지표": "거래량 배율", "값(%)": None if sig['vol_ratio'] is None else round(sig['vol_ratio'] * 100 - 100, 1)},
                        {"지표": "ATR(14)", "값(%)": None if sig['atr14'] is None else round(sig['atr14'], 4)},
                    ]
                    with st.expander("지표 세부값"):
                        st.dataframe(pd.DataFrame(feat_rows), use_container_width=True)
                    st.info("진입 예시: 상승 확률>60% && 스프레드/유동성 조건 만족 시 분할 진입, 손절은 -stop% 또는 직전 저점 아래에 위치.")

            st.markdown("### 🤖 AI 예측 모델 (Random Forest)")
            if len(hist_chart) >= 60:
                ai_col1, ai_col2 = st.columns([1,1])
                ai_period = ai_col1.selectbox("AI 학습 기간", ["3mo","6mo","1y"], key="ai_period_live", index=1)
                ai_interval = ai_col2.selectbox("AI 인터벌", ["1d","1h"], key="ai_interval_live", index=0)

                if st.button("AI 예측 실행"):
                    with st.spinner("AI 학습/예측 중..."):
                        prob_ai, importances, model, feature_cols, err = get_cached_model(search_ticker, ai_period, ai_interval)
                    if err:
                        st.error(err)
                    else:
                        ai_prob_pct = prob_ai * 100
                        col_ai1, col_ai2 = st.columns([1, 2])
                        col_ai1.metric("AI 예상 상승 확률", f"{ai_prob_pct:.1f}%", delta=f"{ai_prob_pct - 50:.1f}%p vs 50")
                        with col_ai2:
                            if prob_ai > 0.6:
                                st.success(f"AI 의견: 매수 우위 ({ai_prob_pct:.1f}%)")
                                st.progress(min(max(prob_ai, 0), 1), text="상승 확신도")
                            elif prob_ai < 0.4:
                                st.error(f"AI 의견: 매도/관망 우위 ({ai_prob_pct:.1f}%)")
                                st.progress(min(max(prob_ai, 0), 1), text="상승 확신도")
                            else:
                                st.warning(f"AI 의견: 중립 ({ai_prob_pct:.1f}%)")
                                st.progress(min(max(prob_ai, 0), 1), text="상승 확신도")
                        if importances:
                            with st.expander("AI 중요 변수"):
                                imp_df = pd.DataFrame({"Feature": list(importances.keys()), "Importance": list(importances.values())}).sort_values("Importance", ascending=False)
                                st.bar_chart(imp_df.set_index("Feature"))
            else:
                st.warning("AI 학습을 위해 최소 60개 이상의 데이터가 필요합니다.")

        with tab4:
            st.markdown("### S&P500 단기 신호 랭킹")
            col_a, col_b, col_c = st.columns([1,1,1])
            limit = col_a.slider("검사 종목 수", min_value=10, max_value=len(SP500_TICKERS), value=30, step=10)
            interval_choice = col_c.selectbox("인터벌", ["1d", "1h", "5m"], index=0, help="5m는 1개월 이내, 1h는 3개월 이내 데이터만 지원됩니다.")

            period_map = {
                "1d": ["1mo", "3mo", "6mo", "1y"],
                "1h": ["5d", "1mo", "3mo"],
                "5m": ["5d", "1mo"],
            }
            period_options = period_map.get(interval_choice, ["1mo"])
            period_choice = col_b.selectbox("기간", period_options, index=0)

            if st.button("상승 확률 상위 종목 보기", type="primary"):
                with st.spinner("신호 계산 중..."):
                    df_rank = get_sp500_short_signals(limit=limit, interval=interval_choice, period=period_choice)
                    if df_rank.empty:
                        st.warning("신호를 계산할 수 없습니다. (데이터 부족 또는 호출 제한)")
                    else:
                        st.dataframe(df_rank.head(20), use_container_width=True)
                        st.caption("정렬: 상승 확률 내림차순 (상위 20개 표시)")
