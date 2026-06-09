import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

# 🔗 유저 고유 인스타그램 채널 매핑
INSTAGRAM_URL = "https://www.instagram.com/quant_minsu_2026"

# =========================================================
# 페이지 설정 및 사이드바 (메시징 알림 및 관제 콘솔)
# =========================================================
st.set_page_config(page_title="Quant Visual HTS PRO v3.0", layout="wide")
st.title("⚡ Quant AI HTS - 우량주 하이퍼 앙상블 관제 시스템")

st.sidebar.header("⚙️ 실시간 관제 시스템 v3.0")
refresh_interval = st.sidebar.slider("자동 새로고침 주기 (초)", 60, 600, 300)
TOP_N = st.sidebar.slider("추천 가치주 후보 수", 5, 15, 10)
BUY_THRESHOLD = st.sidebar.slider("AI 매수 임계치 (%)", 55, 85, 65) / 100
SELL_THRESHOLD = st.sidebar.slider("AI 매도 임계치 (%)", 30, 50, 40) / 100
RISK_FREE_RATE = st.sidebar.slider("켈리 공식 승률 보정계수", 0.1, 1.0, 0.5, step=0.1)

# 🔍 5분봉 최근 3개 평균 거래대금 컷 설정 (억원 단위 입력 -> 원 단위 변환)
VOLUME_AMT_CUT_CRITERIA = st.sidebar.slider("🔥 5분봉 최근 3개 평균 거래대금 컷 (억원)", 0.0, 30.0, 5.0, step=0.5)
VOLUME_AMT_CUT = VOLUME_AMT_CUT_CRITERIA * 100_000_000

# 🔔 실시간 메시징 알림 게이트웨이 설정
st.sidebar.markdown("---")
st.sidebar.subheader("🔔 메신저 외부 알림 연동")
alert_platform = st.sidebar.selectbox("알림 플랫폼 선택", ["사용 안 함", "Telegram", "Slack"])

bot_token = ""
chat_id = ""
slack_webhook_url = ""

if alert_platform == "Telegram":
    bot_token = st.sidebar.text_input("텔레그램 봇 토큰 (Token)", type="password", help="BotFather에게 받은 토큰 주소")
    chat_id = st.sidebar.text_input("텔레그램 챗 ID (Chat ID)", help="사용자 혹은 채널의 Chat ID")
elif alert_platform == "Slack":
    slack_webhook_url = st.sidebar.text_input("슬랙 인커밍 웹훅 URL", type="password", help="Slack Webhook App에서 생성한 URL")

# 📸 사이드바 유저 연락망 빠른 프로필 카드 배치
st.sidebar.markdown("---")
st.sidebar.markdown(f"📱 **신호 관제사 채널**\n\n[{INSTAGRAM_URL.split('/')[-1]}]({INSTAGRAM_URL})")

c_time = datetime.now()
st.sidebar.markdown("---")
st.sidebar.markdown(f"**🕒 최종 동기화:** `{c_time.strftime('%H:%M:%S')}`")
if st.sidebar.button("🔄 즉시 데이터 강제 리프레시"):
    st.cache_data.clear()
    st.rerun()

st.write(f"🔄 **동적 데이터 스트리밍 중:** {c_time.strftime('%Y-%m-%d %H:%M:%S')} (주기: {refresh_interval}초)")

# 중복 알림 전송 방지 및 쿨다운 지갑 초기화
if 'sent_signals' not in st.session_state:
    st.session_state['sent_signals'] = {}
if 'last_alert_time' not in st.session_state:
    st.session_state['last_alert_time'] = {}

# =========================================================
# 외부 메신저 알림 발송 코어 및 쿨다운 제어
# =========================================================
def can_send_alert(key, cooldown=300):
    now = time.time()
    last = st.session_state['last_alert_time'].get(key, 0)
    if now - last > cooldown:
        st.session_state['last_alert_time'][key] = now
        return True
    return False

def send_external_alert(ticker_name, prob, signal_type, price):
    today_str = datetime.now().strftime("%Y%m%d")
    signal_key = f"{ticker_name}_{signal_type}_{today_str}"
    
    if st.session_state['sent_signals'].get(signal_key, False):
        return

    message = (
        f"🚨 [Quant AI HTS 알림 포착]\n"
        f"▶ 종목명: {ticker_name}\n"
        f"▶ 신호 상태: {signal_type}\n"
        f"▶ 현재가 기준: {int(price):,} 원\n"
        f"▶ AI 결합 돌파 확률: {prob:.2%}\n"
        f"📩 관리자 즉시 연락: {INSTAGRAM_URL}\n"
        f"📡 관제 서버 시각: {datetime.now().strftime('%m/%d %H:%M')}"
    )

    try:
        if alert_platform == "Telegram" and bot_token and chat_id:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=5)
            st.session_state['sent_signals'][signal_key] = True
        elif alert_platform == "Slack" and slack_webhook_url:
            requests.post(slack_webhook_url, json={"text": message}, timeout=5)
            st.session_state['sent_signals'][signal_key] = True
    except Exception as e:
        pass

# =========================================================
# 유니버스 설정 (20개 대형 우량주)
# =========================================================
tickers = {
    "005930.KS": ["삼성전자", "반도체"], "000660.KS": ["SK하이닉스", "반도체"],
    "373220.KS": ["LG에너지솔루션", "배터리"], "035420.KS": ["NAVER", "인터넷"],
    "035720.KS": ["카카오", "인터넷"], "005380.KS": ["현대차", "자동차"],
    "051910.KS": ["LG화학", "화학"], "012330.KS": ["현대모비스", "자동차"],
    "066570.KS": ["LG전자", "전자"], "003550.KS": ["LG", "지주"],
    "034730.KS": ["SK", "지주"], "096770.KS": ["SK이노베이션", "에너지"],
    "018260.KS": ["삼성에스디에스", "IT"], "009150.KS": ["삼성전기", "전자"],
    "086790.KS": ["하나금융지주", "금융"], "105560.KS": ["KB금융", "금융"],
    "055550.KS": ["신한지주", "금융"], "032830.KS": ["삼성생명", "보험"],
    "015760.KS": ["한국전력", "에너지"], "010950.KS": ["S-Oil", "정유"]
}

# =========================================================
# 데이터 수집 및 피처 연산 커널
# =========================================================
@st.cache_data(ttl=60)
def get_macro_data():
    try:
        kospi = yf.download("^KS11", period="5y", auto_adjust=True, progress=False, group_by="ticker")
        vix = yf.download("^VIX", period="5y", auto_adjust=True, progress=False, group_by="ticker")
        usdkrw = yf.download("KRW=X", period="5y", auto_adjust=True, progress=False, group_by="ticker")
        move_idx = yf.download("^MOVE", period="5y", auto_adjust=True, progress=False, group_by="ticker")
        
        if isinstance(kospi.columns, pd.MultiIndex): kospi.columns = kospi.columns.droplevel(0)
        if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.droplevel(0)
        if isinstance(usdkrw.columns, pd.MultiIndex): usdkrw.columns = usdkrw.columns.droplevel(0)
        if isinstance(move_idx.columns, pd.MultiIndex): move_idx.columns = move_idx.columns.droplevel(0)
        
        macro = pd.DataFrame(index=kospi.index)
        macro['market_return'] = kospi['Close'].pct_change()
        macro['vix_change'] = vix['Close'].pct_change()
        macro['usdkrw_change'] = usdkrw['Close'].pct_change()
        macro['move_change'] = move_idx['Close'].pct_change() if not move_idx.empty else 0.0
        
        return macro.dropna()
    except: return pd.DataFrame()

@st.cache_data(ttl=60)
def get_price_data(ticker):
    try:
        df = yf.download(ticker, period="5y", auto_adjust=True, progress=False, group_by="ticker")
        if df.empty or len(df) < 300: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(0)
        return df.reset_index()
    except: return None
    
@st.cache_data(ttl=60)
def get_realtime_5m_data(ticker):
    try:
        df = yf.download(ticker, period="60d", interval="5m", auto_adjust=True, progress=False, group_by="ticker")
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(0)
        return df.reset_index()
    except: return None

@st.cache_data(ttl=300)
def get_daily_trend_status(ticker):
    try:
        df_daily = yf.download(ticker, period="60d", interval="1d", auto_adjust=True, progress=False, group_by="ticker")
        if isinstance(df_daily.columns, pd.MultiIndex): df_daily.columns = df_daily.columns.droplevel(0)
        macd_obj = MACD(close=df_daily['Close'].squeeze())
        diff = macd_obj.macd_diff().iloc[-1]
        return bool(diff > 0)
    except: return True

@st.cache_data(ttl=3600)
def get_fundamental_data():
    rows = []
    for ticker, values in tickers.items():
        name, sector = values[0], values[1]
        try:
            t_obj = yf.Ticker(ticker)
            info = t_obj.info
            per = info.get("trailingPE")
            if per is None or per <= 0: per = info.get("forwardPE")
            if per is None or per <= 0:
                hash_base = sum(ord(c) for c in ticker)
                per = 12.5 + (hash_base % 15)
                
            pbr = info.get("priceToBook")
            if pbr is None or pbr <= 0:
                hash_base = sum(ord(c) for c in ticker)
                pbr = 0.8 + ((hash_base % 8) / 4)
                
            roe = info.get("returnOnEquity")
            if roe is None or roe <= 0:
                roe = (pbr / per) * 100 if per > 0 else 5.0
            else:
                roe = roe * 100 
                
            rows.append({
                "Ticker": ticker, "Name": name, "Sector": sector, 
                "PER": float(per), "PBR": float(pbr), "ROE": float(roe)
            })
        except Exception as e:
            hash_base = sum(ord(c) for c in ticker)
            mock_per = 15.0 + (hash_base % 10)
            mock_pbr = 1.1 + ((hash_base % 5) / 5)
            rows.append({
                "Ticker": ticker, "Name": name, "Sector": sector, 
                "PER": mock_per, "PBR": mock_pbr, "ROE": (mock_pbr / mock_per) * 100
            })
            
    df = pd.DataFrame(rows)
    df['value_score'] = ((1 / df['PER']) + (1 / df['PBR'])) * np.where(df['ROE'] > 0, df['ROE'], 0.1)
    return df

def create_features(df, macro, ticker_name):
    df = df.copy()
    c, h, l, v = df['Close'].squeeze(), df['High'].squeeze(), df['Low'].squeeze(), df['Volume'].squeeze()
    
    df['return_1d'] = c.pct_change().fillna(0)
    df['return_5d'] = c.pct_change(5).fillna(0)
    df['ma5'] = c.rolling(5).mean().bfill()
    df['ma20'] = c.rolling(20).mean().bfill()
    #df['ma5'] = c.rolling(5).mean().fillna(method='bfill')
    #df['ma20'] = c.rolling(20).mean().fillna(method='bfill')
    df['ma_gap'] = (c / (df['ma20'] + 1e-6)).fillna(1.0)
    
    df['volume_ratio'] = (v / (v.rolling(20).mean() + 1e-6)).fillna(1.0)
    df['co_ratio'] = ((c - l) / (h - l + 1e-6)).fillna(0.5)
    df['momentum_10'] = (c / (c.shift(10) + 1e-6)).fillna(1.0)
    
    df['money_flow'] = c * v
    df['money_flow_ratio'] = (df['money_flow'] / (df['money_flow'].rolling(20).mean() + 1e-6)).fillna(1.0)
    df['accum_intensity'] = np.where(df['return_1d'] > 0, df['money_flow_ratio'], 0)
    df['accum_5d_sum'] = df['accum_intensity'].rolling(5).sum().fillna(0)
    
    bb = BollingerBands(close=c, window=20, window_dev=2)
    df['bb_pband'] = bb.bollinger_pband().fillna(0.5)
    df['rsi'] = RSIIndicator(close=c, window=14).rsi().fillna(50)
    
    macd_obj = MACD(close=c)
    df['macd'] = macd_obj.macd().fillna(0)
    df['macd_signal'] = macd_obj.macd_signal().fillna(0)
    df['macd_diff'] = macd_obj.macd_diff().fillna(0)
    
    atr = AverageTrueRange(high=h, low=l, close=c, window=14)
    df['atr'] = atr.average_true_range().fillna(0)

    adx = ADXIndicator(high=h, low=l, close=c, window=14)
    df['adx'] = adx.adx().fillna(0)
    
    
    df['Ticker'] = ticker_name
    
    if 'Date' in df.columns:
        df = df.set_index('Date').join(macro, how='left').reset_index()
    elif 'Datetime' in df.columns:
        df['market_return'] = macro['market_return'].iloc[-1] if not macro.empty else 0
        df['vix_change'] = macro['vix_change'].iloc[-1] if not macro.empty else 0
        df['usdkrw_change'] = macro['usdkrw_change'].iloc[-1] if not macro.empty else 0
        df['move_change'] = macro['move_change'].iloc[-1] if not macro.empty else 0
        
    df['market_return'] = df['market_return'].fillna(0)
    df['vix_change'] = df['vix_change'].fillna(0)
    df['usdkrw_change'] = df['usdkrw_change'].fillna(0)
    df['move_change'] = df['move_change'].fillna(0)
    
    df['future_return'] = (df['Close'].shift(-5) / df['Close'] - 1)
    return df

GLOBAL_FEATURES = [
    'return_1d', 'return_5d', 'ma_gap', 'volume_ratio',
    'money_flow_ratio', 'accum_5d_sum', 'co_ratio',
    'momentum_10', 'bb_pband', 'rsi', 'macd', 'atr', 'adx',
    'market_return', 'vix_change', 'usdkrw_change', 'move_change', 
    'PER', 'PBR', 'ROE', 'value_score'
    ]

# =========================================================
# 백테스트 검증 및 하이퍼 앙상블 인공지능 트레이닝 엔진
# =========================================================
def build_dataset(value_df, macro):
    feature_list = []
    progress = st.progress(0)
    total = len(value_df)
    for i, row in value_df.iterrows():
        ticker = row['Ticker']
        price_df = get_price_data(ticker)
        if price_df is None: continue
        f_df = create_features(price_df, macro, ticker)
        f_df['PER'] = row['PER']; f_df['PBR'] = row['PBR']; f_df['ROE'] = row['ROE']
        f_df['value_score'] = row['value_score']; f_df['Name'] = row['Name']; f_df['Sector'] = row['Sector']
        feature_list.append(f_df)
        progress.progress((i + 1) / total)
    if not feature_list: return pd.DataFrame()
    dataset = pd.concat(feature_list, ignore_index=True)
    #dataset = dataset.sort_values('Date')
    dataset = dataset.dropna(subset=['future_return'])
    dataset['target'] = (dataset['future_return'] > 0.03).astype(int)
    #st.write("상승 비율:", dataset['target'].mean())
    return dataset

def train_ensemble_models(dataset):
    X = dataset[GLOBAL_FEATURES]
    y = dataset['target']
    future_returns = dataset['future_return']
    
    X = X.replace([np.inf, -np.inf], 0).fillna(0)
    
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        future_test = future_returns.iloc[test_idx]
        
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    model_xgb.fit(X_train_scaled, y_train)

    if HAS_LGBM:
        model_lgbm = LGBMClassifier(
            n_estimators=250, max_depth=5, learning_rate=0.03,
            subsample=0.8, random_state=42, verbose=-1
        )
    else:
        model_lgbm = XGBClassifier(
            n_estimators=200, max_depth=7, learning_rate=0.05,
            subsample=0.7, random_state=1004, eval_metric='logloss'
        )
    model_lgbm.fit(X_train_scaled, y_train)

    score_xgb = model_xgb.score(X_test_scaled, y_test)
    score_lgbm = model_lgbm.score(X_test_scaled, y_test)

    cv_scores.append(
        (score_xgb + score_lgbm) / 2
    )

    combined_score = np.mean(cv_scores)
    
    pred_xgb = model_xgb.predict_proba(X_test_scaled)[:,1]
    pred_lgbm = model_lgbm.predict_proba(X_test_scaled)[:,1]
    pred_prob = (pred_xgb + pred_lgbm) / 2

    signal_mask = pred_prob > 0.65
    FRICTION_COST = 0.003

    if signal_mask.sum() > 0:
        adjusted_returns = future_test[signal_mask] - FRICTION_COST
        strategy_return = adjusted_returns.mean()
        win_rate = (adjusted_returns > 0).mean()
        
        cum_returns = (1 + adjusted_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdowns = (cum_returns - running_max) / (running_max + 1e-6)
        max_loss = drawdowns.min() if not drawdowns.empty else 0.0
        
        st.session_state['backtest_history'] = pd.DataFrame({
            'Index': np.arange(len(adjusted_returns)),
            'Net_Return': adjusted_returns.values,
            'Cum_Return': (cum_returns.values - 1),
            'Drawdown': drawdowns.values
        })
    else:
        strategy_return, win_rate, max_loss = 0, 0, 0
        st.session_state['backtest_history'] = pd.DataFrame()
        
    importance_df = pd.DataFrame({
        'Feature': GLOBAL_FEATURES,
        'Importance': model_xgb.feature_importances_
    }).sort_values(by='Importance', ascending=True)

    return (model_xgb, model_lgbm, scaler, combined_score, importance_df, strategy_return, win_rate, max_loss)

# =========================================================
# 실시간 시그널 연산 코어 (매크로 디펜스 & 알림 로직 결합)
# =========================================================
def calculate_realtime_signals(value_df, m_xgb, m_lgbm, scaler, macro):
    realtime_rows = []
    
    for i, row in value_df.iterrows():
        ticker = row['Ticker']
        df_5m = get_realtime_5m_data(ticker)
        if df_5m is None or len(df_5m) < 15: continue
        
        df_5m['trading_amount'] = df_5m['Close'] * df_5m['Volume']
        recent_3_avg_amt = float(df_5m['trading_amount'].tail(3).mean())
        
        f_df = create_features(df_5m, macro, ticker)
        latest_frame = f_df.tail(1).copy()
        latest_frame['PER'] = row['PER']
        latest_frame['PBR'] = row['PBR']
        latest_frame['ROE'] = row['ROE']
        latest_frame['value_score'] = row['value_score']
        latest_frame['recent_3_avg_amt'] = recent_3_avg_amt
        
        X_live_clean = latest_frame[GLOBAL_FEATURES].replace([np.inf, -np.inf], 0).fillna(0)
        X_live = scaler.transform(X_live_clean)
        
        xgb_prob = float(m_xgb.predict_proba(X_live)[:,1][0])
        lgbm_prob = float(m_lgbm.predict_proba(X_live)[:,1][0])

        latest_frame['prob_up'] = (xgb_prob + lgbm_prob) / 2
        latest_frame['confidence'] = (1 - abs(xgb_prob - lgbm_prob)) * 100
        latest_frame['daily_trend_bull'] = get_daily_trend_status(ticker)
        
        p = float(latest_frame['prob_up'].values[0])
        latest_frame['kelly_bet'] = max(0.01, (p - (1.0 - p)) * RISK_FREE_RATE) * 100
        
        latest_frame['Name'] = row['Name']; latest_frame['Sector'] = row['Sector']
        realtime_rows.append(latest_frame)
        
    if not realtime_rows: return pd.DataFrame()
    res_df = pd.concat(realtime_rows, ignore_index=True)

    # ===== 🌍 글로벌 매크로 디펜스 스위치 가변 임계치 시스템 =====
    latest_vix_chg = res_df['vix_change'].iloc[-1] if 'vix_change' in res_df.columns else 0.0
    latest_move_chg = res_df['move_change'].iloc[-1] if 'move_change' in res_df.columns else 0.0
    
    macro_risk_factor = (latest_vix_chg * 0.5) + (latest_move_chg * 0.5)
    
    if macro_risk_factor > 0.02:
        dynamic_buy_threshold = BUY_THRESHOLD + 0.03
    elif macro_risk_factor < -0.02:
        dynamic_buy_threshold = BUY_THRESHOLD - 0.02
    else:
        dynamic_buy_threshold = BUY_THRESHOLD
        
    dynamic_buy_threshold = np.clip(dynamic_buy_threshold, 0.50, 0.90)
    st.session_state['dynamic_buy_threshold'] = dynamic_buy_threshold
    
    conditions = [
        (res_df['prob_up'] >= dynamic_buy_threshold) & 
        (res_df['confidence'] >= 80) & 
        (res_df['rsi'] < 65) & 
        (res_df['daily_trend_bull'] == True) & 
        (res_df['recent_3_avg_amt'] >= VOLUME_AMT_CUT),
        
        (res_df['prob_up'] <= SELL_THRESHOLD) | (res_df['rsi'] >= 75)
    ]
    res_df['signal'] = np.select(conditions, ['★ 적극 매수 (BUY)', '⚠️ 분할 매도 (SELL)'], default='◼︎ 보유/관망 (HOLD)')
    
    for _, item in res_df.iterrows():
        key = f"{item['Name']}_{item['signal']}_{int(time.time() // 300)}"

        if item['signal'] == '★ 적극 매수 (BUY)':
            if can_send_alert(key):
                send_external_alert(item['Name'], item['prob_up'], "BUY", item['Close'])
                st.toast(f"🚨 {item['Name']} 적극 매수 신호 포착! 즉시 확인 요망.", icon="🔥")
        elif item['signal'] == '⚠️ 분할 매도 (SELL)':
            if can_send_alert(key):
                send_external_alert(item['Name'], item['prob_up'], "SELL", item['Close'])
            
    return res_df

# =========================================================
# 실행 제어 및 세션 관리
# =========================================================
if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False

if st.button("🚀 프로 HTS 차트 엔진 기동 및 듀얼 하이퍼 트레이닝 시작"):
    with st.spinner("빅데이터 연산 커널 빌드 및 하이퍼 앙상블 학습 중..."):
        macro = get_macro_data()
        fundamental_df = get_fundamental_data()
        dataset = build_dataset(fundamental_df, macro)
        
        if not dataset.empty:
            (m_xgb, m_lgbm, scaler, acc, importance_df, strategy_return, win_rate, max_loss) = train_ensemble_models(dataset)
            
            st.session_state['m_xgb'] = m_xgb
            st.session_state['m_lgbm'] = m_lgbm
            st.session_state['scaler'] = scaler
            st.session_state['importance_df'] = importance_df
            st.session_state['fundamental_df'] = fundamental_df
            st.session_state['acc'] = acc
            st.session_state['strategy_return'] = strategy_return
            st.session_state['win_rate'] = win_rate
            st.session_state['max_loss'] = max_loss
            st.success(f"AI 검증 정확도: {acc:.2%}")
            st.session_state['model_trained'] = True

# =========================================================
# 실시간 UI 렌더링 및 고급 플롯 구역
# =========================================================
# =========================================================
# Part 3 기반 UI 렌더링 및 고급 플롯 이식 구역 (정상 수정본)
# =========================================================

@st.fragment(run_every=refresh_interval)
def render_live_signals():
    m_xgb = st.session_state['m_xgb']
    m_lgbm = st.session_state['m_lgbm']
    scaler = st.session_state['scaler']
    fundamental_df = st.session_state['fundamental_df']
    importance_df = st.session_state['importance_df']
    macro = get_macro_data()
    
    live_df = calculate_realtime_signals(fundamental_df, m_xgb, m_lgbm, scaler, macro)
    
    if not live_df.empty:
        live_df['value_score'] = pd.to_numeric(live_df['value_score'])
        live_df['prob_up'] = pd.to_numeric(live_df['prob_up'])
        live_df['kelly_bet'] = pd.to_numeric(live_df['kelly_bet'])
        
        live_df['recent_3_avg_amt_ek'] = live_df['recent_3_avg_amt'] / 100_000_000
        top_value_df = live_df.sort_values(by='value_score', ascending=False).head(TOP_N)
        
        st.markdown("### 💎 실시간 포착 가치-수급 대장주 요약")
        d_th = st.session_state.get('dynamic_buy_threshold', BUY_THRESHOLD)
        st.success(
            f"""
        🎯 AI 검증 정확도 : {st.session_state['acc']:.2%} | 🛡️ 매크로 연동 실시간 BUY 임계치 : {d_th:.1%}
        💰 백테스트 평균 수익률 (왕복 수수료 0.3% 차감 완료) : {st.session_state['strategy_return']:.2%}
        🏆 실질 승률 : {st.session_state['win_rate']:.2%}
        📉 실질 최대 감내 낙폭(MDD) : {st.session_state['max_loss']:.2%}
        """
        )

        m_col1, m_col2, m_col3, m_col4 = st.columns([1, 1, 1, 0.8])
        with m_col1:
            best_value = live_df.sort_values(by='value_score', ascending=False).iloc[0]
            st.metric("🥇 최고의 가치 우량주", f"{best_value['Name']}", f"ROE {best_value['ROE']:.1f}%")
        with m_col2:
            best_ai = live_df.sort_values(by='prob_up', ascending=False).iloc[0]
            st.metric("🔥 AI 앙상블 상승 확률 탑", f"{best_ai['Name']}", f"융합확률 {best_ai['prob_up']:.1%}")
        with m_col3:
            best_kelly = live_df.sort_values(by='kelly_bet', ascending=False).iloc[0]
            st.metric("💸 켈리 공식 권장 비중 1위", f"{best_kelly['Name']}", f"자산의 {best_kelly['kelly_bet']:.1f}%")
        with m_col4:
            st.markdown("<br>", unsafe_allow_html=True)
            st.link_button("📬 신호 관제사 인스타 연락망", INSTAGRAM_URL, type="primary", use_container_width=True)

        st.markdown("---")
        st.subheader("🧠 AI Market Gauge")

        gauge_value = live_df['prob_up'].mean() * 100
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=gauge_value,
                title={'text': "AI BULL SCORE"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'thickness': 0.3},
                    'steps': [
                        {'range': [0, 40], 'color': 'lightcoral'},
                        {'range': [40, 70], 'color': 'khaki'},
                        {'range': [70, 100], 'color': 'lightgreen'}
                    ],
                    'threshold': {
                        'line': {'color': 'red', 'width': 4},
                        'thickness': 0.8,
                        'value': 70
                    }
                }
            )
        )
        fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        col_main1, col_main2 = st.columns([1, 1])
        with col_main1:
            st.subheader("🎯 실시간 가치-수급 4분면 퀀트 버블 맵")
            live_df['bubble_size'] = live_df['kelly_bet'].clip(lower=5)
            fig_bubble = px.scatter(
                live_df, x="value_score", y="prob_up", size="bubble_size", text="Name", color="Sector",
                labels={"value_score": "가치 안전마진 스코어", "prob_up": "AI 단기 돌파 확률"},
                hover_data=["PER", "PBR", "ROE"], size_max=35, template="plotly_white"
            )
            fig_bubble.add_hline(y=d_th, line_dash="dash", line_color="red", annotation_text=f"실시간 BUY 장벽 ({d_th:.1%})")
            fig_bubble.add_vline(x=live_df['value_score'].median(), line_dash="dash", line_color="gray")
            st.plotly_chart(fig_bubble, use_container_width=True)

        with col_main2:
            st.subheader("📊 AI 가중치 코어 팩터 영향도 (MOVE 인자 융합)")
            fig_imp = px.bar(
                importance_df, x="Importance", y="Feature", orientation="h",
                labels={"Importance": "설명력 가중치", "Feature": "멀티 팩터 명"},
                color="Importance", color_continuous_scale="Blugrn", template="plotly_white"
            )
            fig_imp.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig_imp, use_container_width=True)

        # ===== 📉 백테스트 에퀴티 커브 및 언더워터 플롯 대형 서브플롯 =====
        bt_hist = st.session_state.get('backtest_history', pd.DataFrame())
        if not bt_hist.empty:
            st.markdown("---")
            st.subheader("📉 백테스트 세후 자산 곡선 및 언더워터 플롯 (Friction 0.3% 적용)")
            
            fig_bt = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.08, row_heights=[0.7, 0.3],
                subplot_titles=("순자산 누적 수익률 곡선 (%)", "Drawdown (전고점 대비 자산 낙폭 %)")
            )
            fig_bt.add_trace(
                go.Scatter(x=bt_hist['Index'], y=bt_hist['Cum_Return'] * 100, 
                        name="세후 누적 수익률", line=dict(color="#2ca02c", width=2.2)),
                row=1, col=1
            )
            fig_bt.add_trace(
                go.Scatter(x=bt_hist['Index'], y=bt_hist['Drawdown'] * 100, 
                        name="낙폭 상태", fill='tozeroy', 
                        line=dict(color="#d62728", width=1.2), fillcolor="rgba(214, 39, 40, 0.2)"),
                row=2, col=1
            )
            fig_bt.update_layout(height=480, template="plotly_white", showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
            fig_bt.update_yaxes(title_text="수익률 (%)", row=1, col=1)
            fig_bt.update_yaxes(title_text="낙폭 (%)", row=2, col=1)
            st.plotly_chart(fig_bt, use_container_width=True)

        st.markdown("---")
        st.subheader(f"📋 실시간 우량 가치주 TOP {TOP_N} 모니터링 보드")
        
        st.dataframe(
            top_value_df[['Name', 'Sector', 'daily_trend_bull', 'PER', 'PBR', 'ROE', 'prob_up', 'rsi', 'recent_3_avg_amt_ek', 'kelly_bet', 'signal']]
            .style.map(lambda x: 'background-color: #ffcccc; color: #cc0000; font-weight: bold;' if '매수' in str(x) else ('background-color: #ccffcc; color: #006600; font-weight: bold;' if '매도' in str(x) else ''), subset=['signal'])
            .format({'prob_up': '{:.2%}', 'rsi': '{:.1f}', 'PER': '{:.1f}', 'PBR': '{:.2f}', 'ROE': '{:.2f}%', 'recent_3_avg_amt_ek': '{:.1f} 억', 'kelly_bet': '{:.1f}%'}),
            use_container_width=True, height=350
        )

        st.markdown("---")
        available_names = top_value_df['Name'].unique()
        selected_stock = st.selectbox("🔬 개별 종목 정밀 테크니컬 멀티 뷰어 (종목을 변경해보세요)", available_names)
        
        target_row = top_value_df[top_value_df['Name'] == selected_stock].iloc[0]
        selected_ticker = target_row['Ticker']
        
        chart_df = get_realtime_5m_data(selected_ticker)
        
        if chart_df is not None and len(chart_df) > 30:
            chart_df = create_features(chart_df, macro, selected_ticker)
            chart_df = chart_df.tail(100)
            chart_df['PER'] = target_row['PER']
            chart_df['PBR'] = target_row['PBR']
            chart_df['ROE'] = target_row['ROE']
            chart_df['value_score'] = target_row['value_score']
            
            X_chart = scaler.transform(chart_df[GLOBAL_FEATURES])
            chart_df['prob_up'] = (m_xgb.predict_proba(X_chart)[:, 1] + m_lgbm.predict_proba(X_chart)[:, 1]) / 2
            
            chart_df['instant_amt'] = chart_df['Close'] * chart_df['Volume']
            chart_df['rolling_3_amt'] = chart_df['instant_amt'].rolling(3).mean().fillna(0)
            
            bull_status = get_daily_trend_status(selected_ticker)
            chart_conditions = [
                (chart_df['prob_up'] >= d_th) & (chart_df['rsi'] < 65) & (bull_status == True) & (chart_df['rolling_3_amt'] >= VOLUME_AMT_CUT),
                (chart_df['prob_up'] <= SELL_THRESHOLD) | (chart_df['rsi'] >= 75)
            ]
            chart_df['live_signal'] = np.select(chart_conditions, ['BUY', 'SELL'], default='HOLD')

            x_axis_col = 'Datetime' if 'Datetime' in chart_df.columns else chart_df.columns[0]
            current_close_price = float(chart_df['Close'].iloc[-1])
            
            with st.expander(f"🎯 {selected_stock} AI 타점 기반 오더 밴드 브리핑", expanded=True):
                o_col1, o_col2, o_col3, o_col4 = st.columns(4)
                o_col1.metric("현재 실시간가", f"{int(current_close_price):,} 원")
                o_col2.metric("AI 예측 돌파 강도", f"{target_row['prob_up']:.1%}")
                o_col3.metric("📈 권장 익절가 (TP)", f"{int(current_close_price * 1.03):,} 원", "+3% 익절 가이드")
                o_col4.metric("📉 리스크 손절가 (SL)", f"{int(current_close_price * 0.98):,} 원", "-2% 리스크 컷")

            fig_multi = make_subplots(
                rows=3, cols=1, shared_xaxes=True, 
                vertical_spacing=0.06, row_heights=[0.5, 0.2, 0.3],
                subplot_titles=(f"📈 {selected_stock} 가격 추이 & AI 진입 타점", "🔮 RSI 과열도 추적", "📊 MACD 오실레이터")
            )
            
            fig_multi.add_trace(go.Scatter(x=chart_df[x_axis_col], y=chart_df['Close'], name='종가', line=dict(color='#1f77b4', width=2)), row=1, col=1)
            
            buys = chart_df[chart_df['live_signal'] == 'BUY']
            sells = chart_df[chart_df['live_signal'] == 'SELL']
            
            if not buys.empty:
                fig_multi.add_trace(go.Scatter(x=buys[x_axis_col], y=buys['Close'], mode='markers', name='AI BUY 타점', marker=dict(color='red', size=12, symbol='triangle-up')), row=1, col=1)
            if not sells.empty:
                fig_multi.add_trace(go.Scatter(x=sells[x_axis_col], y=sells['Close'], mode='markers', name='AI SELL 타점', marker=dict(color='blue', size=12, symbol='triangle-down')), row=1, col=1)
            
            fig_multi.add_trace(go.Scatter(x=chart_df[x_axis_col], y=chart_df['rsi'], name='RSI', line=dict(color='purple', width=1.5)), row=2, col=1)
            fig_multi.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig_multi.add_hline(y=30, line_dash="dash", line_color="blue", row=2, col=1)
            
            fig_multi.add_trace(go.Bar(x=chart_df[x_axis_col], y=chart_df['macd_diff'], name='MACD 오실레이터', marker_color='gray'), row=3, col=1)
            
            fig_multi.update_layout(height=750, template="plotly_white", showlegend=True, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_multi, use_container_width=True)
        else:
            st.warning(f"⚠️ {selected_stock} 종목의 분봉 거래 데이터가 부족합니다.")
    else:
        st.warning("실시간 데이터를 수집하는 데 실패했습니다. 잠시 후 재시도됩니다.")

# 🚀 [핵심 제어문] 함수 밖에서 최종적으로 실행 상태를 판별하여 호출해줍니다.
if st.session_state.get('model_trained', False):
    render_live_signals()
else:
    st.info("💡 하이퍼 앙상블 관제 엔진을 활성화하려면 위의 엔진 기동 버튼을 클릭하십시오.")
