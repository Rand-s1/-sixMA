import requests
import ta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import time
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置页面配置
st.set_page_config(
    page_title="鹅的均线密集度扫描器 Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    /* 主要背景和主题 */
    .main {
        padding-top: 2rem;
    }
    
    /* 标题样式 */
    .big-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* 卡片样式 */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #4ecdc4;
        margin: 1rem 0;
    }
    
    /* 按钮样式 */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* 数据表格样式 */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* 侧边栏样式 */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* 警告和信息框样式 */
    .stAlert {
        border-radius: 10px;
    }
    
    /* 进度条样式 */
    .stProgress > div > div {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置常量
class MAConfig:
    ENDPOINTS = ["https://api.bitget.com"]
    PRODUCT_TYPE = "usdt-futures"
    LIMIT = 200  # 增加K线数量以支持MA120
    SLEEP_BETWEEN_REQUESTS = 0.5
    MAX_WORKERS = 10
    
    # 均线配置
    MA_PERIODS = [20, 60, 120]
    EMA_PERIODS = [20, 60, 120]
    LOOKBACK_CANDLES = 5  # 检查最近5根K线
    
    # UI配置
    TIMEFRAMES = {
        "1小时": "1H",
        "4小时": "4H", 
        "1天": "1D"
    }

def create_header():
    """创建页面头部"""
    st.markdown('<h1 class="big-title">📊 鹅的均线密集度扫描器 Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">🎯 MA20/60/120 + EMA20/60/120 密集区域扫描 + MA20触碰检测</p>', unsafe_allow_html=True)
    st.markdown("---")

def create_ma_sidebar():
    """创建均线扫描器侧边栏"""
    with st.sidebar:
        st.markdown("### ⚙️ 扫描设置")
        
        # 时间框架选择
        timeframe_display = st.selectbox(
            "📊 时间框架",
            options=list(MAConfig.TIMEFRAMES.keys()),
            index=1,  # 默认4小时
            help="选择K线时间周期"
        )
        timeframe = MAConfig.TIMEFRAMES[timeframe_display]
        
        st.markdown("### 🎯 密集度设置")
        
        # 密集度阈值设置
        density_threshold = st.slider(
            "密集度阈值 (%)",
            min_value=0.5,
            max_value=10.0,
            value=3.0,
            step=0.1,
            help="6条均线价格范围相对当前价格的百分比阈值"
        )
        
        # 成交量过滤
        volume_top_percent = st.slider(
            "成交量排名前 (%)",
            min_value=10,
            max_value=100,
            value=100,
            step=10,
            help="只扫描成交量排名前N%的合约"
        )
        
        # 高级设置
        with st.expander("🔧 高级设置"):
            show_charts = st.checkbox("显示图表分析", value=True)
            show_ma_values = st.checkbox("显示均线数值", value=False)
            min_price = st.number_input("最低价格过滤", value=0.0, help="过滤价格过低的币种")
            ma20_touch_only = st.checkbox("仅显示MA20触碰", value=False, help="只显示有MA20触碰的结果")
            
        return timeframe, density_threshold, volume_top_percent, show_charts, show_ma_values, min_price, ma20_touch_only

def ping_endpoint(endpoint: str) -> bool:
    """测试端点是否可用"""
    url = f"{endpoint}/api/v2/mix/market/candles"
    params = {
        "symbol": "BTCUSDT",
        "granularity": "4H",
        "limit": 1,
        "productType": MAConfig.PRODUCT_TYPE,
    }
    try:
        r = requests.get(url, params=params, timeout=5)
        return r.status_code == 200 and r.json().get("code") == "00000"
    except:
        return False

def get_working_endpoint() -> str:
    """获取可用端点"""
    for ep in MAConfig.ENDPOINTS:
        for _ in range(3):
            if ping_endpoint(ep):
                return ep
            time.sleep(1)
    raise RuntimeError("无可用端点，请检查网络连接")

def get_usdt_symbols(base: str) -> List[str]:
    """获取USDT永续合约交易对"""
    url = f"{base}/api/v2/mix/market/contracts"
    params = {"productType": MAConfig.PRODUCT_TYPE}
    
    try:
        r = requests.get(url, params=params, timeout=5)
        j = r.json()
        if j.get("code") != "00000":
            raise RuntimeError(f"获取交易对失败: {j}")
        symbols = [c["symbol"] for c in j["data"]]
        logger.info(f"找到 {len(symbols)} 个USDT永续合约")
        return symbols
    except Exception as e:
        logger.error(f"获取交易对错误: {e}")
        raise

def fetch_candles(base: str, symbol: str, granularity: str) -> pd.DataFrame:
    """获取K线数据"""
    url = f"{base}/api/v2/mix/market/candles"
    params = {
        "symbol": symbol,
        "granularity": granularity,
        "limit": MAConfig.LIMIT,
        "productType": MAConfig.PRODUCT_TYPE,
    }
    
    try:
        r = requests.get(url, params=params, timeout=10)
        j = r.json()
        if j.get("code") != "00000":
            return pd.DataFrame()
            
        cols = ["ts", "open", "high", "low", "close", "volume_base", "volume_quote"]
        df = pd.DataFrame(j["data"], columns=cols)
        df[["open", "high", "low", "close", "volume_base", "volume_quote"]] = df[
            ["open", "high", "low", "close", "volume_base", "volume_quote"]
        ].astype(float)
        df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms")
        return df.sort_values("ts").reset_index(drop=True)
    except Exception as e:
        logger.error(f"{symbol} K线获取失败: {e}")
        return pd.DataFrame()

def fetch_all_tickers(base: str) -> Dict[str, dict]:
    """批量获取ticker数据"""
    url = f"{base}/api/v2/mix/market/tickers"
    params = {"productType": MAConfig.PRODUCT_TYPE}
    
    try:
        r = requests.get(url, params=params, timeout=5)
        j = r.json()
        
        logger.info(f"Ticker API响应: code={j.get('code')}, msg={j.get('msg')}")
        
        if j.get("code") != "00000":
            logger.error(f"API返回错误: {j}")
            return {}
            
        if not isinstance(j.get("data"), list):
            logger.error(f"API数据格式错误: {type(j.get('data'))}")
            return {}
        
        tickers = {}
        for item in j["data"]:
            try:
                # 打印第一个item的结构，用于调试
                if len(tickers) == 0:
                    logger.info(f"Ticker数据结构示例: {list(item.keys())}")
                
                # 兼容不同的字段名
                symbol = item.get("symbol", "")
                if not symbol:
                    continue
                
                # 尝试不同的字段名
                change24h = 0.0
                if "change24h" in item:
                    change24h = float(item["change24h"]) * 100
                elif "chgUtc" in item:
                    change24h = float(item["chgUtc"]) * 100
                elif "changeUtc24h" in item:
                    change24h = float(item["changeUtc24h"]) * 100
                
                # 成交量字段
                volume = 0.0
                if "baseVolume" in item:
                    volume = float(item["baseVolume"])
                elif "baseVol" in item:
                    volume = float(item["baseVol"])
                elif "vol24h" in item:
                    volume = float(item["vol24h"])
                
                # 价格字段
                price = 0.0
                if "close" in item:
                    price = float(item["close"])
                elif "last" in item:
                    price = float(item["last"])
                elif "lastPr" in item:
                    price = float(item["lastPr"])
                
                tickers[symbol] = {
                    "change24h": change24h,
                    "volume": volume,
                    "price": price
                }
                
            except (ValueError, KeyError, TypeError) as e:
                logger.warning(f"处理ticker数据失败 {item.get('symbol', 'unknown')}: {e}")
                continue
        
        logger.info(f"成功获取 {len(tickers)} 个ticker数据")
        return tickers
        
    except requests.exceptions.RequestException as e:
        logger.error(f"网络请求失败: {e}")
        return {}
    except Exception as e:
        logger.error(f"获取ticker数据失败: {e}")
        return {}

def check_ma20_touch(ma20_value: float, k_high: float, k_low: float, k_open: float, k_close: float) -> dict:
    """
    检查K线是否触碰MA20线
    返回详细的触碰信息
    """
    if ma20_value is None:
        return {
            'touched': False,
            'touch_type': 'none',
            'detail': '无MA20数据'
        }
    
    # 允许的误差范围（0.1%）
    tolerance = ma20_value * 0.001
    
    # 检查各种触碰情况
    touch_conditions = {
        'high_touch': abs(k_high - ma20_value) <= tolerance,  # 最高价触碰
        'low_touch': abs(k_low - ma20_value) <= tolerance,    # 最低价触碰
        'open_touch': abs(k_open - ma20_value) <= tolerance,  # 开盘价触碰
        'close_touch': abs(k_close - ma20_value) <= tolerance, # 收盘价触碰
        'cross_through': k_low <= ma20_value <= k_high,       # K线穿过MA20
        'open_close_cross': (k_open <= ma20_value <= k_close) or (k_close <= ma20_value <= k_open)  # 开收盘价跨越
    }
    
    # 判断触碰类型
    if touch_conditions['cross_through']:
        if touch_conditions['open_close_cross']:
            if k_close > k_open:
                touch_type = '阳线穿越'
                detail = f'阳线穿越MA20 (开:{k_open:.4f} 收:{k_close:.4f} MA20:{ma20_value:.4f})'
            else:
                touch_type = '阴线穿越'
                detail = f'阴线穿越MA20 (开:{k_open:.4f} 收:{k_close:.4f} MA20:{ma20_value:.4f})'
        else:
            touch_type = '影线触碰'
            detail = f'影线触碰MA20 (高:{k_high:.4f} 低:{k_low:.4f} MA20:{ma20_value:.4f})'
    elif touch_conditions['high_touch']:
        touch_type = '上影触碰'
        detail = f'上影线精确触碰MA20 ({k_high:.4f}≈{ma20_value:.4f})'
    elif touch_conditions['low_touch']:
        touch_type = '下影触碰'
        detail = f'下影线精确触碰MA20 ({k_low:.4f}≈{ma20_value:.4f})'
    elif touch_conditions['close_touch']:
        touch_type = '收盘触碰'
        detail = f'收盘价精确触碰MA20 ({k_close:.4f}≈{ma20_value:.4f})'
    elif touch_conditions['open_touch']:
        touch_type = '开盘触碰'
        detail = f'开盘价精确触碰MA20 ({k_open:.4f}≈{ma20_value:.4f})'
    else:
        return {
            'touched': False,
            'touch_type': 'none',
            'detail': f'未触碰MA20 (区间:{k_low:.4f}-{k_high:.4f} MA20:{ma20_value:.4f})'
        }
    
    return {
        'touched': True,
        'touch_type': touch_type,
        'detail': detail
    }

def calculate_ma_density(df: pd.DataFrame, density_threshold: float) -> Tuple[Optional[dict], int]:
    """
    计算均线密集度，并检测K线是否触碰MA20
    返回: (density_info, candle_count)
    """
    try:
        close_series = pd.Series(df["close"].astype(float)).reset_index(drop=True)
        candle_count = len(close_series)
        
        # 检查数据是否足够
        min_required = max(MAConfig.MA_PERIODS + MAConfig.EMA_PERIODS) + MAConfig.LOOKBACK_CANDLES
        if candle_count < min_required:
            return None, candle_count
        
        # 计算所有均线
        mas = {}
        for period in MAConfig.MA_PERIODS:
            mas[f'MA{period}'] = ta.trend.sma_indicator(close_series, window=period)
        
        for period in MAConfig.EMA_PERIODS:
            mas[f'EMA{period}'] = ta.trend.ema_indicator(close_series, window=period)
        
        # 获取K线的高低价数据
        high_series = pd.Series(df["high"].astype(float)).reset_index(drop=True)
        low_series = pd.Series(df["low"].astype(float)).reset_index(drop=True)
        open_series = pd.Series(df["open"].astype(float)).reset_index(drop=True)
        
        # 检查最近N根K线的密集度
        current_price = close_series.iloc[-1]
        best_density = None
        
        for i in range(MAConfig.LOOKBACK_CANDLES):
            idx = -(i + 1)  # 从最新开始往前检查
            
            # 获取当前K线的所有均线值
            ma_values = []
            ma_details = {}
            
            for ma_name, ma_series in mas.items():
                if idx < -len(ma_series) or pd.isna(ma_series.iloc[idx]):
                    continue
                value = ma_series.iloc[idx]
                ma_values.append(value)
                ma_details[ma_name] = value
            
            if len(ma_values) < 6:  # 需要6条均线都有值
                continue
            
            # 计算密集度
            max_ma = max(ma_values)
            min_ma = min(ma_values)
            density_range = ((max_ma - min_ma) / current_price) * 100
            
            # 检查是否满足密集度条件
            if density_range <= density_threshold:
                # 🎯 检查K线是否触碰MA20
                ma20_value = ma_details.get('MA20', None)
                k_high = high_series.iloc[idx]
                k_low = low_series.iloc[idx]
                k_open = open_series.iloc[idx]
                k_close = close_series.iloc[idx]
                
                # 检查K线是否触碰MA20的逻辑
                ma20_touch_info = check_ma20_touch(ma20_value, k_high, k_low, k_open, k_close)
                
                # 计算均线中心与当前价格的距离
                ma_center = (max_ma + min_ma) / 2
                distance_from_price = abs((ma_center - current_price) / current_price) * 100
                
                density_info = {
                    'density_range': density_range,
                    'max_ma': max_ma,
                    'min_ma': min_ma,
                    'ma_center': ma_center,
                    'distance_from_price': distance_from_price,
                    'ma_details': ma_details,
                    'candle_index': i,  # 第几根K线前
                    'position_vs_price': 'above' if ma_center > current_price else 'below',
                    
                    # 🎯 MA20触碰信息
                    'ma20_touch': ma20_touch_info['touched'],
                    'ma20_touch_type': ma20_touch_info['touch_type'],
                    'ma20_touch_detail': ma20_touch_info['detail'],
                    'ma20_value': ma20_value,
                    'k_line_info': {
                        'open': k_open,
                        'high': k_high,
                        'low': k_low,
                        'close': k_close
                    }
                }
                
                # 取密集度最好的(范围最小的)
                if best_density is None or density_range < best_density['density_range']:
                    best_density = density_info
        
        return best_density, candle_count
        
    except Exception as e:
        logger.error(f"指标计算错误: {e}")
        return None, 0

def filter_by_volume(tickers: Dict[str, dict], top_percent: int) -> set:
    """按成交量过滤前N%的合约"""
    if not tickers:
        return set()
    
    # 按成交量排序
    sorted_by_volume = sorted(
        [(symbol, data["volume"]) for symbol, data in tickers.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    # 计算前N%的数量
    total_count = len(sorted_by_volume)
    top_count = max(1, int(total_count * top_percent / 100))
    
    # 返回前N%的交易对
    return {symbol for symbol, _ in sorted_by_volume[:top_count]}

def fetch_candles_wrapper(args) -> tuple:
    """并行获取K线数据的包装函数"""
    base, symbol, granularity = args
    df = fetch_candles(base, symbol, granularity)
    if not df.empty:
        df["symbol"] = symbol
    return symbol, df

def scan_ma_density(base: str, symbols: List[str], granularity: str, 
                   density_threshold: float, volume_top_percent: int, 
                   min_price: float = 0, ma20_touch_only: bool = False) -> Tuple[List[dict], dict]:
    """扫描均线密集度"""
    start_time = time.time()
    results = []
    
    # 获取ticker数据
    with st.spinner("📊 正在获取市场数据..."):
        tickers = fetch_all_tickers(base)
        if not tickers:
            st.warning("⚠️ 无法获取完整的市场数据，将使用默认值")
            tickers = {}
    
    # 按成交量过滤
    volume_filtered_symbols = filter_by_volume(tickers, volume_top_percent)
    filtered_symbols = [s for s in symbols if s in volume_filtered_symbols]
    
    st.info(f"📊 成交量过滤后剩余 {len(filtered_symbols)} 个交易对")
    
    # 进度条容器
    progress_container = st.empty()
    status_container = st.empty()
    
    # 并行获取K线数据
    candle_data = {}
    total_symbols = len(filtered_symbols)
    processed = 0
    
    with ThreadPoolExecutor(max_workers=MAConfig.MAX_WORKERS) as executor:
        futures = [executor.submit(fetch_candles_wrapper, (base, symbol, granularity)) for symbol in filtered_symbols]
        
        for future in as_completed(futures):
            symbol, df = future.result()
            processed += 1
            
            if not df.empty:
                candle_data[symbol] = df
                
            # 更新进度
            progress = processed / total_symbols
            progress_container.progress(progress, text=f"🔄 获取K线数据: {processed}/{total_symbols}")
            status_container.info(f"⏱️ 正在处理: {symbol}")
    
    # 清除进度显示
    progress_container.empty()
    status_container.empty()
    
    # 处理数据
    with st.spinner("🧮 正在计算均线密集度和MA20触碰..."):
        insufficient_data = []
        
        for symbol in filtered_symbols:
            try:
                if symbol not in candle_data:
                    continue
                    
                df = candle_data[symbol]
                density_info, candle_count = calculate_ma_density(df, density_threshold)
                
                if density_info is None:
                    insufficient_data.append(symbol)
                    continue
                
                # 如果设置了只显示MA20触碰，则过滤
                if ma20_touch_only and not density_info["ma20_touch"]:
                    continue
                
                # 使用默认值如果ticker数据不可用
                ticker_data = tickers.get(symbol, {
                    "change24h": 0, 
                    "volume": 0, 
                    "price": 0
                })
                
                # 应用价格过滤
                if ticker_data["price"] < min_price:
                    continue
                
                results.append({
                    "symbol": symbol,
                    "density_range": round(density_info["density_range"], 2),
                    "distance_from_price": round(density_info["distance_from_price"], 2),
                    "position": density_info["position_vs_price"],
                    "candle_ago": density_info["candle_index"],
                    "change24h": round(ticker_data["change24h"], 2),
                    "volume": ticker_data["volume"],
                    "price": ticker_data["price"],
                    "max_ma": density_info["max_ma"],
                    "min_ma": density_info["min_ma"],
                    "ma_center": density_info["ma_center"],
                    "ma_details": density_info["ma_details"],
                    "k_lines": candle_count,
                    
                    # MA20触碰信息
                    "ma20_touch": density_info["ma20_touch"],
                    "ma20_touch_type": density_info["ma20_touch_type"],
                    "ma20_touch_detail": density_info["ma20_touch_detail"],
                    "ma20_value": density_info["ma20_value"],
                    "k_line_info": density_info["k_line_info"]
                })
                    
            except Exception as e:
                logger.warning(f"{symbol} 处理失败: {e}")
                continue
    
    # 优先显示有MA20触碰的结果，然后按密集度排序
    results.sort(key=lambda x: (not x["ma20_touch"], x["density_range"]))
    
    scan_stats = {
        "scan_time": time.time() - start_time,
        "total_symbols": len(symbols),
        "volume_filtered": len(filtered_symbols),
        "processed_symbols": len(candle_data),
        "insufficient_data": len(insufficient_data),
        "results_count": len(results)
    }
    
    return results, scan_stats

def format_ma_dataframe(df: pd.DataFrame, show_ma_values: bool = False) -> pd.DataFrame:
    """格式化均线密集度数据框"""
    if df.empty:
        return df
        
    def add_signal_icon(row):
        density = row["density_range"]
        position = row["position"]
        ma20_touch = row["ma20_touch"]
        
        # 基础密集度图标
        if density < 1.0:
            icon = "🎯"  # 极度密集
        elif density < 2.0:
            icon = "🔥"  # 高度密集
        else:
            icon = "📊"  # 密集
        
        # MA20触碰标识
        if ma20_touch:
            ma20_icon = "🎪"  # 触碰MA20的特殊标识
        else:
            ma20_icon = ""
        
        # 位置图标
        pos_icon = "⬆️" if position == "above" else "⬇️"
        
        return f"{icon}{ma20_icon}{pos_icon} {row['symbol']}"
    
    df_formatted = df.copy()
    df_formatted["交易对"] = df.apply(add_signal_icon, axis=1)
    df_formatted["密集度"] = df_formatted["density_range"].apply(lambda x: f"{x:.2f}%")
    df_formatted["距价格"] = df_formatted["distance_from_price"].apply(lambda x: f"{x:.2f}%")
    df_formatted["K线前"] = df_formatted["candle_ago"].apply(lambda x: f"{x}根前")
    df_formatted["24h涨跌"] = df_formatted["change24h"].apply(lambda x: f"{x:+.2f}%")
    df_formatted["位置"] = df_formatted["position"].apply(lambda x: "价格上方" if x == "above" else "价格下方")
    df_formatted["MA20触碰"] = df_formatted.apply(
        lambda row: f"✅ {row['ma20_touch_type']}" if row["ma20_touch"] else "❌ 未触碰", axis=1
    )
    
    columns = ["交易对", "密集度", "距价格", "位置", "K线前", "MA20触碰", "24h涨跌"]
    
    if show_ma_values:
        df_formatted["价格区间"] = df.apply(
            lambda row: f"{row['min_ma']:.4f} ~ {row['max_ma']:.4f}", axis=1
        )
        df_formatted["MA20值"] = df_formatted["ma20_value"].apply(lambda x: f"{x:.4f}" if x else "N/A")
        columns.extend(["价格区间", "MA20值"])
    
    return df_formatted[columns]

def create_ma_statistics_cards(results: List[dict], scan_stats: dict):
    """创建统计信息卡片"""
    if not results:
        return
        
    very_dense = len([r for r in results if r["density_range"] < 1.0])
    dense = len([r for r in results if 1.0 <= r["density_range"] < 2.0])
    ma20_touched = len([r for r in results if r["ma20_touch"]])
    above_price = len([r for r in results if r["position"] == "above"])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎯 极密集(<1%)",
            value=f"{very_dense}",
            help="密集度小于1%的币种"
        )
        
    with col2:
        st.metric(
            label="🔥 高密集(1-2%)",
            value=f"{dense}",
            help="密集度在1-2%之间的币种"
        )
        
    with col3:
        st.metric(
            label="🎪 MA20触碰",
            value=f"{ma20_touched}",
            help="K线触碰MA20的币种数量"
        )
        
    with col4:
        st.metric(
            label="⬆️ 价格上方",
            value=f"{above_price}",
            help="均线密集区在当前价格上方"
        )

def create_density_chart(results: List[dict]):
    """创建密集度分布图"""
    if not results:
        return None
        
    df = pd.DataFrame(results)
    
    fig = px.histogram(
        df, 
        x="density_range", 
        nbins=20,
        title="均线密集度分布",
        labels={"density_range": "密集度 (%)", "count": "币种数量"},
        color_discrete_sequence=["#4ecdc4"]
    )
    
    fig.update_layout(
        template="plotly_white",
        height=400,
        showlegend=False
    )
    
    return fig

def create_ma20_touch_chart(results: List[dict]):
    """创建MA20触碰类型分布图"""
    if not results:
        return None
        
    df = pd.DataFrame(results)
    ma20_touched = df[df["ma20_touch"] == True]
    
    if ma20_touched.empty:
        return None
    
    touch_counts = ma20_touched["ma20_touch_type"].value_counts()
    
    fig = px.pie(
        values=touch_counts.values,
        names=touch_counts.index,
        title="MA20触碰类型分布",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        template="plotly_white",
        height=400
    )
    
    return fig

def main():
    # 创建页面头部
    create_header()
    
    # 创建侧边栏并获取参数
    timeframe, density_threshold, volume_top_percent, show_charts, show_ma_values, min_price, ma20_touch_only = create_ma_sidebar()
    
    # 主要内容区域
    col1, col2 = st.columns([3, 1])
    
    with col2:
        # 扫描按钮
        if st.button("🚀 开始扫描", key="ma_scan_button", help="点击开始扫描均线密集度"):
            scan_pressed = True
        else:
            scan_pressed = False
            
        # 显示当前设置
        with st.expander("📋 当前设置", expanded=True):
            st.write(f"⏰ **时间框架**: {timeframe}")
            st.write(f"🎯 **密集度阈值**: {density_threshold}%")
            st.write(f"📊 **成交量过滤**: 前{volume_top_percent}%")
            st.write(f"📈 **均线组合**: MA20/60/120 + EMA20/60/120")
            st.write(f"🔍 **检查范围**: 最近{MAConfig.LOOKBACK_CANDLES}根K线")
            if ma20_touch_only:
                st.write(f"🎪 **特殊过滤**: 仅显示MA20触碰")
    
    with col1:
        if not scan_pressed:
            # 显示使用说明
            st.markdown("""
            ### 🎯 均线密集度扫描器

            **功能说明**：
            - 🎯 扫描6条均线密集通过的区域
            - 📊 MA20/60/120 + EMA20/60/120组合
            - 🔍 检查最近5根K线内的密集情况
            - 🎪 检测K线是否触碰MA20线
            - 📈 密集度基于当前价格百分比计算

            **应用场景**：
            - 🚪 **突破信号**: 价格接近密集区时关注突破
            - 🛡️ **支撑阻力**: 密集区通常形成强支撑/阻力
            - 📊 **趋势转折**: 均线密集区域常是变盘点
            - 🎪 **MA20信号**: 触碰MA20往往是重要技术信号

            **参数说明**：
            - **密集度阈值**: 6条均线价格范围占当前价格的百分比
            - **成交量过滤**: 只扫描活跃度高的合约
            - **位置标识**: 显示密集区相对当前价格的位置
            - **MA20检测**: 自动检测各种MA20触碰类型

            **图标说明**：
            - 🎯 极密集(<1%) | 🔥 高密集(1-2%) | 📊 普通密集
            - 🎪 MA20触碰 | ⬆️ 价格上方 | ⬇️ 价格下方
            """)
            return
    
    if scan_pressed:
        try:
            # 获取API端点
            with st.spinner("🔗 连接到Bitget API..."):
                base = get_working_endpoint()
                st.success("✅ API连接成功")
            
            # 获取交易对
            with st.spinner("📋 获取交易对列表..."):
                symbols = get_usdt_symbols(base)
                st.success(f"✅ 找到 {len(symbols)} 个USDT永续合约")
            
            # 执行扫描
            results, scan_stats = scan_ma_density(
                base, symbols, timeframe, density_threshold, 
                volume_top_percent, min_price, ma20_touch_only
            )
            
            # 显示扫描统计
            st.success(f"✅ 扫描完成! 耗时 {scan_stats['scan_time']:.1f} 秒")
            
            if scan_stats['insufficient_data'] > 0:
                st.info(f"ℹ️ 有 {scan_stats['insufficient_data']} 个币种数据不足，已跳过")
            
            # 显示统计卡片
            create_ma_statistics_cards(results, scan_stats)
            
            # 显示结果
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if results:
                filter_text = "MA20触碰" if ma20_touch_only else "均线密集"
                st.markdown(f"### 🎯 {filter_text}区域发现 (密集度 ≤ {density_threshold}%)")
                
                # 分类显示
                ma20_results = [r for r in results if r["ma20_touch"]]
                other_results = [r for r in results if not r["ma20_touch"]]
                
                if ma20_results:
                    st.markdown("#### 🎪 MA20触碰信号")
                    ma20_df = pd.DataFrame(ma20_results)
                    formatted_ma20 = format_ma_dataframe(ma20_df, show_ma_values)
                    st.dataframe(formatted_ma20, use_container_width=True, hide_index=True)
                    
                    # 显示详细触碰信息
                    with st.expander("🔍 MA20触碰详情"):
                        for r in ma20_results[:5]:  # 只显示前5个
                            st.write(f"**{r['symbol']}**: {r['ma20_touch_detail']}")
                
                if other_results and not ma20_touch_only:
                    st.markdown("#### 📊 其他密集信号")
                    other_df = pd.DataFrame(other_results)
                    formatted_other = format_ma_dataframe(other_df, show_ma_values)
                    st.dataframe(formatted_other, use_container_width=True, hide_index=True)
                
                # 下载按钮
                all_results_df = pd.DataFrame(results)
                csv_data = all_results_df.to_csv(index=False)
                st.download_button(
                    label="📥 下载扫描结果 CSV",
                    data=csv_data,
                    file_name=f"ma_density_{timeframe}_{current_time.replace(' ', '_').replace(':', '-')}.csv",
                    mime="text/csv"
                )
                
                # 显示图表
                if show_charts:
                    st.markdown("---")
                    st.markdown("### 📊 数据分析")
                    
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        density_chart = create_density_chart(results)
                        if density_chart:
                            st.plotly_chart(density_chart, use_container_width=True)
                    
                    with chart_col2:
                        ma20_chart = create_ma20_touch_chart(results)
                        if ma20_chart:
                            st.plotly_chart(ma20_chart, use_container_width=True)
                        else:
                            st.info("📊 暂无MA20触碰数据")
                
            else:
                filter_text = "MA20触碰且" if ma20_touch_only else ""
                st.info(f"🤔 当前没有发现{filter_text}密集度 ≤ {density_threshold}% 的均线密集区域")
            
            # 扫描详情
            with st.expander("ℹ️ 扫描详情"):
                st.write(f"**扫描时间**: {current_time}")
                st.write(f"**处理时间**: {scan_stats['scan_time']:.2f} 秒")
                st.write(f"**总交易对数**: {scan_stats['total_symbols']}")
                st.write(f"**成交量过滤后**: {scan_stats['volume_filtered']}")
                st.write(f"**成功处理**: {scan_stats['processed_symbols']}")
                st.write(f"**符合条件**: {scan_stats['results_count']}")
                st.write(f"**数据不足**: {scan_stats['insufficient_data']}")
                
        except Exception as e:
            st.error(f"❌ 扫描过程中发生错误: {str(e)}")
            logger.error(f"扫描错误: {e}")

    # 页脚
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>📊 均线密集度扫描器 Pro - 捕捉密集区突破机会</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
