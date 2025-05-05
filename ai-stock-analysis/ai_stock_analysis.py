import os
import asyncio

import streamlit as st
import yfinance as yf

from openai import OpenAI
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings
from dotenv import load_dotenv, find_dotenv

# point explicitly at .env in your project root
dotenv_path = find_dotenv('.env', usecwd=True)
print("🔍 .env found at:", dotenv_path)

load_dotenv(dotenv_path, override=True)

raw = os.getenv("OPENAI_API_KEY")

load_dotenv()
# 1) Load your key, preferably via Streamlit secrets (or fallback to ENV)
#    Create a file at ~/.streamlit/secrets.toml with:
#      [openai]
#      api_key = "sk-XXX..."
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("🚨 OpenAI API key not found! Add it to ~/.streamlit/secrets.toml or set $OPENAI_API_KEY.")
    st.stop()

# 2) Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# 3) Define your Agent exactly as before
agent = Agent(
    "openai:gpt-4",
    deps_type=str,
    client=client,
    model_settings=ModelSettings(temperature=0),
    system_prompt="""
You are an advanced AI stock rating agent designed to analyze financial reports, 
historical price data, and key technical indicators to evaluate stocks. 
Your goal is to assign a rating to each stock based on a scale from Strong Buy (A) 
to Strong Sell (E) and provide a clear explanation for your rating.

Consider the following factors:
- Revenue growth 
- Profitability 
- Price history trends
- Technical indicators 

After the analysis, assign one of the following ratings and provide a detailed explanation for the rating:
A - Strong Buy: The stock is undervalued with strong growth potential, solid financials, and positive market momentum.
B - Buy: The stock has good fundamentals and technical indicators but may have some risks or uncertainties.
C - Hold: The stock is fairly valued with mixed signals from fundamental and technical analysis. Holding is advised unless major catalysts emerge.
D - Sell: The stock shows weak fundamentals, declining trends, or negative market sentiment, suggesting downside risk.
E - Strong Sell: The stock has serious financial or structural problems, significant downside risk, or bearish trends indicating potential losses.
"""
)

# 4) Your tools
@agent.tool
def fetch_stock_info(ctx: RunContext[str]):
    info = yf.Ticker(ctx.deps).info
    return {
        "longName": info.get("longName"),
        "marketCap": info.get("marketCap"),
        "sector": info.get("sector"),
    }

@agent.tool
def fetch_quarterly_financials(ctx: RunContext[str]):
    df = yf.Ticker(ctx.deps).quarterly_financials.T
    return df[["Total Revenue", "Net Income"]].to_csv()

@agent.tool
def fetch_annual_financials(ctx: RunContext[str]):
    df = yf.Ticker(ctx.deps).financials.T
    return df[["Total Revenue", "Net Income"]].to_csv()

@agent.tool
def fetch_weekly_price_history(ctx: RunContext[str]):
    df = yf.Ticker(ctx.deps).history(period="1y", interval="1wk")
    return df.to_csv()

@agent.tool
def calculate_rsi_weekly(ctx: RunContext[str]):
    stock = yf.Ticker(ctx.deps)
    data = stock.history(period='1y', interval='1wk')
    delta = data['Close'].diff()

    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

symbol = st.selectbox('Please select a stock symbol', ['AAPL', 'TSLA', 'OXY'])
result = agent.run_sync("Analyze this stock", deps=symbol)
st.markdown(result.data)
