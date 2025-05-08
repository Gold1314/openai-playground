import yfinance as yf
from fastmcp import FastMCP
import os
import openai
from dotenv import load_dotenv
load_dotenv()

mcp = FastMCP("stocks")

@mcp.tool()
def fetch_stock_info(symbol: str) -> dict:
    """Get Company's general information."""
    stock = yf.Ticker(symbol)
    return stock.info

@mcp.tool()
def fetch_price_history(symbol: str, period: str = "1y", interval: str = "1mo") -> dict:
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period, interval=interval)
    return hist.reset_index().to_dict(orient="list")

@mcp.tool()
def fetch_quarterly_financials(symbol: str) -> dict:
    """Get stock quarterly financials."""
    stock = yf.Ticker(symbol)
    return stock.quarterly_financials.to_dict()

@mcp.tool()
def fetch_annual_financials(symbol: str) -> dict:
    """Get stock annual financials."""
    stock = yf.Ticker(symbol)
    return stock.financials.T.to_dict()

@mcp.tool()
def fetch_balance_sheet(symbol: str) -> dict:
    stock = yf.Ticker(symbol)
    return stock.balance_sheet.T.to_dict()

@mcp.tool()
def fetch_cash_flow(symbol: str) -> dict:
    stock = yf.Ticker(symbol)
    return stock.cashflow.T.to_dict()

@mcp.tool()
def get_recommendation(symbol: str) -> dict:
    """Get Buy/Hold/Sell recommendation using LLM."""
    import json
    import re
    # Fetch stock info and annual financials
    stock = yf.Ticker(symbol)
    info = stock.info
    annual = stock.financials.T.to_dict()
    # Select only key fields for the prompt
    key_info = {
        "symbol": info.get("symbol"),
        "longName": info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "currentPrice": info.get("currentPrice"),
        "marketCap": info.get("marketCap"),
        "trailingPE": info.get("trailingPE"),
        "revenueGrowth": info.get("revenueGrowth"),
        "dividendYield": info.get("dividendYield"),
        "beta": info.get("beta"),
        "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
        "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
    }
    # For annual, just send a few key metrics for the last 3 years
    annual_summary = {}
    for metric in ["Total Revenue", "Net Income"]:
        if metric in annual:
            years = list(annual[metric].keys())[-3:]
            # Convert keys to strings to avoid serialization errors
            annual_summary[metric] = {str(k): annual[metric][k] for k in years}
    prompt = f"""
    You are a financial analyst. Given the following key stock information and annual financials, provide a one-word recommendation (Buy, Hold, or Sell) and a one-sentence reason.\n\nStock Info: {json.dumps(key_info)}\nAnnual Financials (last 3 years): {json.dumps(annual_summary)}\n\nFormat: <Recommendation> - <Reason>
    """
    # Call OpenAI (new API)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return {"error": "OPENAI_API_KEY not set in environment."}
    try:
        client = openai.OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}]
        )
        rec_text = response.choices[0].message.content
        match = re.match(r'\s*(Buy|Sell|Hold)\s*[-:]?\s*(.*)', rec_text, re.IGNORECASE)
        if match:
            rec = match.group(1).capitalize()
            reason = match.group(2).strip()
        else:
            rec = rec_text.strip()
            reason = ''
        icon = {'Buy': '🟢', 'Sell': '🔴', 'Hold': '🟡'}.get(rec, 'ℹ️')
        return {"recommendation": rec, "reason": reason, "icon": icon}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    mcp.run(transport="stdio")