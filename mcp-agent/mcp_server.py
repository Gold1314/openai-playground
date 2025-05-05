import yfinance as yf
from fastmcp import FastMCP

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

if __name__ == "__main__":
    mcp.run(transport="stdio")