import os
import sys
import dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

# 1. Initialize the stateless REST client
dotenv.load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    print("Error: Missing Alpaca API Credentials in Environment Variables.")
    sys.exit(1)

# Set paper=True to use your demo/virtual environment
client = TradingClient(api_key=API_KEY, secret_key=SECRET_KEY, paper=True)

def run_trading_cycle(trade_direction: str, qty: float = 0.1):
    print("Connecting to Alpaca Paper REST API...")
    
    # 2. Check for Open Positions (e.g., Are we currently holding ETH?)
    print("\n--- Scanning Active Positions ---")
    positions = client.get_all_positions()
    
    if trading_direction == "BUY":
        if positions:
            if positions[0].side != "long":
                client.close_position("ETH/USD")
                place_market_order(symbol="ETH/USD", qty=qty, side=OrderSide.BUY)
        else:
            place_market_order(symbol="ETH/USD", qty=qty, side=OrderSide.BUY)
            
    elif trading_direction == "SELL":
        if positions:
            if positions[0].side != "short":
                client.close_position("ETH/USD")
                place_market_order(symbol="ETH/USD", qty=qty, side=OrderSide.SELL)
        else:
            place_market_order(symbol="ETH/USD", qty=qty, side=OrderSide.SELL)
    else:
        if positions:
            client.close_position("ETH/USD")

def place_market_order(symbol: str, qty: float, side: OrderSide):
    """Submits a crypto market order."""
    print(f"Submitting Market Order: {side.value} {qty} {symbol}")
    
    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.GTC  # Good 'Til Cancelled for crypto
    )
    
    try:
        order = client.submit_order(order_data)
        print(f"Order successfully submitted! Order ID: {order.id}")
    except Exception as e:
        print(f"Failed to execute order: {e}")

if __name__ == "__main__":
    run_trading_cycle()