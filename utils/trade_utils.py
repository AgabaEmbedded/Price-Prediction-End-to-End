import MetaTrader5 as mt5
import time
import os

def manage_ic_markets_scheduled_trade(
    symbol, 
    direction, 
    volume=0.01, 
    magic_number=123456,
    terminal_path=None
):
    """
    Manages active trades on a freshly booted, non-persistent EC2 environment.
    Launches the MT5 terminal if closed and waits for live server synchronization.
    """
    direction = direction.upper()
    if direction not in ['BUY', 'SELL', 'HOLD']:
        print(f"Invalid direction: {direction}")
        return

    #if direction == 'HOLD':
    #    print("Direction is HOLD. Taking no action.")
    #    return

    # 1. Handle initialization and programmatic startup
    init_success = False
    if terminal_path and os.path.exists(terminal_path):
        print(f"Launching MT5 terminal via explicit path: {terminal_path}")
        init_success = mt5.initialize(path=terminal_path)
    else:
        print("Launching MT5 terminal via default registry path...")
        init_success = mt5.initialize()

    if not init_success:
        print(f"MT5 Initialization failed. Error code: {mt5.last_error()}")
        return

    print("Terminal process verified. Waiting 15 seconds for network handshake & sync...")
    time.sleep(15) # Essential buffer time for a cold boot connection to IC Markets

    # 2. Check broker synchronization status
    terminal_info = mt5.terminal_info()
    if terminal_info is None:
        print("Failed to pull terminal specifications.")
        mt5.shutdown()
        return
        
    if not terminal_info.connected:
        print("Terminal is open but offline. Re-attempting connection fallback...")
        time.sleep(10)
        # Refresh terminal status check
        if not mt5.terminal_info().connected:
            print("Critical: Could not establish a connection to IC Markets servers.")
            mt5.shutdown()
            return

    # Ensure symbol visibility 
    mt5.symbol_select(symbol, True)
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Symbol {symbol} is unavailable or misspelled.")
        mt5.shutdown()
        return

    # 3. Position Evaluation and Management Logic
    active_positions = mt5.positions_get(symbol=symbol)
    matching_trade_exists = False

    if active_positions:
        for pos in active_positions:
            if pos.magic != magic_number:
                continue
                
            current_dir = 'BUY' if pos.type == 0 else 'SELL'
            
            if current_dir == direction:
                print(f"Active matching {current_dir} trade found for {symbol}. Keeping open.")
                matching_trade_exists = True
            else:
                print(f"Conflicting position found (Ticket: {pos.ticket}). Sending close request...")
                close_position(pos)
                time.sleep(1) # Brief cooldown for execution acknowledgment

    # 4. Route new market order execution
    if not matching_trade_exists and direction in ['BUY', 'SELL']:
        place_market_order(symbol, direction, volume, magic_number)

    # Clean shut down to free up system resources
    mt5.shutdown()
    print("Execution pipeline finished. Connection gracefully severed.")

def close_position(position):
    tick = mt5.symbol_info_tick(position.symbol)
    order_type = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY
    price = tick.bid if position.type == 0 else tick.ask
    
    close_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": order_type,
        "position": position.ticket,
        "price": price,
        "deviation": 20,
        "magic": position.magic,
        "comment": "Closed via Scheduled Sync",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(close_request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Failed to clear ticket {position.ticket}: code={result.retcode}")
    else:
        print(f"Successfully closed ticket {position.ticket}")

def place_market_order(symbol, direction, volume, magic_number):
    tick = mt5.symbol_info_tick(symbol)
    order_type = mt5.ORDER_TYPE_BUY if direction == 'BUY' else mt5.ORDER_TYPE_SELL
    price = tick.ask if direction == 'BUY' else tick.bid

    order_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "deviation": 20,
        "magic": magic_number,
        "comment": f"Scheduled Execution: {direction}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(order_request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Failed to open {direction} position: code={result.retcode}")
    else:
        print(f"Order filled successfully for {symbol}. Ticket ID: {result.order}")

