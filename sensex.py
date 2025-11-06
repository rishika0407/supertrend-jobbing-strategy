# pending -- 
    # network pitfall 

# all the imports required
import requests
import pandas as pd
import threading
from datetime import datetime
import time
import pyotp
import yaml
import csv
from api_helper import NorenApiPy
from datetime import timedelta
import yfinance as yf

# to stop the warnings to get print in the terminal
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# varibales
start_time        = "09:20:00"      #start time to take the trades
end_time          = "15:20:00"      #end time to take the trades, don't take trades after this
square_off_time   = "15:28:00"      #square-off time, close all the trades that exists
minCandle         = 1               #candle min (user defined)
atr_period        = 10              #window size for atr calculation (user defined)
multiplier        = 3               #multiplier for the atr calculation
capital           = 100000          #capital to consider (user defined)
lotSize           = 20
pnl_log_file      = "15thSEPTSEN.csv"    #file path for logs
strike            = 1               #strike to consider 
profit            = 2               #profit % for each trade (user defined)
loss              = 0               #loss % for each trade (user defined)        
                                    #list to hold the candles and supertrend 
candles           = pd.DataFrame(columns=['Datetime', 'Open', 'Close', 'High', 'Low'])
superTrend        = pd.DataFrame()
                                    #list to hold the trades
positions         = pd.DataFrame(columns=['trading_symbol', 'exchange_token', 'entry_price', 'type', 'strike', 'entry_time', 'lots', 'lots_size'])
index_token       = None            #exchange token for index
api               = NorenApiPy()    #api intialization
cred              = {}              #values to hold the credentials for the api 
instrumentKeys    = []              #instrument keys to subscribe for data on the websocket
callOptions       = pd.DataFrame()  #call-options data
putOptions        = pd.DataFrame()  #put-options data 
instrumentToBuy   = "ATM"  
last_min          = None
buffer_price      = []             #list to hold the candle data before creating the candle 
last_minute       = None

def previous_day_data():
    global candles, minCandle
    # getting the data for previous trading day

    symbol = "^BSESN"
    print(f"Downloading data for {symbol}...")

    # Download 1-min data for last 2 days
    data = yf.download(symbol, period="2d", interval=f"{minCandle}m", group_by='ticker')

    # Fix multi-level columns (flatten)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(1)  # Get 'Close', 'Open', etc.

    # Reset index and convert to IST
    data = data.reset_index()
    data['Datetime'] = data['Datetime'].dt.tz_convert('Asia/Kolkata')

    # Sort and reset index
    data = data.sort_values("Datetime").reset_index(drop=True)

    # Final formatting
    data = data[["Datetime", "Open", "Close", "High", "Low"]]

    candles = pd.concat([candles, data])

def make_request(index, exchange_token):
    global cred, api
    try:
        # getting the credentials using yaml file
        with open('cred.yml', 'r') as file:
            cred = yaml.safe_load(file)
        
        # 2FA for the api
        cred['factor2'] = pyotp.TOTP(cred['gauthkey']).now()
        api.login(userid=cred['user'], password=cred['pwd'], twoFA=cred['factor2'],
                  vendor_code=cred['vc'], api_secret=cred['apikey'], imei=cred['imei'])
        
        # using the get_quotes method to get the current value for the particular token
        resp = api.get_quotes(index, exchange_token)
        
        # return the ltp (last traded price)
        return resp['lp']
    except Exception as e:
        print(f"Error while make quotes function: {e}")
        return None

def getNiftyATM():
    try:
        # getting the nifty spot value using the api
        close_value = float(make_request('BSE', '1'))
        
        # round off to get the atm [NIFTY - strike difference is 50, and that's why we round it off to 50] 
        return round(close_value / 100) * 100
    except Exception as e:
        print(f"Error while getting NIFTY ATM: {e}")
        return None

def gettingInstruments(targetStrike, strikeRange, index, exchange='BSE_FO'):
    global index_token, callOptions, putOptions
    try:
        fileUrl = "https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz"
        # reading the master file from the url
        symboldf = pd.read_csv(fileUrl)
        symboldf["expiry"] = pd.to_datetime(symboldf["expiry"]).dt.date
        
        # filtering the instruments for option and nifty index
        df = symboldf[(symboldf.instrument_type == 'OPTIDX') &
                    (symboldf.tradingsymbol.str.startswith(index)) &
                    (symboldf.exchange == exchange)]
        
        # getting the list of expiry
        expiryList = sorted(df['expiry'].unique().tolist())
        
        # selecting the current expiry
        df2 = df[df.expiry == expiryList[0]].sort_values(by='strike')
        
        # filtering the instruments based on our strike range 
        filteredDf = df2[(df2['strike'] >= targetStrike - strikeRange) &
                        (df2['strike'] <= targetStrike + strikeRange)]
        
        # changing the prefix for infin api (NSE_FO ==> NFO, BSE_FO ==> BFO)
        filteredDf['instrument_key'] = filteredDf['instrument_key'].str.replace("BSE_FO", "BFO")
        
        # filtering and storing it in the dataframes based on call and put option
        callOptions = filteredDf[filteredDf['option_type'] == "CE"]
        putOptions = filteredDf[filteredDf['option_type'] == "PE"]
        
        if index == 'SENSEX':
            # getting the exchange token for index 
            index_token = str(symboldf[symboldf['instrument_key'] == 'BSE_INDEX|SENSEX']['exchange_token'].iloc[0])
            print(index_token)
        return filteredDf['instrument_key'].tolist() + [f'BSE|{index_token}'], callOptions, putOptions
        
    except Exception as e:
        print(f"Error while getting the instruments : {e}")


def process_candle(tick):
    global buffer_price, last_min, candles, superTrend, atr_period

    now = datetime.now()
    current_min = now.strftime("%H:%M")

    if last_min is None:
        last_min = current_min

    # New minute started — create candle
    if current_min != last_min:
        if buffer_price:
            o = buffer_price[0]
            h = max(buffer_price)
            l = min(buffer_price)
            c = buffer_price[-1]
            ts = now - timedelta(minutes=1)

            candles.loc[len(candles)] = [ts, o, c, h, l]
            print(f"New Candle: {ts} | O: {o}, H: {h}, L: {l}, C: {c}")

        # Reset for next minute
        buffer_price = []
        last_min = current_min

        if len(candles) >= atr_period:
            superTrend = calculate_supertrend(candles, atr_period, multiplier)
            print(superTrend)
            
    # Append latest tick
    try:
        price = float(tick.get('lp', 0))
        if price > 0:
            buffer_price.append(price)
    except Exception as e:
        print(f"[ERROR] Tick processing error: {e}")

    
def calculate_supertrend(df, period=10, multiplier=3):
    df = df.copy()
    # True Range (TR)
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    # ATR
    df['ATR'] = 0
    for i in range(0, len(df)):
        if i >= (period-1):
            if i == (period-1):
                df['ATR'] = df['TR'].rolling(window=period).mean()
            else:
                # print(df['ATR'].iloc[i], df['ATR'].iloc[i-1], df['TR'].iloc[i], period)
                df['ATR'].iloc[i] = (df['ATR'].iloc[i-1] * (period - 1) + df['TR'].iloc[i]) / period
    
    # HL2
    hl2 = (df['High'] + df['Low']) / 2
    
    # Basic Bands
    df['Upper Basic'] = hl2 + (multiplier * df['ATR'])
    df['Lower Basic'] = hl2 - (multiplier * df['ATR'])
    
    # Final Bands & Supertrend init
    df['Upper Band'] = 0.0
    df['Lower Band'] = 0.0
    
    # Compute final upper and lower bands
    for i in range(0, len(df)):
        if i < (period-1):
            df.at[i, 'Upper Basic'] = 0.00
            df.at[i, 'Lower Basic'] = 0.00
            df.at[i, 'Upper Band']  = 0.00
            df.at[i, 'Lower Band']  = 0.00
        else:
            df.at[i, 'Upper Band'] = (
                df.at[i, 'Upper Basic']
                if df.at[i, 'Upper Basic'] < df.at[i-1, 'Upper Band'] or df.at[i-1, 'Close'] > df.at[i-1, 'Upper Band']
                else df.at[i-1, 'Upper Band']
            )
            df.at[i, 'Lower Band'] = (
                df.at[i, 'Lower Basic']
                if df.at[i, 'Lower Basic'] > df.at[i-1, 'Lower Band'] or df.at[i-1, 'Close'] < df.at[i-1, 'Lower Band']
                else df.at[i-1, 'Lower Band']
            )
    
# Ensure the Supertrend column exists
    df['Supertrend'] = 0.0

    # Set the Supertrend value
    for i in range(0, len(df)):
        if i < (period - 1):
            df.at[i, 'Supertrend'] = 0.00
        else:
            if (df.at[i-1, 'Supertrend'] == df.at[i-1, 'Upper Band']) and (df.at[i, 'Close'] <= df.at[i, 'Upper Band']):
                df.at[i, 'Supertrend'] = df.at[i, 'Upper Band']
            elif (df.at[i-1, 'Supertrend'] == df.at[i-1, 'Upper Band']) and (df.at[i, 'Close'] >= df.at[i, 'Upper Band']):
                df.at[i, 'Supertrend'] = df.at[i, 'Lower Band']
            elif (df.at[i-1, 'Supertrend'] == df.at[i-1, 'Lower Band']) and (df.at[i, 'Close'] >= df.at[i, 'Lower Band']):
                df.at[i, 'Supertrend'] = df.at[i, 'Lower Band']
            elif (df.at[i-1, 'Supertrend'] == df.at[i-1, 'Lower Band']) and (df.at[i, 'Close'] <= df.at[i, 'Lower Band']):
                df.at[i, 'Supertrend'] = df.at[i, 'Upper Band']
            else:
                df.at[i, 'Supertrend'] = 0.00

    # Clean up (optional)
    # df.drop(columns=['H-L', 'H-PC', 'L-PC', 'Upper Basic', 'Lower Basic', 'Upper Band', 'Lower Band'], inplace=True)

    return df
   
def super_trend_signal():
    global superTrend

    # getting the current and previous close price
    curr = superTrend.iloc[-1]
    prev = superTrend.iloc[-2]
    
    # building the logic
    if curr['Close'] > curr['Supertrend'] and prev['Close'] <= prev['Supertrend']:
        return "Buy Call"
    elif curr['Close'] < curr['Supertrend'] and prev['Close'] >= prev['Supertrend']:
        return "Buy Put"
    return None

def place_position(option_df, option_type, strike_input):
    global positions, capital, lotSize
    
    atm = getNiftyATM()

    # Determine strike offset based on input
    try:
        if strike_input == "ATM":
            strike_offset = 0
        elif "ITM" in strike_input:
            distance = int(strike_input.replace("ITM", ""))
            strike_offset = -100 * distance if option_type == "CE" else 100 * distance
        elif "OTM" in strike_input:
            distance = int(strike_input.replace("OTM", ""))
            strike_offset = 100 * distance if option_type == "CE" else -100 * distance
        else:
            print(f"[ERROR] Invalid strike input: {strike_input}")
            return
    except Exception as e:
        print(f"[ERROR] Could not parse strike input '{strike_input}': {e}")
        return

    strike = atm + strike_offset
    opt = option_df[option_df['strike'] == strike]

    if opt.empty:
        print(f"[WARN] No option found for strike {strike}")
        return

    try:
        opt_row = opt.iloc[0]
        token = opt_row['exchange_token']
        trading_symbol = opt_row['tradingsymbol']
        
        entry_price = float(make_request("BFO", str(token)))
        cost_per_lot = entry_price * lotSize
        max_lots = int(capital // cost_per_lot)

        if max_lots < 1:
            print(f"[SKIPPED] 1 lot of {trading_symbol} costs ₹{cost_per_lot:.2f} > capital ₹{capital}")
            return
        
        total_cost = cost_per_lot * max_lots
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        positions.loc[len(positions)] = [
            trading_symbol, token, entry_price, option_type, strike, timestamp, max_lots, lotSize
        ]

        print(positions)
        print(f"[ENTRY] {option_type} | Symbol: {trading_symbol} | Strike: {strike} | Entry: ₹{entry_price:.2f} | Lots: {max_lots} | Total Cost: ₹{total_cost:.2f} | Time: {timestamp}")
        
    except Exception as e:
        print(f"[ERROR] Failed to place order: {e}")


def check_and_exit_positions(message):
    global positions, lotSize, profit, pnl_log_file, square_off_time

    try:
        token_int = int(message['tk'])

        # Check if token exists in positions
        if token_int not in list(positions['exchange_token']):
            return

        # Get the matching row and index
        row = positions[positions['exchange_token'] == token_int].iloc[0]
        idx = positions.index[positions['exchange_token'] == token_int][0]

        # Parse necessary info
        token = row['exchange_token']
        trading_symbol = row['trading_symbol']
        entry_price = float(row['entry_price'])
        lots = int(row['lots'])  # number of lots held
        total_qty = lots * lotSize  # total quantity

        # Get LTP
        try:
            ltp = float(message.get('lp'))
        except:
            ltp = float(make_request("BFO", str(token)))

        # Calculate PnL
        pnl_per_unit = ltp - entry_price
        total_pnl = pnl_per_unit * total_qty

        # Exit time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"{token} PnL: ₹{total_pnl:.2f} | Entry: {entry_price} | LTP: {ltp} | Lots: {lots}")

        # Exit if profit target met
        if (pnl_per_unit >= (profit / 100) * entry_price) or (datetime.now().strftime("%H:%M:%S") == square_off_time):
            # Log to CSV
            with open(pnl_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    trading_symbol,
                    token,
                    row['type'],
                    row['strike'],
                    row['entry_time'],
                    entry_price,
                    row['lots'], 
                    row['lots_size'],
                    ltp,
                    timestamp,
                    total_pnl
                ])

            print(f"[EXIT] {row['type']} | Symbol: {trading_symbol} | Entry: ₹{entry_price} | Exit: ₹{ltp} | PnL: ₹{total_pnl:.2f} | Time: {timestamp}")
            
            # Drop exited position
            positions.drop(index=idx, inplace=True)
            
    except Exception as e:
        print(f"[ERROR] Failed in check_and_exit_positions: {e}")

def event_handler_order_update(message):
    print("order event: " + str(message))

def event_handler_quote_update(message):
    global index_token, callOptions, putOptions, superTrend, instrumentToBuy, start_time, end_time, last_minute
    # if it is the index token then update candle data
    if message['tk'] == str(index_token):
        process_candle(message)
    else:
        pass
    # if there are no positions
    if positions.empty:
        
        # checking the length of supertrend to be calculated [it should be greater than the atr period]
        if len(superTrend) > atr_period + 1:
            # getting the signal for supertrend 
            signal = super_trend_signal()
            current_minute = datetime.now().strftime("%Y-%m-%d %H:%M")
            if (datetime.now().strftime("%H:%M:%S") >= start_time) and (datetime.now().strftime("%H:%M:%S") < end_time): 
                if current_minute != last_minute:
                    if signal == "Buy Call":
                        
                        # creating the positions according to the signal
                        place_position(callOptions, "Call", instrumentToBuy)
                        last_minute = current_minute
                    elif signal == "Buy Put":
                        
                        # creating the positions according to the signal
                        place_position(putOptions, "Put", instrumentToBuy)
                        last_minute = current_minute
        
        else:
            pass
    else:
        
        # if position is not empty then keep the track of the positions
        check_and_exit_positions(message)

def open_callback():
    global socket_opened
    socket_opened = True
    print('app is connected')
    api.subscribe(instrumentKeys, feed_type='d')

def main():
    global instrumentKeys, callOptions, putOptions
    # getting the atm 
    atm = getNiftyATM()
    print(f"ATM : {atm}")
    
    # getting the instruments, callOptions and the putOptions
    instrumentKeys, callOptions, putOptions = gettingInstruments(atm, strikeRange=800, index='SENSEX', exchange='BSE_FO')
    print(instrumentKeys)
    # first writing the headers for the csv file 
    with open(pnl_log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Trading Symbol", "Exchange Token", "Type", "Strike", "Entry Time", "Entry Price", "Lots", "Lot Size", "LTP", "Exit Time", "PNL" ])
   
    # api login 
    ret = api.login(userid=cred['user'], password=cred['pwd'], twoFA=cred['factor2'],
                    vendor_code=cred['vc'], api_secret=cred['apikey'], imei=cred['imei'])
    
    previous_day_data()
    # starting the websocket
    while True:
        if ret:
            ret = api.start_websocket(order_update_callback=event_handler_order_update,
                                      subscribe_callback=event_handler_quote_update,
                                      socket_open_callback=open_callback)
            while True:
                pass

if __name__ == "__main__":
    main()
