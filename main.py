import pandas as pd
import requests as rq
import yfinance as yf
import numpy as np
import time
import math
from tvDatafeed import TvDatafeed, Interval
import sys

PROGRESS_BAR_LENGTH = 40

MAX_NUM_DATES = 5000 # number of rows allocated for storing price data (past number of dates)
NUM_DATES_TV = 4000 # number of bars (may have holes) to download from TV

#username = 'your_username'
#password = 'your_password'
tv = TvDatafeed()

# default globals set upon reading data
loadedTickers = []
numSymbols = 0
numDates = 0
o = []
h = []
l = []
c = []
v = []
scores = []
numScores = 0
sortDescending = True
dates = []
dateToIndex = {}
symbolToIndex = {}

serial = ""
serialPos = 0

VERY_LOW = -2000000000
VERY_HIGH = 2000000000

# backtesting globals
endingDateIndex = 0
startingDateIndex = 0
buyNSymbols = 1
maxOutlierAllowed = 1000000
pctTradeLeverage = 100

nasdaq_url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/refs/heads/main/nasdaq/nasdaq_tickers.txt"
nyse_url   = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/refs/heads/main/nyse/nyse_tickers.txt"
amex_url   = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/refs/heads/main/amex/amex_tickers.txt"



def writeSymbolData(ticker: str, k: int):
    
    s = [ticker + "\n"]
    for i in range(len(o)):
        s.append(str(o[i][k]) + "\n")
        s.append(str(h[i][k]) + "\n")
        s.append(str(l[i][k]) + "\n")
        s.append(str(c[i][k]) + "\n")
        s.append(str(v[i][k]) + "\n")
    
    with open("symbolData.txt", "a") as file:
        file.write("".join(s))

def readSymbolData():
    print("Loading data...\n")

    lines = []
    with open("symbolData.txt", "r") as file:
        for line in file:
            lines.append(line)
    x = 0

    global dates, numDates, numSymbols, loadedTickers, o, h, l, c, v

    loadedTickers.clear()
    o.clear()
    h.clear()
    l.clear()
    c.clear()
    v.clear()
    for i in range(numDates):
        o.append([])
        h.append([])
        l.append([])
        c.append([])
        v.append([])
    
    i = 0
    while(True):
        loadedTickers.append(lines[x])
        x += 1
        symbolToIndex.update({loadedTickers[i]: i})

        for j in range(numDates):
            o[j].append(float(lines[x]))
            x += 1
            h[j].append(float(lines[x]))
            x += 1
            l[j].append(float(lines[x]))
            x += 1
            c[j].append(float(lines[x]))
            x += 1
            v[j].append(float(lines[x]))
            x += 1

        i += 1
        
        if(x >= len(lines)):
            break

    numSymbols = len(loadedTickers)
    

def is_valid_ticker(ticker: str):
    try:
        data = yf.Ticker(ticker)
        info = data.fast_info
        return info is not None
    except Exception:
        return False
    
def getNASDAQData():
    response = rq.get(nasdaq_url)
    nasdaq = response.text.splitlines()

    print(f"NASDAQ tickers found: {len(nasdaq)}")
    
    tickers = []
    for t in nasdaq:
        if(is_valid_ticker(t)):
            tickers.append(t)

    getSomeDataYF(tickers)
    
def getNYSEData():
    response = rq.get(nyse_url)
    nyse = response.text.splitlines()

    print(f"NYSE tickers found: {len(nyse)}")
    
    tickers = []
    for t in nyse:
        if(is_valid_ticker(t)):
            tickers.append(t)

    getSomeDataYF(tickers)
    
def getAMEXData():
    response = rq.get(amex_url)
    amex = response.text.splitlines()

    print(f"AMEX tickers found: {len(amex)}")
    
    tickers = []
    for t in amex:
        if(is_valid_ticker(t)):
            tickers.append(t)

    getSomeDataYF(tickers)
    
def getAllDataYF():

    response = rq.get(nasdaq_url)
    nasdaq = response.text.splitlines()
    response = rq.get(nyse_url)
    nyse = response.text.splitlines()
    response = rq.get(amex_url)
    amex = response.text.splitlines()

    print(f"Total tickers found: {len(nasdaq)+len(nyse)+len(amex)}")
    print("\tNASDAQ: " + str(len(nasdaq)))
    print("\tNYSE: " + str(len(nyse)))
    print("\tAMEX: " + str(len(amex)) + "\n")

    tickers = []
    for t in nasdaq:
        if(is_valid_ticker(t)):
            tickers.append(t)
    for t in nyse:
        if(is_valid_ticker(t)):
            tickers.append(t)
    for t in amex:
        if(is_valid_ticker(t)):
            tickers.append(t)

    getSomeDataYF(tickers)

def initOHLCV(d, s):
    o.clear()
    h.clear()
    l.clear()
    c.clear()
    v.clear()
    for i in range(d):
        o.append([])
        h.append([])
        l.append([])
        c.append([])
        v.append([])
        for j in range(s):
            o[i].append(math.nan)
            h[i].append(math.nan)
            l[i].append(math.nan)
            c[i].append(math.nan)
            v[i].append(math.nan)
    

def getSomeDataYF(tickers: list[str]):

    df = yf.download(tickers, period='10y')
    print("\n")
    #df = df.dropna(axis=1, how='all')
    #df = df.dropna(axis=0, how='all')

    d = df.to_dict(orient='split')

    global numDates, numSymbols, loadedTickers, o, h, l, c, v, dates, dateToIndex, symbolToIndex
    rawDates = d.get('index')
    symbols = d.get('columns')
    numSymbols = int(len(symbols) / 5)

    for i in range(numSymbols):
        loadedTickers.append(symbols[i][1])
        symbolToIndex.update({symbols[i][1]: i})
    
    # insert the data in reverse date order
    initOHLCV(MAX_NUM_DATES, numSymbols)
        
    rawData = d.get('data')
    for i in range(len(rawDates)):
        date = str(rawDates[i])[0:10]
        index = dateToIndex.get(date, -1)
        if(index == -1):
            continue
        else:
            for j in range(numSymbols):
                c[index][j] = rawData[i][j]
            for j in range(numSymbols):
                h[index][j] = rawData[i][j + numSymbols]
            for j in range(numSymbols):
                l[index][j] = rawData[i][j + numSymbols * 2]
            for j in range(numSymbols):
                o[index][j] = rawData[i][j + numSymbols * 3]
            for j in range(numSymbols):
                v[index][j] = rawData[i][j + numSymbols * 4]
    
    # (maybe later) remove dates (rows) and symbols (columns) that were not filled

    # fill in price and volume holes
    for i in range(numSymbols):
        prevClose = o[numDates - 1][i]
        for j in range(numDates - 1, -1, -1):
            if(o[j][i] == math.nan):
                o[j][i] = prevClose
            else:
                prevClose = o[j][i]
            if(h[j][i] == math.nan):
                h[j][i] = prevClose
            if(l[j][i] == math.nan):
                l[j][i] = prevClose
            if(c[j][i] == math.nan):
                c[j][i] = prevClose
            else:
                prevClose = c[j][i]
            if(v[j][i] == math.nan):
                v[j][i] = 0
    
    for i in range(numSymbols):
        writeSymbolData(loadedTickers[i], i)

def getDatesList():
    global dates
    dates.clear()
    df = tv.get_hist(
        symbol = 'AAPL',
        exchange = 'NASDAQ',
        interval = Interval.in_daily,
        n_bars = 5000
    )
    #df = df.dropna(axis=1, how='all')
    #df = df.dropna(axis=0, how='all')

    if(df is None):
        print("Could not get list of dates.")
        return
    else:
        
        d = df.to_dict(orient='split')

        rawDates = d.get('index')
        for j in range(len(rawDates)):
            date = str(rawDates[j])[0:10]
            dates.append(date)

        list.reverse(dates)

        text = "\n".join(dates)
        with open("dates.txt", "w") as file:
            file.write(text)
        
        x = 0
        for i in dates:
            dateToIndex.update({i: x})
            x += 1

        global numDates
        numDates = len(dates)

def readDates():
    global dates
    dates.clear()
    with open("dates.txt", "r") as file:
        for line in file:
            l = line.strip()
            if(len(l) == 10):
                dates.append(l)

    x = 0
    for i in dates:
        dateToIndex.update({i: x})
        x += 1
    
    global numDates
    numDates = len(dates)

def getAllDataTV():
    response = rq.get(nasdaq_url)
    nasdaq = response.text.splitlines()
    response = rq.get(nyse_url)
    nyse = response.text.splitlines()
    response = rq.get(amex_url)
    amex = response.text.splitlines()

    total = len(nasdaq) + len(nyse) + len(amex)
    print(f"Total tickers found: " + str(total))
    print("\tNASDAQ: " + str(len(nasdaq)))
    print("\tNYSE: " + str(len(nyse)))
    print("\tAMEX: " + str(len(amex)) + "\n")

    tickers = []
    exchanges = []
    for t in nasdaq:
        tickers.append(t)
        exchanges.append('NASDAQ')
    for t in nyse:
        tickers.append(t)
        exchanges.append('NYSE')
    for t in amex:
        tickers.append(t)
        exchanges.append('AMEX')
    
    invalid = True
    while(invalid):
        invalid = False
        print("Type a number from 0 to " + str(total - 1) + " to start at that symbol index: ")
        i = getPosIntInput(0, total - 1)
        match(i):
            case -2 | -1:
                invalid = True
            case _:
                getSomeDataTV(tickers, exchanges, i)


def getSomeDataTV(tickers: list[str], exchanges: list[str], start: int):

    global numDates, numSymbols, loadedTickers, o, h, l, c, v
    print("Getting data from " + str(len(tickers)) + " symbols from TradingView...\n")
    
    numDates = 0
    maxNumSymbols = len(tickers)
    initOHLCV(MAX_NUM_DATES, maxNumSymbols)
    
    if(start == 0):
        with open("symbolData.txt", "w") as file:
            file.write("")
    
    k = 0
    for i in range(start, maxNumSymbols):
        
        percent = i / maxNumSymbols
        filled_length = int(PROGRESS_BAR_LENGTH * percent)
        bar = '#' * filled_length + '-' * (PROGRESS_BAR_LENGTH - filled_length)
        sys.stdout.write(f'\rDownloading data... |{bar}| {percent:.1%} ({i}/{maxNumSymbols} done, getting ' + tickers[i] + ')')
        sys.stdout.flush()

        global tv
        if(i % 100 == 0):
            tv = TvDatafeed()

        df = tv.get_hist(
            symbol = tickers[i],
            exchange = exchanges[i],
            interval = Interval.in_daily,
            n_bars = NUM_DATES_TV
        )
        #df = df.dropna(axis=1, how='all')
        #df = df.dropna(axis=0, how='all')
        
        if(df is None):
            pass # maybe add stuff here if necessary
        else:
            ticker = tickers[i]
            symbolToIndex.update({ticker: k})
            
            d = df.to_dict(orient='split')

            rawDates = d.get('index')
            
            loadedTickers.append(tickers[i])
            
            # insert the data in reverse date order
            rawData = d.get('data')
            for j in range(len(rawDates)):
                date = str(rawDates[j])[0:10]
                index = dateToIndex.get(date, -1)
                numDates = max(numDates, index + 1)
                if(index == -1):
                    continue
                else:
                    o[index][k] = rawData[j][1]
                    h[index][k] = rawData[j][2]
                    l[index][k] = rawData[j][3]
                    c[index][k] = rawData[j][4]
                    v[index][k] = rawData[j][5]# fill in price and volume holes

            # fill in holes in the data
            prevClose = o[numDates - 1][i]
            for j in range(numDates - 1, -1, -1):
                if(o[j][i] == math.nan):
                    o[j][i] = prevClose
                else:
                    prevClose = o[j][i]
                if(h[j][i] == math.nan):
                    h[j][i] = prevClose
                if(l[j][i] == math.nan):
                    l[j][i] = prevClose
                if(c[j][i] == math.nan):
                    c[j][i] = prevClose
                else:
                    prevClose = c[j][i]
                if(v[j][i] == math.nan):
                    v[j][i] = 0

            writeSymbolData(tickers[i], k)
            k += 1
    
    numSymbols = k

    if(numDates == 0 or numSymbols == 0):
        print("No data was received.")
        return

    

def gap(dateIndex: int, symbolIndex: int, back = 1):
    x = c[dateIndex + back][symbolIndex]
    y = o[dateIndex][symbolIndex]
    if(inRange(x) and inRange(y)):
        return y - x
    else:
        return None

def change(dateIndex: int, symbolIndex: int, back = 0):
    x = o[dateIndex + back][symbolIndex]
    y = c[dateIndex][symbolIndex]
    if(inRange(x) and inRange(y)):
        return y - x
    else:
        return None

def gap24(dateIndex: int, symbolIndex: int, back = 1):
    x = o[dateIndex + back][symbolIndex]
    y = o[dateIndex][symbolIndex]
    if(inRange(x) and inRange(y)):
        return y - x
    else:
        return None

def change24(dateIndex: int, symbolIndex: int, back = 1):
    x = c[dateIndex + back][symbolIndex]
    y = c[dateIndex][symbolIndex]
    if(inRange(x) and inRange(y)):
        return y - x
    else:
        return None

def gapP(dateIndex: int, symbolIndex: int, back = 1):
    x = c[dateIndex + back][symbolIndex]
    y = o[dateIndex][symbolIndex]
    if(inRange(x) and inRange(y)):
        if(x == 0):
            return 0
        return (y - x) / x
    else:
        return None

def changeP(dateIndex: int, symbolIndex: int, back = 0):
    x = o[dateIndex + back][symbolIndex]
    y = c[dateIndex][symbolIndex]
    if(inRange(x) and inRange(y)):
        if(x == 0):
            return 0
        return (y - x) / x
    else:
        return None

def gap24P(dateIndex: int, symbolIndex: int, back = 1):
    x = o[dateIndex + back][symbolIndex]
    y = o[dateIndex][symbolIndex]
    if(inRange(x) and inRange(y)):
        if(x == 0):
            return 0
        return (y - x) / x
    else:
        return None

def change24P(dateIndex: int, symbolIndex: int, back = 1):
    x = c[dateIndex + back][symbolIndex]
    y = c[dateIndex][symbolIndex]
    if(inRange(x) and inRange(y)):
        if(x == 0):
            return 0
        return (y - x) / x
    else:
        return None

def mean(dateIndex: int, symbolIndex: int, n: int):
    mean = 0
    for i in range(n):
        x = c[dateIndex + i][symbolIndex]
        if(inRange(x)):
            mean += x
        else:
            return VERY_LOW
    mean /= n
    return mean

def sd(dateIndex: int, symbolIndex: int, n: int):
    sd = 0
    for i in range(n):
        x = c[dateIndex + i][symbolIndex]
        if(inRange(x)):
            sd += (x - mean) * (x - mean)
        else:
            return VERY_LOW
    sd /= n
    sd = math.sqrt(sd)
    return sd

def greens(dateIndex: int, symbolIndex: int, n: int):
    c = 0
    for i in range(n):
        ch = change24P(dateIndex + i, symbolIndex)
        if(ch is None):
            return VERY_LOW
        else:
            if(ch > 0):
                c += 1
    return c

def reds(dateIndex: int, symbolIndex: int, n: int):
    c = 0
    for i in range(n):
        ch = change24P(dateIndex + i, symbolIndex)
        if(ch is None):
            return VERY_LOW
        else:
            if(ch < 0):
                c += 1
    return c

def distance(dateIndex: int, symbolIndex: int, n: int):
    d = change24(dateIndex, symbolIndex, n)
    if(d is None):
        return VERY_LOW
    d = abs(d)
    return d

def walk(dateIndex: int, symbolIndex: int, n: int):
    walk = 0
    for i in range(n):
        x = change24(dateIndex + i, symbolIndex)
        if(x is None):
            return VERY_LOW
        walk += abs(x)
    return walk


def getScore(dateIndex: int, symbolIndex: int):

    score = 0
    n = 30

    op = o[dateIndex][symbolIndex]
    hi = h[dateIndex][symbolIndex]
    lo = l[dateIndex][symbolIndex]
    cl = c[dateIndex][symbolIndex]
    vo = v[dateIndex][symbolIndex]
    
    if(vo == math.nan or vo < 500000):
        return VERY_LOW

    if(inRange(op) and inRange(hi) and inRange(lo) and inRange(cl)):
        if(hi - lo == 0):
            return VERY_LOW
        return (hi - cl) / (hi - lo)
    else:
        return VERY_LOW

def inRange(x: float):
    return x > VERY_LOW and x < VERY_HIGH

def getAllScores(dateIndex: int):
    global scores
    scores = []
    for i in range(numSymbols):
        score = getScore(dateIndex, i)
        if(inRange(score)):
            scores.append([loadedTickers[i], score, i])
    global numScores
    numScores = len(scores)

def sortVal(l: list[int]):
    return l[1]

def sortScores():
    global scores
    scores.sort(key=sortVal, reverse=sortDescending)

def printNScores(n: int):
    if(n > numScores):
        print("Cannot print more symbol scores (" + str(n) +
              ") than the number of scores (" + str(numScores) + ").")
    else:
        print("-SYMBOL-  -SCORE-")
        for i in range(n):
            print(str(scores[i][0]).ljust(10) + f"{float(scores[i][1]):.6f}")
        print(str(n) + " symbol scores printed.")

def getCustomData():
    print("Enter each ticker symbol on a separate line with no other characters.")
    print("Enter a blank line to finish.")
    
    tickers = []
    ticker = input()
    while(ticker != ''):
        tickers.append(ticker)
        ticker = input()

    print(f"Total tickers entered: {len(tickers)}")
    getSomeDataYF(tickers)

def backtest():

    while(1):
        print("Type the starting date index from 0 to " + str(numDates - 1) + ": ")
        startingDateIndex = getPosIntInput(0, numDates - 1)
        if(startingDateIndex >= 0):
            break
            
    while(1):
        print("Type the ending date index from 0 to " + str(startingDateIndex) + ": ")
        endingDateIndex = getPosIntInput(0, startingDateIndex)
        if(endingDateIndex >= 0):
            break
    
    while(1):
        print("Type the number of symbols to buy each day from 1 to " + str(numSymbols) + ": ")
        buyNSymbols = getPosIntInput(1, numSymbols)
        if(buyNSymbols >= 0):
            break
    
    while(1):
        print("Type the percent of balance to trade with from 0 to 100: ")
        pctTradeLeverage = getPosIntInput(0, 100)
        if(pctTradeLeverage >= 0):
            break
    
    while(1):
        print("Type the maximum allowed outlying result (in percent) from 1 to 1000000: ")
        maxOutlierAllowed = getPosIntInput(1, 1000000)
        if(maxOutlierAllowed >= 0):
            break

    daysTraded = startingDateIndex - endingDateIndex + 1
    trades = 0
    wins = 0
    losses = 0
    ties = 0
    won = 0
    lost = 0
    total = 0
    minResult = VERY_HIGH
    meanResult = 0
    maxResult = VERY_LOW
    balance = 1
    minScoreTraded = VERY_HIGH
    meanScoreTraded = 0
    maxScoreTraded = VERY_LOW
    outliers = 0
    var = 0
    defined = 0
    scoresDefined = 0
    definedProportion = 0
    scoresDefinedProportion = 0
    profitFactor = 0
    sd = 0

    for date in range(startingDateIndex, endingDateIndex - 1, -1):
        # simulate trading on this day
        getAllScores(date) # you can either use the score from the previous day or get trade results using the next day to avoid overlap.
        sortScores()
        #print(c[date])
        #print(scores)
        numTraded = min(numScores, buyNSymbols)
        for symbolIndex in range(numTraded):
            # simulate trading this symbol
            score = scores[symbolIndex][1]
            if(score > VERY_LOW and score < VERY_HIGH):
                scoresDefined += 1
            maxScoreTraded = max(maxScoreTraded, score)
            minScoreTraded = min(minScoreTraded, score)
            meanScoreTraded += score

            entry = c[date][scores[symbolIndex][2]]
            exit = c[date - 1][scores[symbolIndex][2]]
            result = 0
            if(inRange(entry) and inRange(exit)):
                if(entry != 0):
                    result = (exit - entry) / entry
                defined += 1

            if(result > maxOutlierAllowed / 100):
                #print(str(dates[date]) + " " + str(scores[symbolIndex]) + " " + str(result) + " " + str(c[date][scores[symbolIndex][2]]) + " " + str(c[date - 1][scores[symbolIndex][2]]))
                result = 0
                outliers += 1
            
            if(result > 0):
                wins += 1
                won += result
            elif(result < 0):
                losses += 1
                lost += result
            else:
                ties += 1

            trades += 1
            minResult = min(minResult, result)
            maxResult = max(maxResult, result)
            total += result
            var += result * result
            balance *= (1 + result) * (pctTradeLeverage / 100)

    if(trades != 0):
        meanScoreTraded /= trades
        var /= trades
        definedProportion = defined / trades
        scoresDefinedProportion = scoresDefined / trades

    if(lost != 0):
        profitFactor = -1 * won / lost

    sd = math.sqrt(var)
    sr = 0
    if(sd != 0):
        sr = total / sd
    
    print("\nBACKTESTING RESULTS from date index " + str(startingDateIndex) + " to " + str(endingDateIndex) + " (" + str(daysTraded) + " days), " + str(numSymbols) + " total symbols:")
    print("Trades: " + str(trades))
    print("Trades with defined results: " + str(defined) + "/" + str(trades) + " = " + str(definedProportion))
    print("Trades with defined scores: " + str(scoresDefined) + "/" + str(trades) + " = " + str(scoresDefinedProportion))
    print("W/L/T: " + str(wins) + "/" + str(losses) + "/" + str(ties))
    print("Won/Lost/Net: " + "{:.6f}".format(won) + "/" + "{:.6f}".format(lost) + "/" + "{:.6f}".format(total))
    print("Outliers: " + str(outliers))
    print("Var/SD/SR: " + "{:.6f}".format(var) + "/" + "{:.6f}".format(sd) + "/" + "{:.6f}".format(sr))
    print("Profit Factor: " + "{:.6f}".format(profitFactor))
    print("Final Balance: " + "{:.6f}".format(balance))
    
    if(trades != 0):
        meanResult = total / trades
    print("Min/mean/max result: " + "{:.6f}".format(minResult) + "/" + "{:.6f}".format(meanResult) + "/" + "{:.6f}".format(maxResult))
    tpd = 0
    if(daysTraded != 0):
        tpd = trades / daysTraded
    print("Mean trades per day: " + "{:.6f}".format(tpd))
    print("Min/mean/max score traded: " + "{:.6f}".format(minScoreTraded) + "/" +
          "{:.6f}".format(meanScoreTraded) + "/" + "{:.6f}".format(maxScoreTraded))
    print("")


def getPosIntInput(lower: int, upper: int):
    s = input()
    if(s == ''):
        return -1
    if(len(s) > 8):
        return -2
    flag = True
    for ch in s:
        if(ch < '0' or ch > '9'):
            flag = False
            break
    if(flag):
        if(int(s) < lower or int(s) > upper):
            return -2
        else:
            return int(s)
    else:
        return -1

def getInput():
    print("Type")
    print("    a blank line to use all symbols from built-in (or recently downloaded) data,")
    print("    (Not up-to-date unless you have already downloaded data, in which case this data will be read)")
    print("    T to download all symbols from TV (very slow but reliable data),")
    print("    F to download all symbols from YF (fast but limited data),")
    print("    N to download NASDAQ symbols,")
    print("    Y to download NYSE symbols,")
    print("    A to download AMEX symbols,")
    print("    C to download your own custom symbols.\n")

    symbolList = input()
    match(symbolList):
        case 'T' | 't':
            #getSomeDataTV(
            #    ['AAPL', 'ABUS', 'PRAX', 'NVTS', 'SSII', 'MVST', 'IRON', 'CRML', 'PGEN', 'FLNC', 'PZZA', 'TNGX', 'POWI', 'INDI', 'ATAI', 'ESTA', 'DYN', 'BTDR', 'PTGX', 'JBHT', 'ANDE', 'BITF'],
            #    ['NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ']
            #)
            getAllDataTV()
        case 'F' | 'f':
            #getSomeDataYF(
            #    ['AAPL', 'ABUS', 'PRAX', 'NVTS', 'SSII', 'MVST', 'IRON', 'CRML', 'PGEN', 'FLNC', 'PZZA', 'TNGX', 'POWI', 'INDI', 'ATAI', 'ESTA', 'DYN', 'BTDR', 'PTGX', 'JBHT', 'ANDE', 'BITF'],
            #)
            getAllDataYF()
        case 'N' | 'n':
            getNASDAQData()
        case 'Y' | 'y':
            getNYSEData()
        case 'A' | 'a':
            getAMEXData()
        case 'C' | 'c':
            getCustomData()
        case _:
            readSymbolData()
    
    #print("Type a symbol score formula code:")
    # scoreFormula = input()
    
    print("Type A to sort the scores ascendingly or anything else to sort descendingly:")
    sortOrderString = input()
    global sortDescending
    sortDescending = sortOrderString != 'a' and sortOrderString != 'A'

    while(True):
        invalid = True
        while(invalid):
            invalid = False
            print("Type a number from 0 to " + str(numDates - 1) + " to print the scores for that date index or anything else to backtest using the previously typed info: ")
            i = getPosIntInput(0, numDates - 1)
            match(i):
                case -2:
                    invalid = True
                case -1:
                    backtest()
                case _:
                    getAllScores(i)
                    sortScores()

                    invalid = True
                    while(invalid):
                        invalid = False
                        print("Type a number from 0 to " + str(numScores) + " to print the top N scores: ")
                        i = getPosIntInput(0, numScores)
                        match(i):
                            case -2 | -1:
                                invalid = True
                            case _:
                                printNScores(i)


def main():
    getDatesList()
    #readDates()
    
    getInput()
    

main()
