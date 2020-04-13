import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

stock="XOM"
start = datetime.datetime(2020,2,1)
end = datetime.datetime(2020,3,31)

df=web.DataReader(stock, 'yahoo', start, end).drop(columns=['Adj Close'])
#a=df.drop(columns=['Adj Close'])
df1=web.DataReader(stock, 'yahoo', start, end)
actions = web.DataReader(stock, 'yahoo-actions', start, end)
df['Diviend'] = actions.value


df_SP500 = web.DataReader("^GSPC", 'yahoo', start,end)

#df['MA']= df['Close'].rolling(250).mean()*0.5

#Eliminate the influence of the market
df['SP500Close'] = df_SP500['Close']
df['High'] = df['High']/df['SP500Close']
df['Low'] = df['Low']/df['SP500Close']
df['Close'] = df['Close']/df['SP500Close']
Comparison = (df['Open'].iloc[-1]/df['Open'].iloc[0]-1)*100

df.tail()
df['Close.Diff'] = df['Close'].diff()
#df['CloseRateDelta'] = df['Close.Diff']/df['Close']*100


def calculate_k(ticker_df,cycle, M1 ):
    Close = ticker_df['Close']
    highest_hi = ticker_df['High'].rolling(window = cycle).max()
    lowest_lo = ticker_df['Low'].rolling(window=10).min()
    ticker_df['rsv'] = (Close - lowest_lo)/(highest_hi - lowest_lo)*100
    ticker_df['K'] = ticker_df['rsv'].rolling(window=M1).mean()
    ticker_df['K']  =  ticker_df['K'] .fillna(50)
    ticker_df['k_diff'] = ticker_df['K'].diff()
    ticker_df['K_prev'] = ticker_df['K'] - ticker_df['k_diff']
    ticker_df['KChangingRatio'] = ticker_df['K']/ticker_df['K_prev']
    return ticker_df

def calculate_dj(ticker_df, M2 ):
    ticker_df['D'] = ticker_df['K'].rolling(window = M2).mean()
    ticker_df['D'] = ticker_df['D'].fillna(50)
    ticker_df['D_diff'] = ticker_df['D'].diff()
    ticker_df['D_prev'] = ticker_df['D'] - ticker_df['D_diff']
    ticker_df['DChangingRatio'] = ticker_df['D']/ticker_df['D_prev']
    ticker_df['J'] = M2*ticker_df['K']-(M2-1)*ticker_df['D']
    return ticker_df
    
def stochastic_oscillator(ticker_df,cycle=12, M1=4, M2= 3):
    ticker_df = calculate_k(ticker_df,cycle,M1)
    ticker_df = calculate_dj(ticker_df, M2)
    return ticker_df

stochastic_oscillator(df)   
kddf = df[['D','K']]
kddf['Close'] = df1['Close']
kddf.plot()

plt.figure()
kddf.plt.plot().show()

dist=0
df['flag_buy'] = 0 #buy indicator
df.loc[(df['k_diff']>0) & (df['D_diff']>0) & (df['K']>df['D']+dist) & (df['K_prev']<df['D_prev']) & (df['D']<=20) ,'flag_buy'] = 1# buy indicator & (df['J']<0)

df['flag_sell'] = 0 #sell indicator

df.loc[(df['k_diff']<0) & (df['D_diff']<0) & (df['K']<df['D']-dist) & (df['K_prev']>df['D_prev']) & (df['D']>=80),'flag_sell'] = 1 # sell indicator & (df['J']>100)

df['flag_buy_diff'] = df['flag_buy'].diff()
df['flag_buy_diff'] = df['flag_buy_diff'].apply(lambda x: 1 if x ==1 else 0)
df['flag_sell_diff'] = df['flag_sell'].diff()
df['flag_sell_diff'] = df['flag_sell_diff'].apply(lambda x: 1 if x ==1 else 0)
#df['flag_sell_diff'][df['flag_buy_diff']==-1] = 1

count = 0 

df['buy_rolling']= df['flag_buy_diff'].rolling(35).sum()
df['sell_rolling']= df['flag_sell_diff'].rolling(35).sum()
df['sell_MA'] = 0
df.loc[(df['Adj Close']<df['MA'])  ,'sell_MA'] = 1

df['sell_MA_diff'] = df['sell_MA'].diff()
#df['sell_MA_diff'] = df['sell_MA_diff'].apply(lambda x: 1 if x ==1 else 0)
df['flag_trade'] = 0
df.loc[(df['buy_rolling']==1) & (df['flag_buy_diff']==1)&(df['sell_MA']!=1),'flag_trade'] = 1# &(df['sell_MA']!=1)
df.loc[(df['sell_rolling']==3) & (df['flag_sell_diff']==1) | (df['sell_MA_diff']==1),'flag_trade'] = -1# | (df['sell_MA_diff']==1)

df['flag_trade_diff'] = 0


for a in df['flag_trade'].reset_index().index:
    count+= df['flag_trade'].iloc[a]
    if count<0:
        count = 0
        print(0)
    elif count > 1:
        count = 1
        print(1)
    else:
        df.loc[a ,'flag_trade_diff'] = df['flag_trade'].iloc[a]

if len(df['Open'][df['flag_trade_diff']==1]) > len(df['Open'][df['flag_trade_diff']==-1]):
    Yield = ((df['Open'][df['flag_trade_diff']==-1].product())/(df['Open'][df['flag_trade_diff']==1].product())*(df['Open'].iloc[-1])-1)*100
    print(1)
else:
    Yield = ((df['Open'][df['flag_trade_diff']==-1].product())/(df['Open'][df['flag_trade_diff']==1].product())-1)*100


Record = df[['Date', 'High', 'Low', 'Open', 'Close', 'Volume','flag_trade','flag_trade_diff','J']][df['flag_trade_diff']!=0]#df.to_csv('data.csv')    
