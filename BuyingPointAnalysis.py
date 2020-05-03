import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

stock="XOM"
start = datetime.datetime(2010,1,1)
end = datetime.datetime(2020,5,31)

df=web.DataReader(stock, 'yahoo', start, end).drop(columns=['Adj Close'])
rawdata=web.DataReader(stock, 'yahoo', start, end)
df_SP500 = web.DataReader("^GSPC", 'yahoo', start,end)

df['Vol/MA10'] = df['Volume']/df['Volume'].rolling(10).mean()
#Eliminate the influence of the market, 
df['SP500'] = df_SP500['Close']
df['SP500_ROC'] = 100*df['SP500'].diff(1)/df['SP500'].shift(1)
# df['High'] = df['High']/df['SP500']
# df['Low'] = df['Low']/df['SP500']
# df['Close'] = df['Close']/df['SP500']
df['Close_ROC'] = 100*df['Close'].diff(1)/df['Close'].shift(1)


def calculate_k(ticker_df,cycle, M1 ):
    Close = ticker_df['Close']
    highest_hi = ticker_df['High'].rolling(window = cycle).max()
    lowest_lo = ticker_df['Low'].rolling(window=10).min()
    ticker_df['rsv'] = (Close - lowest_lo)/(highest_hi - lowest_lo)*100
    ticker_df['K'] = ticker_df['rsv'].rolling(window=M1).mean()
    ticker_df['K']  =  ticker_df['K'] .fillna(50)
    ticker_df['K_diff'] = ticker_df['K'].diff()
    ticker_df['K_prev'] = ticker_df['K'] - ticker_df['K_diff']
    #ticker_df['K_ROC'] = ticker_df['K']/ticker_df['K_prev']
    return ticker_df

def calculate_dj(ticker_df, M2 ):
    ticker_df['D'] = ticker_df['K'].rolling(window = M2).mean()
    ticker_df['D'] = ticker_df['D'].fillna(50)
    ticker_df['D_diff'] = ticker_df['D'].diff()
    ticker_df['D_prev'] = ticker_df['D'] - ticker_df['D_diff']
    #ticker_df['D_ROC'] = ticker_df['D']/ticker_df['D_prev']
    ticker_df['J'] = M2*ticker_df['K']-(M2-1)*ticker_df['D']
    return ticker_df

def stochastic_oscillator(ticker_df,cycle=12, M1=4, M2= 3):
    ticker_df = calculate_k(ticker_df,cycle,M1)
    ticker_df = calculate_dj(ticker_df, M2)
    return ticker_df


def buyjudge(ticker_df,cycle=20,gain=0.04):
    ticker_df['Max'] = ticker_df['High'].rolling(window = cycle).max().shift(-cycle)
    ticker_df['Good?'] =0
    df.loc[df['Max']>(1+gain)*ticker_df['Close'],'Good?'] = 1
    return ticker_df

stochastic_oscillator(df)

#Get rid of rows with NA value
df.dropna(axis=0, how='any', inplace=True)

kddf=[]
kddf = df.loc[:,['D','K']] 
kddf['Close'] = rawdata['Close']
kddf.plot()
plt.figure()
kddf.plot()
plt.show()

df['Intersection'] = 0 #buy indicator
#df.loc[(df['K_diff']>0) & (df['D_diff']>0) & (df['K']>df['D']) & (df['K_prev']<df['D_prev']) & (df['D']<=25) ,'Intersection'] = 1 
df.loc[(df['K']>df['D']) & (df['K_prev']<df['D_prev']) & (df['D']<=30) ,'Intersection'] = 1 
df['# Inter 10-day'] = df['Intersection'].rolling(10).sum()

# df['MA50']= df['Close'].rolling(50)
df['Close/MA20']= df['Close']/df['Close'].rolling(20).mean()
df['Close/MA50']= df['Close']/df['Close'].rolling(50).mean()

buyjudge(df)
#Get rid of rows with NA value
df.dropna(axis=0, how='any', inplace=True)

X=df.loc[df['Intersection']==1]
X=X.loc[:,['# Inter 10-day','Vol/MA10','SP500_ROC','Close_ROC','rsv','K','D','J','K_diff','D_diff','Close/MA20','Close/MA50',]] 
Y=df.loc[df['Intersection']==1]
Y=Y.loc[:,'Good?'] 