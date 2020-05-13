import pandas as pd
import datetime
import numpy as np
from array import *
import pandas_datareader.data as web
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model, datasets
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier as Xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
matplotlib.style.use('ggplot')

# Define functions
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="lower left")
    plt.xlim([0.5, 0.95])
    plt.ylim([0, 1])
    plt.title(stock+'\n'+method_list.columns[index])

#Define Stochastic Osciliator
def calculate_k(ticker_df,cycle, M1 ):
    Close = ticker_df['Close']
    highest_hi = ticker_df['High'].rolling(window = cycle).max()
    lowest_lo = ticker_df['Low'].rolling(window=10).min()
    ticker_df['rsv'] = (Close - lowest_lo)/(highest_hi - lowest_lo)*100
    ticker_df['K'] = ticker_df['rsv'].rolling(window=M1).mean()
    ticker_df['K']  =  ticker_df['K'] .fillna(50)
    ticker_df['K_diff'] = ticker_df['K'].diff()
    ticker_df['K_prev'] = ticker_df['K'] - ticker_df['K_diff']
    ticker_df['K_ROC'] = ticker_df['K']/ticker_df['K_prev']
    return ticker_df
def calculate_dj(ticker_df, M2 ):
    ticker_df['D'] = ticker_df['K'].rolling(window = M2).mean()
    ticker_df['D'] = ticker_df['D'].fillna(50)
    ticker_df['D_diff'] = ticker_df['D'].diff()
    ticker_df['D_prev'] = ticker_df['D'] - ticker_df['D_diff']
    ticker_df['D_ROC'] = ticker_df['D']/ticker_df['D_prev']
    ticker_df['J'] = M2*ticker_df['K']-(M2-1)*ticker_df['D']
    ticker_df['J_diff'] = ticker_df['J'].diff()
    ticker_df['J_prev'] = ticker_df['J'] - ticker_df['J_diff']
    ticker_df['J_ROC'] = ticker_df['J']/ticker_df['J_prev']
    return ticker_df
def stochastic_oscillator(ticker_df,cycle=12, M1=4, M2= 3):
    ticker_df = calculate_k(ticker_df,cycle,M1)
    ticker_df = calculate_dj(ticker_df, M2)
    return ticker_df
#Evenly separeate all days into good buying points and poor buying points
def buyjudge(ticker_df,gain=0.024,cycle=10):
    ticker_df['Max'] = ticker_df['Close'].rolling(window = cycle).max().shift(-cycle)
    ticker_df['Good Buy Point?'] =0
    df.loc[(df['Max']>(1+gain)*ticker_df['Close']),'Good Buy Point?'] = 1#& (ticker_df['Min']<*ticker_df['Close'])
    return ticker_df

method_name = [{
                'Bayes':GaussianNB(var_smoothing=1e-09),
                'Bayese-2':GaussianNB(var_smoothing=1e-02),
                'Bayese-1':GaussianNB(var_smoothing=1e-01),
                'Bayese0.5':GaussianNB(var_smoothing=0.5),
                'Bayese1':GaussianNB(var_smoothing=1),
                'Bayese3':GaussianNB(var_smoothing=3),
                # 'SVC(C=1)':svm.SVC(probability=True),
                # 'SVC(C=1.5)':svm.SVC(C=1.5,probability=True),
                #'SVC(linear, C=1)':svm.SVC(kernel='linear', C=1,probability=True),
                # 'SVC(linear, C=2)':svm.SVC(kernel='linear',probability=True),
                # 'SVC(poly, C=2)':svm.SVC(kernel='poly',probability=True),
                #'XGBT(λ=1.2)':Xgb(reg_lambda=1.2),#Result of parameter tunning in XGBPara.py
                #'XGBT(λ=1.4)':Xgb(reg_lambda=1.4,max_depth=150),
                }]
method_list=pd.DataFrame(method_name)
ResultTable=DataFrame(columns=['Stock','Method','AvgScores','StdScores'])
start = datetime.datetime(2005,1,1)
end = datetime.datetime(2020,5,31)
df_SP500 = web.DataReader("^GSPC", 'yahoo', start,end)
stocklist=['MSFT'] #Load ticker data'MSFT','AAPL','AMZN','GOOG','FB','JNJ','V','PG','JPM','UNH','MA','INTC','VZ','HD','T','PFE','MRK','PEP']
for stock in stocklist:
    df=web.DataReader(stock, 'yahoo', start, end).drop(columns=['Adj Close'])
    rawdata=df
    #Selected indicators
    df['MAVOL200'] = df['Volume']/df['Volume'].rolling(200).mean()
    df['MAVOL20'] = df['Volume']/df['Volume'].rolling(20).mean()
    df['MAVOL10'] = df['Volume']/df['Volume'].rolling(10).mean()
    df['MAVOL5'] = df['Volume']/df['Volume'].rolling(5).mean() 
    df['SP500'] = df_SP500['Close']
    df['SP500_ROC'] = 100*df['SP500'].diff(1)/df['SP500'].shift(1)
    df['Close_ROC'] = 100*df['Close'].diff(1)/df['Close'].shift(1)   
    stochastic_oscillator(df)
    df['Intersection'] = 0
    df.loc[(df['K']>df['D']) & (df['K_prev']<df['D_prev']) & (df['D']<=80)  & (df['D_diff']>0),'Intersection'] = 1# Intersections: K go exceeds D   
    df['# Inter 10-day'] = df['Intersection'].rolling(14).sum()# number of intersections during past 10 days    
    df['Close/MA10']= df['Close']/df['Close'].rolling(10).mean()# df['MA10']= df['Close'].rolling(10) Moving Average of the past 10 days
    df['Close/MA20']= df['Close']/df['Close'].rolling(20).mean()
    df['Close/MA50']= df['Close']/df['Close'].rolling(50).mean()
    df['Close/MA100']= df['Close']/df['Close'].rolling(100).mean()
    df['Close/MA200']= df['Close']/df['Close'].rolling(200).mean()
    buyjudge(df)
    df.dropna(axis=0, how='any', inplace=True)#Get rid of rows with NA value
    #Retrive X and y 
    X=df.loc[:,['# Inter 10-day','Intersection','MAVOL200','MAVOL20','MAVOL10','MAVOL5','SP500_ROC','Close_ROC','rsv','K','D','J',
                'K_ROC','D_ROC','K_diff','D_diff','J_ROC','J_diff','Close/MA10','Close/MA20','Close/MA50',
                'Close/MA100','Close/MA200',]] 
    y=df.loc[:,'Good Buy Point?']
    # split train and test data
    xtrain=X[:3000]
    ytrain=y[:3000]
    xtest=X[3000:]
    ytest=y[3000:]
    Market_GoodRatio=sum(df['Good Buy Point?']==1)/len(df['Good Buy Point?'])#Good Buying Point Ratio in market is manully set to nearly 0.5 
    ResultTable=ResultTable.append({'Stock':stock,'Method':'Market Good Buying Ratio','AvgScores':Market_GoodRatio,'StdScores':0},ignore_index=True)
    #Compare and Plot the precision rate of each algorithm        
    index=0
    for method in method_list.loc[0,:]:
        clf = method
        #cv=TimeSeriesSplit(n_splits=3)
        scores = cross_val_score(clf,xtrain, ytrain, cv=4,scoring='precision')
        print(scores[scores>0])
        series={'Stock':stock,'Method':method_list.columns[index],'AvgScores':scores[scores>0].mean(),'StdScores':scores[scores>0].std()}
        index=index+1
        ResultTable=ResultTable.append(series,ignore_index=True)

name_list= ['Market Good Buying Ratio']
name_list=np.append(name_list,method_list.columns)
for stock in stocklist:
    num_list= ResultTable.loc[ResultTable['Stock']==stock]['AvgScores']
    plt.barh(range(len(num_list)), num_list,tick_label = name_list)
    plt.title(stock+'\nPrecision Rate')
    plt.show()
#Plot precission rate 
index=0
for method in method_list.loc[0,:]:
     clf = method
     clf.fit(xtrain, ytrain)
     predicted = clf.predict_proba(xtest)
     precision, recall, threshold = precision_recall_curve(ytest, predicted[:,1])
     plot_precision_recall_vs_threshold(precision, recall, threshold)
     plt.show()
     index=index+1
#%%  Visualize the points       
#select best method        
#clf = svm.SVC(C=2,probability=True)
clfbuy =GaussianNB(var_smoothing=2) 
#clfbuy =Xgb(reg_lambda=1.2,max_depth=100)
clfbuy.fit(xtrain, ytrain)
predicted = clfbuy.predict_proba(xtest)    

dfplot=pd.DataFrame()
dfplot.loc[:,'Close']=df[3000:]['Close']
dfplot.loc[:,'GoodProb']=predicted[:,1]
for threshold in np.arange(0.58,0.6,0.01):
    dfplot['Good']=0
    dfplot.loc[(dfplot['GoodProb']>threshold),'Good'] = dfplot['Close']
    x=dfplot.index
    y1=dfplot['Close']
    y2=dfplot['Good']
    plt.plot(x, y1,'c',label='Price')
    plt.plot(x, y2, 'o', ms=4.5, label='Buying Point')
    plt.ylim([10, 200])
    plt.title(stock+'\nThreshold='+str(round(threshold,3)))
    plt.legend(loc='upper left')
    plt.show()
#dfplot.to_csv('plot.csv')
