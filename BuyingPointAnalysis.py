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
from xgboost import XGBClassifier as Gxb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
matplotlib.style.use('ggplot')

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="lower left")
    plt.xlim([0.1, 1])
    plt.ylim([0.1, 1])
    plt.title([stock,method_list.columns[index]])

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
    #Judgem whether an intersection is a good buying point, if in the following 14 days, the price goes up by at leasr 3%, it is good
def buyjudge(ticker_df,gain=0.024,cycle=10):
    ticker_df['Max'] = ticker_df['Close'].rolling(window = cycle).max().shift(-cycle)
    ticker_df['Good?'] =0
    df.loc[(df['Max']>(1+gain)*ticker_df['Close']),'Good?'] = 1#& (ticker_df['Min']<*ticker_df['Close'])
    return ticker_df

method_list = [{#'DecisionTree':tree.DecisionTreeClassifier(),
                'Bayes':GaussianNB(),
                #'LogisticRegression':LogisticRegression(),
                'SVC(C=1)':svm.SVC(probability=True),
                'SVC(C=1.5)':svm.SVC(C=1.5,probability=True),
                'SVC(C=2)':svm.SVC(C=2,probability=True),
                'SVC(linear, C=2)':svm.SVC(kernel='linear',probability=True),
                'SVC(poly, C=2)':svm.SVC(kernel='poly',probability=True),
                'RandomForest(random_state=10)':RandomForestClassifier(random_state=10), 
                'XGBT()':Gxb(),
                'XGBT(λ=0.8)':Gxb(reg_lambda=0.8),
                'XGBT(λ=1.2)':Gxb(reg_lambda=1.2),
                'XGBT(λ=1.4)':Gxb(reg_lambda=1.4),
                'XGBT(γ=0.2)': Gxb(gamma=0.2),
                'XGBT(γ=0.3)': Gxb(gamma=0.3),
                'XGBT(γ=0.4)': Gxb(gamma=0.4),
                'XGBT(γ=0.5)': Gxb(gamma=0.5),
                'XGBT(γ=0.55)': Gxb(gamma=0.55),
                }]#
method_list=pd.DataFrame(method_list)
ResultTable=DataFrame(columns=['Stock','Method','AvgScores','StdScores'])

start = datetime.datetime(2005,1,1)
end = datetime.datetime(2020,5,31)
df_SP500 = web.DataReader("^GSPC", 'yahoo', start,end)

#Load ticker data
#stocklist=['MSFT','AAPL','AMZN','GOOG','FB','JNJ','V']
           #,'PG','JPM','UNH','MA','INTC','VZ','HD','T','PFE','MRK','PEP']
stocklist=['MSFT']
for stock in stocklist:
    df=web.DataReader(stock, 'yahoo', start, end).drop(columns=['Adj Close'])
    rawdata=df
    #Some indicators
    df['MAVOL10'] = df['Volume']/df['Volume'].rolling(10).mean()
    df['MAVOL5'] = df['Volume']/df['Volume'].rolling(5).mean()
    #Eliminate the influence of the market, 
    df['SP500'] = df_SP500['Close']
    df['SP500_ROC'] = 100*df['SP500'].diff(1)/df['SP500'].shift(1)
    df['Close_ROC'] = 100*df['Close'].diff(1)/df['Close'].shift(1)   
    stochastic_oscillator(df)
    
    df['Intersection'] = 0 #buy indicator
    df.loc[(df['K']>df['D']) & (df['K_prev']<df['D_prev']) & (df['D']<=50)  & (df['D_diff']>0),'Intersection'] = 1# Intersections: K go exceeds D   
    df['# Inter 10-day'] = df['Intersection'].rolling(14).sum()# number of intersections during past 10 days    
    df['Close/MA10']= df['Close']/df['Close'].rolling(10).mean()# df['MA10']= df['Close'].rolling(10) Moving Average of the past 10 days
    df['Close/MA20']= df['Close']/df['Close'].rolling(20).mean()
    df['Close/MA50']= df['Close']/df['Close'].rolling(50).mean()
    df['Close/MA100']= df['Close']/df['Close'].rolling(100).mean()
    df['Close/MA200']= df['Close']/df['Close'].rolling(200).mean()
    buyjudge(df)
    df.dropna(axis=0, how='any', inplace=True)#Get rid of rows with NA value
    
    # X=df.loc[df['Intersection']==1]
    # X=X.loc[:,['# Inter 10-day','Intersection','MAVOL10','MAVOL5','SP500_ROC','Close_ROC','rsv','K','D','J',
    #             'K_ROC','D_ROC','K_diff','D_diff','J_ROC','J_diff','Close/MA10','Close/MA20','Close/MA50',
    #             'Close/MA100','Close/MA200',]] 
    # y=df.loc[df['Intersection']==1]
    # y=y.loc[:,'Good?']
    
    
    X=df.loc[:,['# Inter 10-day','Intersection','MAVOL10','MAVOL5','SP500_ROC','Close_ROC','rsv','K','D','J',
                'K_ROC','D_ROC','K_diff','D_diff','J_ROC','J_diff','Close/MA10','Close/MA20','Close/MA50',
                'Close/MA100','Close/MA200',]] 
    y=df.loc[:,'Good?']
   
    
    Market_GoodRatio=sum(df['Good?']==1)/len(df['Good?'])
    Intersection_GoodRatio=sum(df.loc[df['Intersection']==1,'Good?']==1)/len(df.loc[df['Intersection']==1,'Good?'])
    ResultTable=ResultTable.append({'Stock':stock,'Method':'Market Good Buying Ratio','AvgScores':Market_GoodRatio,'StdScores':0},ignore_index=True)
    ResultTable=ResultTable.append({'Stock':stock,'Method':'Intersection Good Buying Ratio','AvgScores':Intersection_GoodRatio,'StdScores':0},ignore_index=True)
        
    #try different classification methods and compare the accuracy
    index=0
    for method in method_list.loc[0,:]:
        clf = method
        cv=TimeSeriesSplit(n_splits=3)
        scores = cross_val_score(clf,X, y, cv=cv,scoring='precision')
        print(scores[scores>0])
        series={'Stock':stock,'Method':method_list.columns[index],'AvgScores':scores[scores>0].mean(),'StdScores':scores[scores>0].std()}
        index=index+1
        ResultTable=ResultTable.append(series,ignore_index=True)

name_list= ['Market Good Buying Ratio','Intersection Good Buying Ratio']
name_list=np.append(name_list,method_list.columns)
for stock in stocklist:
    num_list= ResultTable.loc[ResultTable['Stock']==stock]['AvgScores']
    plt.barh(range(len(num_list)), num_list,tick_label = name_list)
    plt.title(stock)
    plt.show()

# # 返回预测标签
# print(clf.predict(X[-1:]))
 
# # 返回预测属于某标签的概率
# print(clf.predict_proba(X[-1:]))
xtrain=X[:3000]
ytrain=y[:3000]
xtest=X[3000:]
ytest=y[3000:]
index=0
for method in method_list.loc[0,:]:
    nb = method
    nb.fit(xtrain, ytrain)
    predicted = nb.predict_proba(xtest)
    precision, recall, threshold = precision_recall_curve(ytest, predicted[:,1])
    plot_precision_recall_vs_threshold(precision, recall, threshold)
    index=index+1
    plt.show()
   
