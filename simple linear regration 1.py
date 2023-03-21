# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 19:27:19 2023

@author: kailas
"""

1] Problem

BUSINESS OBJECTIVE:
    
1) Delivery_time -> Predict delivery time using sorting time..(It is Target Variables)
    
    

#Import Liabrary
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pylab as plt
import scipy
from scipy import stats
import pylab
import statsmodels.formula.api as smf

#dataset

df=pd.read_csv("D:\data science assignment\Assignments\simple linear regression\delivery_time.csv")
df.info()
df.describe()

#Rename the columns
df=df.rename(columns={'Delivery Time':'dt','Sorting Time':'st'})

# graphical visulisation (check Normality)
stats.probplot(df.dt,plot=pylab)
stats.probplot(df.st,plot=pylab)

#Scatter plot

plt.scatter(df.st,df.dt,color='blue')


#To quantify above,we use Co-relation of Coefficent(r)
np.corrcoef(df.st,df.dt)


#Calculate CO-Varience
np.cov(df.st,df.dt)



#Simple Linear Regression
model=smf.ols('dt ~ st',data=df).fit()
model.summary()

#Predictions for best fit line
pred=model.predict(pd.DataFrame(df['st']))


#Finding RMSE
#ACTUAL VALUE - PREDICTED VALUE
error=df.dt - pred 
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse

#Regression Line
plt.scatter(df.st,df.dt)
plt.plot(df.st,pred,'blue')
plt.legend(['predicted line','observed data'])
plt.show()

#we do transformations because we got,
Rsquare(Coefficent of determination=0.68)
so we need to capture more varience...
other factors including (r) and (RMSE) values are fine


#Log Transformation

model1=smf.ols('dt ~ np.log(st)',data=df).fit()
model1.summary()

#Predictions for best fit line

pred1=model1.predict(pd.DataFrame(df['st']))

#Finding RMSE
#ACTUAL VALUE - PREDICTED VALUE
error=df.dt - pred1 
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse


#Regression Line
plt.scatter(df.st,df.dt)
plt.plot(df.st,pred1,'blue')
plt.legend(['predicted line','observed data'])
plt.show()

#Exponantial Transformation
model2=smf.ols('np.log(dt) ~ st',data=df).fit()
model2.summary()


#Predictions for this best fit line
pred2=model2.predict(pd.DataFrame(df['st']))
pred2_exp=np.exp(pred2)
pred2_exp


#Finding RMSE
#ACTUAL VALUE - PREDICTED VALUE
error=df.dt - pred2_exp 
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse


#Regression Line
plt.scatter(df.st,df.dt)
plt.plot(df.st,pred2_exp,'blue')
plt.legend(['predicted line','observed data'])
plt.show()



#Square Root Transforamtions

model3=smf.ols('dt ~ np.sqrt(st)',data=df).fit()
model3.summary()

#Predictions for best fit line
pred3=model3.predict(pd.DataFrame(df['st']))


#Finding RMSE(Error Calculation)
#ACTUAL VALUE - PREDICTED VALUE
error=df.dt - pred3
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse


#Regression Line
plt.scatter(df.st,df.dt)
plt.plot(df.st,pred3,'blue')
plt.legend(['predicted line','observed data'])
plt.show


#Polynominal Transformation
model4=smf.ols('np.log(dt) ~ st + I(st*st)',data=df).fit()
model4.summary()

R-squared:0.765
Hence we use this model...
but also check all other factors is it fine or not?

pred4=model4.predict(pd.DataFrame(df['st']))

#Finding RMSE
#ACTUAL VALUE - PREDICTED VALUE
error=df.dt - pred4 
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse




#The Best Model

from sklearn.model_selection import train_test_split
train,test=train_test_split(df,test_size=0.2)

finalmodel=smf.ols('np.log(dt) ~ st + I(st*st)',data=train).fit()
finalmodel.summary()

#Predictions for test data
testpred=finalmodel.predict(pd.DataFrame(test))
testpred_exp=np.exp(testpred)

#Finding RMSE 
error=df.dt - testpred_exp
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse


#Predictions for Train data
trainpred=finalmodel.predict(pd.DataFrame(train))
trainpred_exp=np.exp(trainpred)


#RMSE for Train data
error=df.dt - trainpred_exp
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse




2]Problem
    

 Salary_hike -> Build a prediction model for Salary_hike
::input variable is yearsexprience and output Variable is Salary   

#DATASET
df=pd.read_csv("D:\data science assignment\Assignments\simple linear regression\Salary_Data.csv")
df.info()
df.describe()

#Rename the Columns
df=df.rename(columns={'YearsExperience':'yexp','Salary':'sal'})



#Graphical Representation (check noramlity) 
stats.probplot(df.yexp,plot=pylab)
stats.probplot(df.sal,plot=pylab)

#Scatter Plot
plt.scatter(df.yexp,df.sal)

# Co-relation Coefficent
np.corrcoef(df.yexp,df.sal)


#Import Liabrary
import statsmodels.formula.api as smf

#Simple Linear Regression
model=smf.ols('sal ~ yexp',data=df).fit()
model.summary()

#Predictions
pred=model.predict(pd.DataFrame(df['yexp']))

#Finding RMSE
error=df.sal - pred
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse


#Regression Line
plt.scatter(df.yexp,df.sal)
plt.plot(df.yexp,pred,'r')
plt.legend(['predicted line','observed data'])
plt.show()


#Firstly we do log Transformations


model1=smf.ols('sal ~ np.log(yexp)',data=df).fit()
model1.summary()

#Predictions
pred1=model1.predict(pd.DataFrame(df['yexp']))

#Finding RMSE
error=df.sal - pred1
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse

#Regression Line
plt.scatter(df.yexp,df.sal)
plt.plot(df.yexp,pred1,'r')
plt.legend(['predicted line','observed data'])
plt.show()

#Secondly we do Exponential Transformations


model2=smf.ols('np.log(sal) ~ yexp',data=df).fit()
model2.summary()

#Predictions
pred2=model2.predict(pd.DataFrame(df['yexp']))
pred2_exp=np.exp(pred2)
pred2_exp

#Finding RMSE
error=df.sal - pred2_exp
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse

#Regression Line
plt.scatter(df.yexp,df.sal)
plt.plot(df.yexp,pred2,'r')
plt.legend(['predicted line','observed data'])
plt.show()




#then do SQRT transformations


model3=smf.ols('sal ~ np.sqrt(yexp)',data=df).fit()
model3.summary()

#Predictions
pred3=model3.predict(pd.DataFrame(df['yexp']))

#Finding RMSE
error=df.sal - pred3
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse

#Regression Line
plt.scatter(df.yexp,df.sal)
plt.plot(df.yexp,pred3,'r')
plt.legend(['predicted line','observed data'])
plt.show()


#Lastly Polynominal Transformations
model4=smf.ols('np.log(sal) ~ yexp + I(yexp*yexp)',data=df).fit()
model4.summary()

#Predictions
pred4=model4.predict(pd.DataFrame(df['yexp']))
pred4_exp=np.exp(pred4)
pred4_exp

#Finding RMSE
error=df.sal - pred4_exp
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse

#Regression Line
plt.scatter(df.yexp,df.sal)
plt.plot(df.yexp,pred4_exp,'r')
plt.legend(['predicted line','observed data'])
plt.show()




#The Best Model
from sklearn.model_selection import train_test_split
train,test=train_test_split(df,test_size=0.2)

finalmodel=smf.ols('sal ~ yexp',data=train).fit()
finalmodel.summary()

#Predictions On Test Data
testpred=finalmodel.predict(pd.DataFrame(test))

#RMSE on Test Data
error=test.sal - testpred
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse

#Predictions On Train Data
trainpred=finalmodel.predict(pd.DataFrame(train))

#RMSE on Test Data
error=train.sal - trainpred
sqr=error * error
mean=np.mean(sqr)
rmse=np.sqrt(mean)
rmse


