# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:44:48 2020

@author: yunusemreemik
"""

import pandas as pd
import numpy as np

def manuplate(df,threshold,sec):
            
        df = df.rename(columns={"logValue" : "value","logTimestamp" : "timestamp"}).drop_duplicates(subset=['timestamp'], keep=False)
        

        df['workTime'] = np.where(pd.notnull(df['ruleId']), 1 , 0)
        
        df1 = pd.Series(df.value.tolist(), index=df.timestamp)
        
        df2 = pd.Series(df.workTime.tolist(), index=df.timestamp)
        
        df3 = pd.Series(df.logID.tolist(), index=df.timestamp)
        
        
        resec = str(sec) + 's'
        
        df1 = df1.resample(str(resec)).mean()
        
        df2 = df2.resample(str(resec)).min()
        
        df33 = df3.resample(str(resec)).min()

        df4 = df3.resample(str(resec)).max()
        
        df = df1.reset_index()
         
        df["workTime"] = df2.values
        
        df['firstLogID'] = df33.values
        
        df['lastLogID'] = df4.values
        
        
        df = df.rename(columns={0 : "value"})
        
        
        
        df = df[pd.notnull(df['value'])]

        df = df[df['value'] != 0].reset_index(drop = True)
        
        
        # change the type of timestamp column for plotting
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # the hours and if it's night or day (7:00-22:00)
        df['hours'] = df['timestamp'].dt.hour
        
        df['daylight'] = ((df['hours'] >= 8) & (df['hours'] <= 19)).astype(int)
        # the day of the week (Monday=0, Sunday=6) and if it's a week end day or week day.
        
        df['DayOfTheWeek'] = df['timestamp'].dt.dayofweek
        
        df['WeekDay'] = (df['DayOfTheWeek'] < 5).astype(int)
        
        # time with int to plot easily
        df['time_epoch'] = (df['timestamp'].astype(np.int64)/100000000000).astype(np.int64)
        # creation of 4 distinct categories that seem useful (week end/day week & night/day)
        
        df['categories'] = df['WeekDay']*2 + df['daylight']
        
        df['tempTime'] = df['timestamp'].shift(-1)
        
        df['timeDifference'] = df['tempTime'] - df['timestamp']
        
        df = df[pd.notnull(df['timeDifference'])]
        
        df['timeDifference'] = df['timeDifference'].apply(lambda x: '{loc}'.format(loc=int(x.total_seconds())))
        
        df['corruption'] = np.where(df['timeDifference'].astype('int') > threshold , 1 , 0)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df['firstLogID'] = df['firstLogID'].astype('int')
        
        df['lastLogID'] = df['lastLogID'].astype('int')

        return df