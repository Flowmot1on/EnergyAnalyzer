# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 17:36:53 2020

@author: yunus
"""

# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 09:47:04 2020

@author: yunus
"""

import dbCon as db
import dfManuplate as dfm
import numpy as np
from tensorflow import keras
from keras.callbacks import EarlyStopping
import pandas as pd
import time
from sklearn import preprocessing


def calculator(loss,diff):

        
    if ((np.max(loss)-np.quantile(loss,0.9999)) >= diff):

        return np.quantile(loss,0.999) + 0.2
    else:

        
        return np.quantile(loss,0.99999)
    
    
if __name__ == '__main__':
    while (1):
        try:
            con = True
            while (con):
                try:
                    
                    df_clients = db.readClients()
                    con = False 
                except Exception as e:
                    print (e)
                    time.sleep(5)
                
        
            df_clients = df_clients.sort_values(by=['clientId'], ascending= True).reset_index(drop = True)
            
            global counter
            
            counter = True
            
            for clnt in df_clients.clientId.unique():
                
                #clnt = 424
                
                objects = df_clients[df_clients['clientId'] == clnt].reset_index(drop=True)
                
                
                for obj in np.unique(objects.objectId): 
                    
                    #obj = 21
                    
                    period = int(objects['period'][0])
                    
                    outliers_fraction = 0.003
                    
                    threshold = 1000
                    
                    #int(objects['threshold'][0])
                    
                    devices = objects[objects['objectId'] == obj].reset_index(drop=True)
               
                    for dvc in devices.deviceId:
                        
                        print (dvc)
                        
                        #dvc = 3255
                        
                        tic = time.process_time()  
                        
                        con = True
                        
                        while (con):
                            
                            try:
                                
                                df1 = db.readLog(dvc,28)
                                
                                #exec("{} = df.copy()".format(f'rg_{dvc}'))
                                
                                con = False
                                
                            except Exception as e:
                                
                                print (e)
                                
                                time.sleep(5)
                                
                        if (len(df1.index) < 100):
                            print("empty",dvc)
                            continue
                        
                           
                        df1 = df1.sort_values(by=['logTimestamp'], ascending=True).reset_index(drop = True)
                        
                        df_temp = dfm.manuplate(df1,1000,60)
        
                        df = df_temp.head(int(len(df_temp)*1)).reset_index(drop=True)
                        print(str(len(df1)) + " ---- " + str(len(df)))
        
                        df = df[['value','timestamp','corruption','workTime']]
                        
                        df = df.set_index('timestamp')
                        
                        min_max_scaler = preprocessing.StandardScaler()
                        np_scaled = min_max_scaler.fit_transform(df)
                        #df_temp = pd.DataFrame(np_scaled)
                        df['value'] = np_scaled
        
                        if dvc in {1,2,4,5,6}:
                            spt = 0.33333
                        else:
                            spt = 0.25
                        test_size = int(len(df_temp) * spt)
                        df = df[df['value'] != 0]
                        train_size = len(df) - test_size
                        train,test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
                        print(train.shape, test.shape)
                        
                        
                        def create_dataset(X, y, time_steps):
                            Xs, ys = [], []
                            for j in range(len(X) - time_steps):
                                v = X.iloc[j:(j + time_steps)].values
                                Xs.append(v)        
                                ys.append(y.iloc[j + time_steps])
                            return np.array(Xs), np.array(ys)
                        
                        TIME_STEPS = 30
        
                        # reshape to [samples, time_steps, n_features]
                        #test_score_df.to_excel('test11.xlsx')
                        #y_test.to_excel('test1.xlsx')
                        X_train, y_train = create_dataset(train[['value']], train.value, TIME_STEPS)
                        X_test, y_test = create_dataset(test[['value']], test.value, TIME_STEPS)
                        
                        model = keras.Sequential()
                        model.add(keras.layers.LSTM(
                            units=64, 
                            input_shape=(X_train.shape[1], X_train.shape[2])
                        ))
                        model.add(keras.layers.Dropout(rate=0.2))
                        model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
                        model.add(keras.layers.LSTM(units=64, return_sequences=True))
                        model.add(keras.layers.Dropout(rate=0.2))
                        model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=X_train.shape[2])))
                        model.compile(loss='mae', optimizer='adam')
                        es = EarlyStopping(monitor='val_loss', mode='min')
                        epoch = 4000
                        history = model.fit(
                                X_train, y_train,
                                epochs = epoch,
                                batch_size = 8,
                                validation_split=spt,
                                shuffle=False,
                                callbacks=[es]
                            )
                        
                        
                        X_train_pred = model.predict(X_train)
        
                        train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)    
                        
                        X_test_pred = model.predict(X_test)
        
                        test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
        
                        test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
                        test_score_df['loss'] = test_mae_loss
                        THRESHOLD = calculator(test_score_df.loss,1)
                        test_score_df['threshold'] = THRESHOLD                   
                        test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
                        test_score_df['value'] = test[TIME_STEPS:].value
                        
                        
                        anomalies = test_score_df[test_score_df.anomaly == True]
                        toc = time.process_time()  
              
                        scaler = preprocessing.StandardScaler()
                        scaler = scaler.fit(train[['value']])
                        
        
                        anomalies = anomalies.reset_index()
                        df_temp = df_temp.merge(anomalies, on='timestamp', how='left')
                        """
                        print('rnnAnomal  --- ' + str(toc - tic) )     
                        
                        exec("{} = df_temp.copy()".format(f'rg_{dvc}'))  
                        """
                        df_anomaly = df_temp[df_temp.anomaly == True].reset_index(drop=True)
                        
                        try:
                                
                            if(len(df_anomaly)):
                                for i in df_anomaly.index:
                                    
                                    db.createAnomalyReport( dvc, int(obj), 2 ,int(df_anomaly.firstLogID[i]), int(df_anomaly.lastLogID[i]))
                                print("Report Row Created")
                            else:
                                print("Anomaly does not exist")
                        except Exception as e:
                            print (e)
                            time.sleep(5)
                       
        except Exception as e:
                 print (e)
                 print ("Rebooting...")
                 time.sleep(20)