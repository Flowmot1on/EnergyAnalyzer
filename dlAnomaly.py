# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:43:33 2020

@author: yunusemreemik
"""
# specific libraries for RNN

from sklearn import preprocessing

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from matplotlib import pyplot as plt


import pandas as pd
import numpy as np
import time
import tensorflow as tf



def rnnAnomaly(df,outliersFraction,dvc_stat,device):
    
    #select and standardize data
    data_n = df[['value','workTime','corruption']]  
    min_max_scaler = preprocessing.StandardScaler()
    np_scaled = min_max_scaler.fit_transform(data_n)
    data_n = pd.DataFrame(np_scaled)
    
    # important parameters and train/test size
    prediction_time = 1 
    testdatasize = int(len(df.index)*(0.25))
    #testdatasize = 2000
    unroll_length = 50
    testdatacut = testdatasize + unroll_length  + 1 
    
    #train data
    x_train = data_n[0:-prediction_time-testdatacut].as_matrix()
    y_train = data_n[prediction_time:-testdatacut  ][0].as_matrix()
    
    # test data
    x_test = data_n[0-testdatacut:-prediction_time].as_matrix()
    y_test = data_n[prediction_time-testdatacut:  ][0].as_matrix()
    #unroll: create sequence of 50 previous data points for each data points
    def unroll(data,sequence_length=24):
        result = []
        for index in range(len(data) - sequence_length):
            result.append(data[index: index + sequence_length])
        return np.asarray(result)
    
    # adapt the datasets for the sequence data shape
    x_train = unroll(x_train,unroll_length)
    x_test  = unroll(x_test,unroll_length)
    y_train = y_train[-x_train.shape[0]:]
    y_test  = y_test[-x_test.shape[0]:]
    
    
    if (device not in dvc_stat.keys()):
        model = Sequential()
        
        model.add(LSTM(
            input_dim=x_train.shape[-1],
            output_dim=50,
            return_sequences=True))
        model.add(Dropout(0.2))
        
        model.add(LSTM(
            100,
            return_sequences=False))
        model.add(Dropout(0.2))
        
        model.add(Dense(
            units=1))
        model.add(Activation('linear'))
        
        start = time.time()
        model.compile(loss='mse', optimizer='rmsprop')
        print('compilation time : {}'.format(time.time() - start))
        
        # Train the model
        #nb_epoch = 350
        model.fit(
            x_train,
            y_train,
            batch_size=3028,
            nb_epoch=10,
            validation_split=0.1)
        
        # create the list of difference between prediction and test data
        loaded_model = model
        diff=[]
        ratio=[]
        p = loaded_model.predict(x_test)

        score = model.evaluate(x_train,y_train, verbose=0)
        #print("SCR   :  --   %s: %.2f%%" % (model.metrics_names[1], score[1]*100))
        
        fpath = 'train_models/model_{}.h5'.format(device)
        config = model.save(fpath)

        #dvc_stat.add(device, True)
        
    #elif dvc_stat[device] == True:
    else:
         
        fpath = 'train_models/model_{}.h5'.format(device)
        
        new_model = tf.keras.models.load_model(fpath)
        # Check that the state is preserved
        new_model.compile(loss='mse', optimizer='rmsprop')
        score = new_model.evaluate(x_train,y_train, verbose=0)
        new_loaded_model = new_model
        diff=[]
        ratio=[]
        p = new_loaded_model.predict(x_test)
        print (score)
        #np.testing.assert_allclose(p, new_predictions, rtol=1e-6, atol=1e-6)
        
    
    print("model predicted")         


    for u in range(len(y_test)):
        pr = p[u][0]
        ratio.append((y_test[u]/pr)-1)
        diff.append(abs(y_test[u]- pr))
    # plot the prediction and the reality (for the test data)

    # select the most distant prediction/reality data points as anomalies
    diff = pd.Series(diff)
    number_of_outliers = int(outliersFraction*len(diff))
    threshold = diff.nlargest(number_of_outliers).min()
    # data with anomaly label (test data part)
    test = (diff >= threshold).astype(int)
    # the training data part where we didn't predict anything (overfitting possible): no anomaly
    complement = pd.Series(0, index=np.arange(len(data_n)-testdatasize))
    # # add the data to the main
    df['anomaly27'] = complement.append(test, ignore_index='True')
    
    
    #-------------------
    
    print(  df['anomaly27'].value_counts())
    
    # visualisation of anomaly throughout time (viz 1)
    fig, ax = plt.subplots()
    
    df = df.reset_index()
    # visualisation of anomaly throughout time (viz 1)
    fig, ax = plt.subplots()
    a = df.loc[df['anomaly27'] == 1, ['Index', 'value']] #anomaly  
    ax.plot(df['Index'], df['value'], color='blue')
    ax.scatter(a['Index'],a['value'], color='red')
    ax.set_title('KMeans Outliers')
    plt.show()
    return df,dvc_stat

def getCounter(counter):
    if counter:
        counter = False
        return counter 