# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 13:46:35 2020

@author: yunusemreemik
"""

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np



def getDistanceByPoint(data, model):
    distance = pd.Series()
    for i in range(0,len(data)):
        Xa = np.array(data.loc[i])
        Xb = model.cluster_centers_[model.labels_[i]-1]
        distance.at[i] = np.linalg.norm(Xa-Xb)
        #distance.set_value(i, np.linalg.norm(Xa-Xb))
    return distance

def kmeansAnomaly(df,outliersFraction):
    
    data = df[['value','workTime','corruption']]  
    #data = df[['value','workTime', 'hours', 'daylight', 'DayOfTheWeek', 'WeekDay','corruption']]
    

    min_max_scaler = preprocessing.StandardScaler()
    
    np_scaled = min_max_scaler.fit_transform(data)
    
    data = pd.DataFrame(np_scaled)
    # reduce to 2 importants features
    pca = PCA(n_components=2)
    data = pca.fit_transform(data)
    
    # standardize these 2 new features
    min_max_scaler = preprocessing.StandardScaler()
    np_scaled = min_max_scaler.fit_transform(data)
    data = pd.DataFrame(np_scaled)
    
    
    # calculate with different number of centroids to see the loss plot (elbow method)
    n_cluster = range(1, 20)
    kmeans = [KMeans(n_clusters=i).fit(data) for i in n_cluster]
    scores = [kmeans[i].score(data) for i in range(len(kmeans))]
    
    # Not clear for me, I choose 15 centroids arbitrarily and add these data to the central dataframe
    
    df['cluster'] = kmeans[14].predict(data)
    df['principal_feature1'] = data[0].tolist()
    df['principal_feature2'] = data[1].tolist()
    df['cluster'].value_counts()
    

    # get the distance between each point and its nearest centroid. The biggest distances are considered as anomaly
    distance = getDistanceByPoint(data, kmeans[14])
    number_of_outliers = int(outliersFraction*len(distance))
    threshold = distance.nlargest(number_of_outliers).min()
    # anomaly21 contain the anomaly result of method 2.1 Cluster (0:normal, 1:anomaly) 
    df['anomaly21'] = (distance >= threshold).astype(int).tolist()
    
    df = df.reset_index()
    # visualisation of anomaly throughout time (viz 1)
    fig, ax = plt.subplots()
    
    a = df.loc[df['anomaly21'] == 1, ['Index', 'value']] #anomaly  
    ax.plot(df['Index'], df['value'], color='blue')
    ax.scatter(a['Index'],a['value'], color='red')
    ax.set_title('KMeans Outliers')
    plt.show()
    return df

def ellepticEnvelopeAnomaly(df,outliersFraction):
    
    # creation of 4 differents data set based on categories defined before
    df_class0 = df.loc[df['categories'] == 0, 'value']
    df_class1 = df.loc[df['categories'] == 1, 'value']
    df_class2 = df.loc[df['categories'] == 2, 'value']
    df_class3 = df.loc[df['categories'] == 3, 'value']
    

    # apply ellipticEnvelope(gaussian distribution) at each categories
    envelope =  EllipticEnvelope(contamination = outliersFraction) 
    X_train = df_class0.values.reshape(-1,1)
    envelope.fit(X_train)
    df_class0 = pd.DataFrame(df_class0)
    df_class0['deviation'] = envelope.decision_function(X_train)
    df_class0['anomaly'] = envelope.predict(X_train)
    
    envelope =  EllipticEnvelope(contamination = outliersFraction) 
    X_train = df_class1.values.reshape(-1,1)
    envelope.fit(X_train)
    df_class1 = pd.DataFrame(df_class1)
    df_class1['deviation'] = envelope.decision_function(X_train)
    df_class1['anomaly'] = envelope.predict(X_train)
    
    envelope =  EllipticEnvelope(contamination = outliersFraction) 
    X_train = df_class2.values.reshape(-1,1)
    envelope.fit(X_train)
    df_class2 = pd.DataFrame(df_class2)
    df_class2['deviation'] = envelope.decision_function(X_train)
    df_class2['anomaly'] = envelope.predict(X_train)
    
    envelope =  EllipticEnvelope(contamination = outliersFraction) 
    X_train = df_class3.values.reshape(-1,1)
    envelope.fit(X_train)
    df_class3 = pd.DataFrame(df_class3)
    df_class3['deviation'] = envelope.decision_function(X_train)
    df_class3['anomaly'] = envelope.predict(X_train)
    

    # add the data to the main 
    df_class = pd.concat([df_class0, df_class1, df_class2, df_class3])
    df['anomaly22'] = df_class['anomaly']
    df['anomaly22'] = np.array(df['anomaly22'] == -1).astype(int) 
    # visualisation of anomaly throughout time (viz 1)
    fig, ax = plt.subplots()
    a = df.loc[df['anomaly22'] == 1, ['time_epoch', 'value']] #anomaly  
    ax.plot(df['time_epoch'], df['value'], color='blue')
    ax.scatter(a['time_epoch'],a['value'], color='red')
    ax.set_title('Elliptic Envelope Multi Clustering')
    plt.show()
    return df

def IsolationForestAnomaly(df,outliersFraction):
    
    # Take useful feature and standardize them 
    data = df[['value','workTime','corruption']]            
    #data = df[['value', 'hours','workTime', 'daylight', 'DayOfTheWeek', 'WeekDay','corruption']]
    min_max_scaler = preprocessing.StandardScaler()
    np_scaled = min_max_scaler.fit_transform(data)
    data = pd.DataFrame(np_scaled)
    # train isolation forest 
    model =  IsolationForest(n_estimators = 100, 
                        max_samples = 256,
                        contamination = outliersFraction, 
                        random_state = np.random.RandomState(42))
    model.fit(data)
    # add the data to the main  
    df['anomaly23'] = pd.Series(model.predict(data))
    df['anomaly23'] = df['anomaly23'].map( {1: 0, -1: 1} )
    
    df = df.reset_index()
    fig, ax = plt.subplots()
    #df1 = df.tail(i*100).reset_index()
    df1 = df.reset_index()
    a = df1.loc[df['anomaly23'] == 1, ['index', 'value']] #anomaly  
    # visualisation of anomaly throughout time (viz 1)
    ax.plot(df['index'], df['value'], color='blue')
    ax.scatter(a['index'],a['value'], color='red')
    ax.set_title('Isolation Forest Forecasting')
    plt.show()
    
    return df,a


def oneClassSVMAnomaly(df,outliersFraction):
    data = df[['value','workTime','corruption']]  
    #data = df[['value', 'hours', 'daylight', 'DayOfTheWeek', 'WeekDay']]
    min_max_scaler = preprocessing.StandardScaler()
    np_scaled = min_max_scaler.fit_transform(data)
    # train one class SVM 
    model =  OneClassSVM(nu=0.95 * outliersFraction) #nu=0.95 * outliers_fraction  + 0.05
    data = pd.DataFrame(np_scaled)
    model.fit(data)
    # add the data to the main  
    df['anomaly26'] = pd.Series(model.predict(data))
    df['anomaly26'] = df['anomaly26'].map( {1: 0, -1: 1} )
    print(df['anomaly26'].value_counts())
    
    # visualisation of anomaly throughout time (viz 1)
    fig, ax = plt.subplots()
    
    a = df.loc[df['anomaly26'] == 1, ['time_epoch', 'value']] #anomaly  
    ax.plot(df['time_epoch'], df['value'], color='blue')
    ax.scatter(a['time_epoch'],a['value'], color='red')
    ax.set_title('One Class SVM Forecasting')
    plt.show()
    
    return df



