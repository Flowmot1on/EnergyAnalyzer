# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 09:08:11 2020

@author: yunusemreemik
"""

import pyodbc
import configparser 
import pandas as pd

def configure():
    config = configparser.ConfigParser()
    config.read('config.ini')  
    global conn
    conn = pyodbc.connect(Driver = config['MsSqlDB']['Driver'],
                                   Server = config['MsSqlDB']['Server'],
                                   Database = config['MsSqlDB']['Database'],
                                   User = config['MsSqlDB']['user'],
                                   Password = config['MsSqlDB']['pass'],
                                   Trusted_connection =config['MsSqlDB']['Trusted_connection'])
    
def readLog(deviceId,pastTime):
    
    configure()
    
    query = "EXEC SP_getDeviceLog {},{}".format(deviceId,pastTime)
    
    log = pd.read_sql_query(str(query), conn)
    
    return log

def readClients():
    
    configure()
    
    devices = pd.read_sql_query("EXEC SP_getDevices", conn)
    
    return devices

def createAnomalyReport(deviceID,objectID,algorithmId,firstLogID,lastLogID):
    
    cursor = conn.cursor()
    
    params = (int(deviceID),int(objectID),int(algorithmId),int(firstLogID),int(lastLogID))
   
    cursor.execute("EXEC [SPCreateAnomalyLogId] ?,?,?,?,?;",params)

    conn.commit()
    
   

def writeDb(conn,annmly,dvc,obj,algorithmId):
    
    con1 = True
    
    while(con1):
        
        try:
            
            createAnomalyReport(conn,annmly,dvc,obj,algorithmId)
            
            con1 = False
            
        except Exception as e:
            
            print (e)
            


