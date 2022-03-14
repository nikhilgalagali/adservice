import sys
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import tools.normalize_data as normalize_data

from statsmodels.tsa.stattools import adfuller
from pandas.tseries.offsets import DateOffset
import logging


import tools.time_series_metadata as time_series_metadata
from tools.long_term_seasonal_plus_trend import long_term_seasonal_plus_trend

def log_transform(df_complete_data,log_transform):
 
    numberOfColumns=len(log_transform)
    for c in range(numberOfColumns-1):
        if log_transform[c]==1:
            df_complete_data.iloc[:,c]=np.log(df_complete_data.iloc[:,c])
            
            
def exponentiate(df_complete_data,log_transform):
 
    numberOfColumns=len(log_transform)
            
    for c in range(1,numberOfColumns):        
        if log_transform[c-1]==1:        
            # Retransform data to original scale
            df_new.loc[:,'KPI_{}'.format(c)]=np.exp(df_new.loc[:,'KPI_{}'.format(c)])
            df_new.loc[:,'Expected_{}'.format(c)]=np.exp(df_new.loc[:,'Expected_{}'.format(c)])    
            df_new.loc[:,'forecast_test_{}'.format(c)]=np.exp(df_new.loc[:,'forecast_test_{}'.format(c)])            
            
                        
def data_transformer(df_complete_data,df_label_data,operation_mode,data_scale,training_window,test_window,kpi_name,perform_point_anomaly,misc_values,trend_anomaly_window,data_frequency):
    
    originalColumnNames=df_complete_data.columns
    numberOfColumns=len(df_complete_data.columns)
    
    columnNames=[]
    for c in range(numberOfColumns):
        if c==0:
            columnNames.append('Date')
        else:
            columnNames.append('KPI_{}'.format(c))

    df_complete_data.columns=columnNames
    
    if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
        logging.debug('{}: Normalizing data'.format(kpi_name))
    else:
        logging.debug('Normalizing data')  
        
    # Normalize dataset
    if (operation_mode=='executor' or operation_mode=='tester') and perform_point_anomaly:
        normalize_data.pred_normalize_data(df_complete_data,data_scale)        
    else:    
        data_scale=normalize_data.normalize_data(df_complete_data)

    if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
        logging.debug('{}: SUCCESS Normalizing data'.format(kpi_name))
    else:
        logging.debug('SUCCESS Normalizing data')
                
    #########################################################
    ## TRANSFORM df-complete-data from dataframe to timeseries dataframe
    #########################################################
            
    df_complete_data.set_index('Date', inplace=True, drop=True)
    df_complete_data.index = pd.to_datetime(df_complete_data.index)
    
    if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
        df_label_data.columns=columnNames
        df_label_data.set_index('Date', inplace=True, drop=True)
        df_label_data.index = pd.to_datetime(df_label_data.index)
        
    ###########################################
    ## UPDATE TO TRAINING WINDOW END TIMESTAMP
    ###########################################
                
    if (not isinstance(training_window,float)):
        training_window[1]=df_complete_data.index[int(0.8*len(df_complete_data.loc[:test_window]))] 
        
    # TBD
    #data_transformation.log_transform(df_complete_data,log_transform)

    if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
        logging.debug('{}: Extracting frequency, seasonal periods and harmonics'.format(kpi_name))
    else:
        logging.debug('Extracting frequency, seasonal periods and harmonics')
        
    # Determine data frequency and cycle period
    if operation_mode in ['trainer','tester','executor']:
        df_complete_data,data_frequency,cycle_period,seasonal_periods,harmonics=\
        time_series_metadata.production_data_frequency(df_complete_data,data_frequency,training_window,\
                                                       operation_mode,kpi_name)
    else:
        df_complete_data,data_frequency,training_window,test_window,\
                                            cycle_period,seasonal_periods,harmonics=\
        time_series_metadata.data_frequency(df_complete_data,data_frequency,training_window,\
                                            test_window,operation_mode,kpi_name)
        

    # Number data points in which to display trend anomalies
    if trend_anomaly_window==None:
        trend_anomaly_window=4*cycle_period 

    if data_frequency!=None:
                
        if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
            logging.debug('{}: SUCCESS Extracting frequency, seasonal periods and harmonics'.format(kpi_name))
        else:
            logging.debug('SUCCESS Extracting frequency, seasonal periods and harmonics')

        if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
            logging.debug('{}: Missing time stamp imputation'.format(kpi_name))
        else:
            logging.debug('Missing time stamp imputation')

        # Missing time stamp imputation
        df_complete_data=time_series_metadata.missing_timestamp_imputation(df_complete_data,data_frequency)    

        if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
            df_label_data=time_series_metadata.missing_label_imputation(df_label_data,data_frequency)

        if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
            logging.debug('{}: SUCCESS Missing time stamp imputation'.format(kpi_name))
        else:
            logging.debug('SUCCESS Missing time stamp imputation')
                
        if len(misc_values)>0 and data_frequency=='daily':
            misc_values['prev_christmas']['date']=pd.to_datetime(misc_values['prev_christmas']['date'])
            misc_values['prev_DAT']['date']=pd.to_datetime(misc_values['prev_DAT']['date'])
            misc_values['prev_major_iOS_release']['date']=\
            pd.to_datetime(misc_values['prev_major_iOS_release']['date'])
            misc_values['prev_major_iPhone_release']['date']=\
            pd.to_datetime(misc_values['prev_major_iPhone_release']['date'])
            
            misc_values['prev_christmas']['value']=\
            float(misc_values['prev_christmas']['value'])/10**data_scale[0]
            
            misc_values['prev_DAT']['value']=\
            float(misc_values['prev_DAT']['value'])/10**data_scale[0]
            
            misc_values['prev_major_iOS_release']['value']=\
            float(misc_values['prev_major_iOS_release']['value'])/10**data_scale[0]
            
            misc_values['prev_major_iPhone_release']['value']=\
            float(misc_values['prev_major_iPhone_release']['value'])/10**data_scale[0]
            
        
        
    return df_complete_data,df_label_data,data_scale,training_window,test_window,data_frequency,cycle_period,seasonal_periods,\
           harmonics,trend_anomaly_window,originalColumnNames,numberOfColumns,misc_values

def prior_outlier_detection(df_complete_data,df_label_data,us_holiday_flag,chinese_holiday_flag,\
                            operation_mode,kpi_name,training_window,cycle_period,plottingWindow,model_type):
    # This function does the prior outlier detection
    
    if isinstance(training_window,float):
        
        df_training=df_complete_data.iloc[:int(training_window*len(df_complete_data))]

        # outliers in training data and within display window based on past info
        df_display_training=df_training.iloc[int(training_window*len(df_complete_data))-\
                        int(plottingWindow*training_window*len(df_complete_data)):]
        
    elif isinstance(training_window,list):
        
        df_training=df_complete_data.loc[training_window[0]:training_window[1]]

        # outliers in training data and within display window based on past info
        df_display_training=df_training.loc[training_window[0]:]        
    
    df_anom_training=df_display_training[df_display_training['KPI_1'].isnull()]
    count_of_anomalies_training=len(df_anom_training)    

    if model_type not in ['sr','sr_batch']:
        
        # Automatic prior outlier identification requires atleast two cycles of data
        try:

            if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                logging.debug('{}: STL computation'.format(kpi_name))
            else:
                logging.debug('STL computation')

            decomp=long_term_seasonal_plus_trend(df_training['KPI_1'].reset_index(),cycle_period)    
            seasonal_plus_trend=decomp['seasonal_plus_trend']
            diff_array=df_training['KPI_1']-seasonal_plus_trend['trend']-seasonal_plus_trend['seasonal']    

            if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                logging.debug('{}: SUCCESS STL computation'.format(kpi_name))
            else:
                logging.debug('SUCCESS STL computation')

            for t in range(len(df_training)):

                if np.abs(diff_array[t])/(6*np.abs(diff_array).median())>1:

                        # Check if user has provided any feedback
                        if (operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor') and \
                            (df_label_data.loc[df_training.index[t]]!='N')[0]:

                            if (df_label_data.loc[df_training.index[t]]=='Y')[0]:  # User sets the datapoint as anomaly

                                if isinstance(training_window,float):

                                    if t>=int(training_window*len(df_complete_data))-\
                                    int(plottingWindow*training_window*len(df_complete_data)):       

                                        if count_of_anomalies_training==0:
                                            df_anom_training=df_training.loc[df_training.index[t:t+1]]
                                        else:
                                            df_anom_training=df_anom_training.append(df_training.loc[df_training.index[t:t+1]])

                                        count_of_anomalies_training=count_of_anomalies_training+1

                                else:

                                    if df_training.index[t]>=training_window[0]:       

                                        if count_of_anomalies_training==0:
                                            df_anom_training=df_training.loc[df_training.index[t:t+1]]
                                        else:
                                            df_anom_training=df_anom_training.append(df_training.loc[df_training.index[t:t+1]])

                                        count_of_anomalies_training=count_of_anomalies_training+1                                

                                df_training.loc[df_training.index[t:t+1],'KPI_1']=np.nan

                            elif (us_holiday_flag.iloc[t]==0 and chinese_holiday_flag.iloc[t]==0):                       # not special, can be outlier

                                if isinstance(training_window,float):

                                    if t>=int(training_window*len(df_complete_data))-\
                                    int(plottingWindow*training_window*len(df_complete_data)):       

                                        if count_of_anomalies_training==0:
                                            df_anom_training=df_training.loc[df_training.index[t:t+1]]
                                        else:
                                            df_anom_training=df_anom_training.append(df_training.loc[df_training.index[t:t+1]])

                                        count_of_anomalies_training=count_of_anomalies_training+1   

                                else:

                                    if df_training.index[t]>=training_window[0]:       

                                        if count_of_anomalies_training==0:
                                            df_anom_training=df_training.loc[df_training.index[t:t+1]]
                                        else:
                                            df_anom_training=df_anom_training.append(df_training.loc[df_training.index[t:t+1]])

                                        count_of_anomalies_training=count_of_anomalies_training+1                                  

                                df_training.loc[df_training.index[t:t+1],'KPI_1']=np.nan

                        else:    


                            if (us_holiday_flag.iloc[t]==0 and chinese_holiday_flag.iloc[t]==0):                         # not special, can be outlier

                                if isinstance(training_window,float):

                                    if t>=int(training_window*len(df_complete_data))-\
                                    int(plottingWindow*training_window*len(df_complete_data)):

                                        if count_of_anomalies_training==0:
                                            df_anom_training=df_training.loc[df_training.index[t:t+1]]
                                        else:
                                            df_anom_training=df_anom_training.append(df_training.loc[df_training.index[t:t+1]])

                                        count_of_anomalies_training=count_of_anomalies_training+1 

                                else:

                                    if df_training.index[t]>=training_window[0]:

                                        if count_of_anomalies_training==0:
                                            df_anom_training=df_training.loc[df_training.index[t:t+1]]
                                        else:
                                            df_anom_training=df_anom_training.append(df_training.loc[df_training.index[t:t+1]])

                                        count_of_anomalies_training=count_of_anomalies_training+1                                 


                                df_training.loc[df_training.index[t:t+1],'KPI_1']=np.nan

        except:

            if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                logging.debug('{}: STL computation FAILED. Ignoring automatic outlier detection'.format(kpi_name))
            else:
                logging.debug('STL computation FAILED. Ignoring automatic outlier detection')
                    
    if isinstance(training_window,float):
        df_test=df_complete_data.iloc[int(training_window*len(df_complete_data)):]
    else:    
        df_test=df_complete_data.iloc[len(df_complete_data.loc[training_window[0]:training_window[1]]):]
    
    return df_training,df_test,df_anom_training,count_of_anomalies_training