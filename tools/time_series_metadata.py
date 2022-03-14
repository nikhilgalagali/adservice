import pandas as pd
import matplotlib
matplotlib.use('agg')
import numpy as np
import logging
import time
from datetime import datetime
import holidays
from workalendar.asia import China
from pandas.tseries.offsets import DateOffset,Day,MonthBegin,Hour


# This is called in production
def production_data_frequency(df_complete_data,data_frequency,training_window,operation_mode,kpi_name):
 
    # Determine the data frequency
                
    if data_frequency=='MIN':

        data_frequency='minutely'
        cycle_period=60

        
        if int(training_window*len(df_complete_data))>(30+int(1440.0/128)+int(1440.0*7.0/128)+10):  #(hourly,daily,weekly)
            
            seasonal_periods=[60.0,1440.0,1440.0*7]
            harmonics=[int(np.floor(60.0/4)),int(np.floor(1440.0/128)),int(np.floor(1440.0*7.0/128))] 
            
        elif int(training_window*len(df_complete_data))>(30+int(1440.0/128)+10):
            
            seasonal_periods=[60.0,1440.0]
            harmonics=[int(np.floor(60.0/4)),int(np.floor(1440.0/128))]

        elif int(training_window*len(df_complete_data))>(30+10):
            
            seasonal_periods=[60.0]
            harmonics=[int(np.floor(60.0/4))]

        else:
            seasonal_periods=[]
            harmonics=[]

    elif data_frequency=='H':    
        data_frequency='hourly'    
        cycle_period=24

        if int(training_window*len(df_complete_data))>(24+int(168.0/32)+10):            
            seasonal_periods=[24.0,168.0]
            harmonics=[int(np.floor(24.0/2)),int(np.floor(168.0/32))]

        elif int(training_window*len(df_complete_data))>(24+10):
            seasonal_periods=[24.0]
            harmonics=[int(np.floor(24.0/2))]

        else:
            sesaonal_periods=[]
            harmonics=[]
        
    elif data_frequency=='D':
        data_frequency='daily'
        cycle_period=7


        if int(training_window*len(df_complete_data))>(7+31.0+366.0+10): # weekly, monthly, yearly seasonality
            seasonal_periods=[7.0,30.5,365.5]
            harmonics=[int(np.floor(7.0/2)),int(np.floor(30.5/2)),int(np.floor(365.5/2))]          

        elif int(training_window*len(df_complete_data))>(7+31+10): # weekly, monthly seasonality
            seasonal_periods=[7.0,30.5]
            harmonics=[int(np.floor(7.0/2)),int(np.floor(30.5/2))]            

        elif int(training_window*len(df_complete_data))>(7+10): # weekly seasonality
            seasonal_periods=[7.0]
            harmonics=[int(np.floor(7.0/2))]

        else:                              # no seasonality
            seasonal_periods=[]
            harmonics=[]

    elif data_frequency=='W':
        data_frequency='weekly'    
        cycle_period=4

        if int(training_window*len(df_complete_data))>(4+int(365.5/7)+10):            
            seasonal_periods=[4.0,365.5/7]
            harmonics=[int(np.floor(4.0/2)),int(np.floor(365.5/7/8))]

        elif int(training_window*len(df_complete_data))>(4+10):
            seasonal_periods=[4.0]
            harmonics=[int(np.floor(4.0/2))]

        else:
            seasonal_periods=[]
            harmonics=[]

    elif data_frequency=='M':    
        data_frequency='monthly'        
        # Transforming date to first day of month
        df_complete_data=df_complete_data.shift(-1,freq='MS')        
        cycle_period=12

        if int(training_window*len(df_complete_data))>(12+10):
            seasonal_periods=[12.0]
            harmonics=[int(np.floor(12.0/2))]
        else:
            seasonal_periods=[]
            harmonics=[]            

    else:
            
        logging.debug('{}: FAILED DATA FREQUENCY DETERMINATION. PASSED DATA FREQUENCY NOT AMONG W,H,D,M,MIN'.format(kpi_name))

        # Setting null values    
        data_frequency=None
        cycle_period=0
        seasonal_periods=None
        harmonics=None    
        
    return df_complete_data,data_frequency,cycle_period,seasonal_periods,harmonics

# This is called in validation mode
def data_frequency(df_complete_data,data_frequency,training_window,test_window,operation_mode,kpi_name):
 
    # Determine the data frequency
    
    if isinstance(training_window,float):


        # TRAINING WINDOW IN TERMS OF FRACTION OF THE DATA TIME SPANS
    
        if df_complete_data.index[1]==df_complete_data.tshift(1,freq='T').index[0]:

            data_frequency='minutely'
            cycle_period=60

            # +10 is for all other parameters besides seasonality
            if int(training_window*len(df_complete_data))>(30+int(1440.0/128)+int(1440.0*7.0/128)+10):  #(hourly,daily,weekly)
            
                seasonal_periods=[60.0,1440.0,1440.0*7]
                harmonics=[int(np.floor(60.0/4)),int(np.floor(1440.0/128)),int(np.floor(1440.0*7.0/128))] 
                      
            elif int(training_window*len(df_complete_data))>(30+int(1440.0/128)+10):            
                seasonal_periods=[60.0,1440.0]
                harmonics=[int(np.floor(60.0/4)),int(np.floor(1440.0/128))]

            elif int(training_window*len(df_complete_data))>(30+10):
                seasonal_periods=[60.0]
                harmonics=[int(np.floor(60.0/4))]

            else:
                seasonal_periods=[]
                harmonics=[]

        elif df_complete_data.index[1]==df_complete_data.tshift(1,freq='5T').index[0]:

            data_frequency='5minutely'
            cycle_period=12

            if int(training_window*len(df_complete_data))>(12+int(288.0/24)+int(288.0*7.0/24)+10):            
                seasonal_periods=[12.0,288.0,288.0*7]
                harmonics=[int(np.floor(12.0/2)),int(np.floor(288.0/24)),int(np.floor(288.0*7.0/24))]
            
            elif int(training_window*len(df_complete_data))>(12+int(288.0/24)+10):            
                seasonal_periods=[12.0,288.0]
                harmonics=[int(np.floor(12.0/2)),int(np.floor(288.0/24))]

            elif int(training_window*len(df_complete_data))>(12+10):
                seasonal_periods=[12.0]
                harmonics=[int(np.floor(12.0/2))]

            else:
                seasonal_periods=[]
                harmonics=[]

        elif df_complete_data.index[1]==df_complete_data.tshift(1,freq='H').index[0]:    
            data_frequency='hourly'    
            cycle_period=24

            if int(training_window*len(df_complete_data))>(24+int(168.0/32)+10):            
                seasonal_periods=[24.0,168.0]
                harmonics=[int(np.floor(24.0/2)),int(np.floor(168.0/32))]

            elif int(training_window*len(df_complete_data))>(24+10):
                seasonal_periods=[24.0]
                harmonics=[int(np.floor(24.0/2))]

            else:
                sesaonal_periods=[]
                harmonics=[]
                
                
        elif df_complete_data.index[1]==df_complete_data.tshift(1,freq='D').index[0]:
            data_frequency='daily'
            cycle_period=7

            # +10 is for all other parameters except the seasonality parameters 
            if int(training_window*len(df_complete_data))>(7+31+366.0+10):# weekly, monthly, yearly seasonality
                seasonal_periods=[7.0,30.5,365.5]
                harmonics=[int(np.floor(7.0/2)),int(np.floor(30.5/2)),int(np.floor(365.5/2))]          

            elif int(training_window*len(df_complete_data))>(7+31+10): # weekly, monthly seasonality
                seasonal_periods=[7.0,30.5]
                harmonics=[int(np.floor(7.0/2)),int(np.floor(30.5/2))]            

            elif int(training_window*len(df_complete_data))>(7+10): # weekly seasonality
                seasonal_periods=[7.0]
                harmonics=[int(np.floor(7.0/2))]

            else:                              # no seasonality
                seasonal_periods=[]
                harmonics=[]

        elif df_complete_data.index[1]==df_complete_data.tshift(1,freq='7D').index[0]:
            data_frequency='weekly'    
            cycle_period=4

            if int(training_window*len(df_complete_data))>(4+int(365.5/7)+10):            
                seasonal_periods=[4.0,365.5/7]
                harmonics=[int(np.floor(4.0/2)),int(np.floor(365.5/7/2))]

            elif int(training_window*len(df_complete_data))>(4+10):
                seasonal_periods=[4.0]
                harmonics=[int(np.floor(4.0/2))]

            else:
                seasonal_periods=[]
                harmonics=[]

        elif df_complete_data.index[1]==df_complete_data.tshift(1,freq='M').index[0]:    
            data_frequency='monthly'        
            # Transforming date to first day of month
            df_complete_data=df_complete_data.shift(-1,freq='MS')        
            cycle_period=12

            if int(training_window*len(df_complete_data))>(12+10):
                seasonal_periods=[12.0]
                harmonics=[int(np.floor(12.0/2))]
            else:
                seasonal_periods=[]
                harmonics=[]            

        elif df_complete_data.index[1]==df_complete_data.tshift(1,freq='MS').index[0]:
            data_frequency='monthly'    
            cycle_period=12     

            if int(training_window*len(df_complete_data))>(12+10):
                seasonal_periods=[12.0]
                harmonics=[int(np.floor(12.0/2))]
            else:
                seasonal_periods=[]
                harmonics=[]

        else:
            if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                logging.debug('{}: FAILED DATA FREQUENCY DETERMINATION. FIRST TWO TIMESTAMPS NOT AT AN ALLOWED INTERVAL.'.format(kpi_name))
            else:
                logging.debug('FAILED DATA FREQUENCY DETERMINATION. FIRST TWO TIMESTAMPS NOT AT AN ALLOWED INTERVAL.')

            # Setting null values    
            data_frequency=None
            cycle_period=0
            seasonal_periods=None
            harmonics=None
        
    else:
        
        # TRAINING WINDOW IN TERMS oF TIME STAMPS

        if df_complete_data.index[1]==df_complete_data.tshift(1,freq='T').index[0]:

            data_frequency='minutely'
            cycle_period=60
            '''
            if len(df_complete_data.loc[training_window[0]:training_window[1]])>(30+int(1440.0/128)+int(1440.0*7.0/128)+10):  #(hourly,daily,weekly)
            
                seasonal_periods=[60.0,1440.0,1440.0*7]
                harmonics=[int(np.floor(60.0/4)),int(np.floor(1440.0/128)),int(np.floor(1440.0*7.0/128))] 
            
            el
            '''
            if len(df_complete_data.loc[training_window[0]:training_window[1]])>(30+int(1440.0/128)+10):
                seasonal_periods=[60.0,1440.0]
                harmonics=[int(np.floor(60.0/4)),int(np.floor(1440.0/128))]
            
            elif len(df_complete_data.loc[training_window[0]:training_window[1]])>(30+10):
                seasonal_periods=[60.0]
                harmonics=[int(np.floor(60.0/4))]

            else:
                seasonal_periods=[]
                harmonics=[]

        elif df_complete_data.index[1]==df_complete_data.tshift(1,freq='5T').index[0]:

            data_frequency='5minutely'
            cycle_period=12
            '''
            if len(df_complete_data.loc[training_window[0]:training_window[1]])>(12+int(288.0/24)+int(288.0*7.0/24)+10):   
                seasonal_periods=[12.0,288.0,288.0*7]
                harmonics=[int(np.floor(12.0/2)),int(np.floor(288.0/24)),int(np.floor(288.0*7.0/24))]
            
            el
            '''
            if len(df_complete_data.loc[training_window[0]:training_window[1]])>(12+int(288.0/24)+10):
                seasonal_periods=[12.0,288.0]
                harmonics=[int(np.floor(12.0/2)),int(np.floor(288.0/24))]

            elif len(df_complete_data.loc[training_window[0]:training_window[1]])>(12+10):
                seasonal_periods=[12.0]
                harmonics=[int(np.floor(12.0/2))]

            else:
                seasonal_periods=[]
                harmonics=[]

        elif df_complete_data.index[1]==df_complete_data.tshift(1,freq='H').index[0]:    
            data_frequency='hourly'    
            cycle_period=24

            if len(df_complete_data.loc[training_window[0]:training_window[1]])>(24+int(168.0/32)+10):            
                seasonal_periods=[24.0,168.0]
                harmonics=[int(np.floor(24.0/2)),int(np.floor(168.0/32))]

            elif len(df_complete_data.loc[training_window[0]:training_window[1]])>(24+10):
                seasonal_periods=[24.0]
                harmonics=[int(np.floor(24.0/2))]

            else:
                seasonal_periods=[]
                harmonics=[]
                                
        elif df_complete_data.index[1]==df_complete_data.tshift(1,freq='D').index[0]:
            data_frequency='daily'
            cycle_period=7


            if len(df_complete_data.loc[training_window[0]:training_window[1]])>(7+31+366.0+10): # weekly, monthly, yearly seasonality
                seasonal_periods=[7.0,30.5,365.5]
                harmonics=[int(np.floor(7.0/2)),int(np.floor(30.5/2)),int(np.floor(365.5/2))]          

            elif len(df_complete_data.loc[training_window[0]:training_window[1]])>(7+31+10): # weekly, monthly seasonality
                seasonal_periods=[7.0,30.5]
                harmonics=[int(np.floor(7.0/2)),int(np.floor(30.5/2))]            

            elif len(df_complete_data.loc[training_window[0]:training_window[1]])>(7+10): # weekly seasonality
                seasonal_periods=[7.0]
                harmonics=[int(np.floor(7.0/2))]

            else:                              # no seasonality
                seasonal_periods=[]
                harmonics=[]

        elif df_complete_data.index[1]==df_complete_data.tshift(1,freq='7D').index[0]:
            data_frequency='weekly'    
            cycle_period=4

            if len(df_complete_data.loc[training_window[0]:training_window[1]])>(4+int(365.5/7)+10):            
                seasonal_periods=[4.0,365.5/7]
                harmonics=[int(np.floor(4.0/2)),int(np.floor(365.5/7/2))]

            elif len(df_complete_data.loc[training_window[0]:training_window[1]])>(4+10):
                seasonal_periods=[4.0]
                harmonics=[int(np.floor(4.0/2))]

            else:
                seasonal_periods=[]
                harmonics=[]

        elif df_complete_data.index[1]==df_complete_data.tshift(1,freq='M').index[0]:    
            data_frequency='monthly'  
            
            # Transforming date to first day of month
            df_complete_data=df_complete_data.shift(-1,freq='MS')
                        
            training_window[0]=training_window[0]-MonthBegin()
            training_window[1]=training_window[1]-MonthBegin()
            test_window=test_window-MonthBegin()
            
            cycle_period=12

            if len(df_complete_data.loc[training_window[0]:training_window[1]])>(12+10):
                seasonal_periods=[12.0]
                harmonics=[int(np.floor(12.0/2))]
            else:
                seasonal_periods=[]
                harmonics=[]            

        elif df_complete_data.index[1]==df_complete_data.tshift(1,freq='MS').index[0]:
            data_frequency='monthly'    
            cycle_period=12     

            if len(df_complete_data.loc[training_window[0]:training_window[1]])>(12+10):
                seasonal_periods=[12.0]
                harmonics=[int(np.floor(12.0/2))]
            else:
                seasonal_periods=[]
                harmonics=[]

        else:
            if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                logging.debug('{}: FAILED DATA FREQUENCY DETERMINATION. FIRST TWO TIMESTAMPS NOT AT AN ALLOWED INTERVAL.'.format(kpi_name))
            else:
                logging.debug('FAILED DATA FREQUENCY DETERMINATION. FIRST TWO TIMESTAMPS NOT AT AN ALLOWED INTERVAL.')

            # Setting null values    
            data_frequency=None
            cycle_period=0
            seasonal_periods=None
            harmonics=None        
        
        
    return df_complete_data,data_frequency,training_window,test_window,\
           cycle_period,seasonal_periods,harmonics

def missing_timestamp_imputation(df_complete_data,data_frequency):
    
    # This function checks if there is any missing timestamp and
    # imputes it with NAN; note missing data (NAN) is handled by 
    # models
            
    if data_frequency=='minutely':    
        df_complete_data=df_complete_data.resample('T').mean()
        
    elif data_frequency=='5minutely':    
        df_complete_data=df_complete_data.resample('5T').mean()
        
    elif data_frequency=='daily':
        df_complete_data=df_complete_data.resample('D').mean()
        
    elif data_frequency=='weekly':
        
        if df_complete_data.index[0].weekday()==0:
            df_complete_data=df_complete_data.resample('W-MON').mean()
        elif df_complete_data.index[0].weekday()==1:
            df_complete_data=df_complete_data.resample('W-TUE').mean()
        elif df_complete_data.index[0].weekday()==2:
            df_complete_data=df_complete_data.resample('W-WED').mean()
        elif df_complete_data.index[0].weekday()==3:
            df_complete_data=df_complete_data.resample('W-THU').mean()
        elif df_complete_data.index[0].weekday()==4:
            df_complete_data=df_complete_data.resample('W-FRI').mean()
        elif df_complete_data.index[0].weekday()==5:
            df_complete_data=df_complete_data.resample('W-SAT').mean()
        elif df_complete_data.index[0].weekday()==6:
            df_complete_data=df_complete_data.resample('W-SUN').mean()
            
    elif data_frequency=='monthly':
        df_complete_data=df_complete_data.resample('MS').mean()
    elif data_frequency=='hourly':
        df_complete_data=df_complete_data.resample('H').mean()
    else:
        print('Data frequency not expected\n Continuing without timestamp imputatation\n')
            
            
    return df_complete_data 

def missing_label_imputation(df_complete_data,data_frequency):
    
    # This function checks if there is any missing timestamp and
    # imputes it with its expected value, the label value is imputed with NAN;
    # note missing data (NAN) is handled by models
    
    if data_frequency=='minutely':
        df_complete_data=df_complete_data.resample('T').first()
        
    elif data_frequency=='5minutely':
        df_complete_data=df_complete_data.resample('5T').first()
        
    elif data_frequency=='daily':
        df_complete_data=df_complete_data.resample('D').first()
        
    elif data_frequency=='weekly':
        
        if df_complete_data.index[0].weekday()==0:
            df_complete_data=df_complete_data.resample('W-MON').first()
        elif df_complete_data.index[0].weekday()==1:
            df_complete_data=df_complete_data.resample('W-TUE').first()
        elif df_complete_data.index[0].weekday()==2:
            df_complete_data=df_complete_data.resample('W-WED').first()
        elif df_complete_data.index[0].weekday()==3:
            df_complete_data=df_complete_data.resample('W-THU').first()
        elif df_complete_data.index[0].weekday()==4:
            df_complete_data=df_complete_data.resample('W-FRI').first()
        elif df_complete_data.index[0].weekday()==5:
            df_complete_data=df_complete_data.resample('W-SAT').first()
        elif df_complete_data.index[0].weekday()==6:
            df_complete_data=df_complete_data.resample('W-SUN').first()
            
    elif data_frequency=='monthly':
        df_complete_data=df_complete_data.resample('MS').first()
    elif data_frequency=='hourly':
        df_complete_data=df_complete_data.resample('H').first()
    else:
        print('Data frequency not expected\n Continuing without timestamp imputatation\n')
            
    return df_complete_data 

def get_holiday_flag(df_complete_data,data_frequency,holidays_and_releases,misc_values):

    us_holiday_flag=df_complete_data.copy().iloc[:,0]
    us_holiday_flag.name='US_holidays'
    chinese_holiday_flag=df_complete_data.copy().iloc[:,0]
    chinese_holiday_flag.name='Chinese_holidays'
    
    if data_frequency=='daily' and holidays_and_releases:
        
        # reading us and chinese holidays
        us_holidays = holidays.UnitedStates()
        cal=China()
        
        chinese_holidays=pd.DataFrame(cal.holidays(2018),columns=['Date','Name'])
        chinese_holidays=chinese_holidays.append(pd.DataFrame(cal.holidays(2019),columns=['Date','Name']))
        chinese_holidays.Date=pd.to_datetime(chinese_holidays.Date)        
        chinese_holidays.set_index('Date',inplace=True)        
                                  
        for t in df_complete_data.index:
                
            if t in us_holidays:
                # Christmas and Day after Thanksgiving are treated in a special manner
                if len(misc_values)>0:
                    if t!=misc_values['prev_christmas']['date'] and\
                       t!=misc_values['prev_DAT']['date']:
                        us_holiday_flag.loc[t]=1
                    else:
                        us_holiday_flag.loc[t]=0
                else:
                    us_holiday_flag.loc[t]=1
            else:
                us_holiday_flag.loc[t]=0
                
            if t in chinese_holidays.index:
                chinese_holiday_flag.loc[t]=1
            else:
                chinese_holiday_flag.loc[t]=0
        
    elif data_frequency=='weekly' and holidays_and_releases:
        
        # reading us and chinese holidays
        us_holidays = holidays.UnitedStates()
        # Adding years to the us_holidays object        
        for year in range(2000,2030):
            datetime(year, 1, 1) in us_holidays
        cal=China()
                
        chinese_holidays=pd.DataFrame(cal.holidays(2018),columns=['Date','Name'])
        chinese_holidays=chinese_holidays.append(pd.DataFrame(cal.holidays(2019),columns=['Date','Name']))
        chinese_holidays.Date=pd.to_datetime(chinese_holidays.Date)
        chinese_holidays.set_index('Date',inplace=True)        
                                    
        for t in df_complete_data.index:
                                
            # check if there was a holiday in the week, assuming date t refers to the end of the week            
            if pd.date_range(t-6*Day(),periods=7,freq='D').isin(us_holidays).any():
                us_holiday_flag.loc[t]=1                
            else:
                us_holiday_flag.loc[t]=0
                
            if pd.date_range(t-6*Day(),periods=7,freq='D').isin(chinese_holidays.index).any():
                chinese_holiday_flag.loc[t]=1
            else:
                chinese_holiday_flag.loc[t]=0        

                                
    else:
        
        us_holiday_flag[:]=0
        chinese_holiday_flag[:]=0
        
        
    return [us_holiday_flag,chinese_holiday_flag]

# THIS FUNCTION DOES NOT VECTORIZE NUMPY COMPUTATIONS
def construct_seasonal_inputs_(df,seasonal_periods,harmonics,training_data_start_time=None,data_frequency=None):
    # seasonal_periods: all seasonalities that are considered

    start_time=time.time()    
    
    df_complete_data=df.copy()

    time_start=0
    
    # This is required when the prediction is required on dataset not used for training
    if training_data_start_time!=None:
        
        if data_frequency=='minutely':
                offset=Minute()
        elif data_frequency=='5minutely':
                offset=Minute(5)
        elif data_frequency=='hourly':
                offset=Hour()
        elif data_frequency=='daily':
                offset=Day()
        elif data_frequency=='weekly':
                offset=Day(7)
        elif data_frequency=='monthly':
                offset=MonthBegin()
                
        while training_data_start_time+time_start*offset!=df_complete_data.index[0]:
            time_start=time_start+1    
    
    
    seasonal_params_names=[]
    for t in range(time_start,len(df_complete_data)+time_start):

        #print t
        
        curr_t_inputs=[]
        for n_s in range(len(seasonal_periods)):

            # Keeping a higher parameter granularity for the first seasonlity period
            for j in range(1,harmonics[n_s]+1):

                lambda_j=(2.0*np.pi*j)/(seasonal_periods[n_s])

                curr_t_inputs.append(np.cos(lambda_j*t))
                curr_t_inputs.append(np.sin(lambda_j*t))

                if t==time_start:
                    seasonal_params_names.append('Seasonality_{}_harmonic_{}_cos'.format(seasonal_periods[n_s],j))
                    seasonal_params_names.append('Seasonality_{}_harmonic_{}_sin'.format(seasonal_periods[n_s],j))

                #print np.cos(lambda_j*t).shape    

        #print curr_t_inputs
                            
        if t==time_start:
            seasonal_inputs=np.array([curr_t_inputs])
        else:     
            seasonal_inputs=np.append(seasonal_inputs,np.array([curr_t_inputs]),axis=0)
                                
            
    end_time=time.time()
    
    #print 'Time difference',end_time-start_time
                    
    return pd.DataFrame(seasonal_inputs,index=df_complete_data.index,columns=seasonal_params_names)

# THIS FUNCTION USES VECTORIZING OF NUMPY OPERATIONS
def construct_seasonal_inputs(df,seasonal_periods,harmonics,training_data_start_time=None,data_frequency=None):
    # seasonal_periods: all seasonalities that are considered
        
    start_time=time.time()            
    df_complete_data=df.copy()

    time_start=0
    
    # This is required when the prediction is required on dataset not used for training
    if training_data_start_time!=None:
        
        if data_frequency=='minutely':
                offset=Minute()
        elif data_frequency=='5minutely':
                offset=Minute(5)
        elif data_frequency=='hourly':
                offset=Hour()
        elif data_frequency=='daily':
                offset=Day()
        elif data_frequency=='weekly':
                offset=Day(7)
        elif data_frequency=='monthly':
                offset=MonthBegin()
                
        while training_data_start_time+time_start*offset!=df_complete_data.index[0]:
            time_start=time_start+1    
    
    
    seasonal_params_names=[]
        
    if len(seasonal_periods)==0:  # no seasonal periods (no seasonality)

        seasonal_inputs=np.zeros((len(df_complete_data),0))        
        return pd.DataFrame(seasonal_inputs,index=df_complete_data.index,columns=seasonal_params_names)
               
    else:
        
        flag=0
        for n_s in range(len(seasonal_periods)):

            # Keeping a higher parameter granularity for the first seasonlity period
            if flag==0:
                lambda_vector=(2.0*np.pi*np.arange(1,harmonics[n_s]+1))/(seasonal_periods[n_s])
                flag=1
            else:
                lambda_vector=np.append(lambda_vector,(2.0*np.pi*np.arange(1,harmonics[n_s]+1))/(seasonal_periods[n_s]))

            for j in range(1,harmonics[n_s]+1):
                seasonal_params_names.append('Seasonality_{}_harmonic_{}_cos'.format(seasonal_periods[n_s],j))
                seasonal_params_names.append('Seasonality_{}_harmonic_{}_sin'.format(seasonal_periods[n_s],j))          
        
        times_column=np.arange(time_start,len(df_complete_data)+time_start)[:,np.newaxis]
        lambda_vector=lambda_vector[np.newaxis,:]

        lambda_matrix=np.matmul(times_column,lambda_vector)

        lambda_matrix_cos=np.cos(lambda_matrix).ravel()
        lambda_matrix_sin=np.sin(lambda_matrix).ravel()

        seasonal_inputs=np.c_[lambda_matrix_cos,lambda_matrix_sin].reshape(lambda_matrix.shape[0],-1)

        end_time=time.time()

        return pd.DataFrame(seasonal_inputs,index=df_complete_data.index,columns=seasonal_params_names)



#['S_{}'.format(i+1) for i in range(seasonal_inputs.shape[1])]

def construct_trend_inputs(df,trend_order,training_data_start_time=None,data_frequency=None):
    # trend_order is the highest degree polynomial that are considered
    
    df_complete_data=df.copy()
    
    time_start=0
        
    # This is required when the prediction is required on dataset not used for training
    if training_data_start_time!=None:
        
        if data_frequency=='minutely':
                offset=Minute()
        elif data_frequency=='5minutely':
                offset=Minute(5)
        elif data_frequency=='hourly':
                offset=Hour()
        elif data_frequency=='daily':
                offset=Day()
        elif data_frequency=='weekly':
                offset=Day(7)
        elif data_frequency=='monthly':
                offset=MonthBegin()
                
        while training_data_start_time+time_start*offset!=df_complete_data.index[0]:
            time_start=time_start+1
    
    for t in range(time_start,len(df_complete_data)+time_start):

        curr_t_inputs=[]

        for degree in range(1,trend_order+1):            
            curr_t_inputs.append(np.power(t,degree))


        if t==time_start:
            trend_inputs=np.array([curr_t_inputs])
        else:
            trend_inputs=np.append(trend_inputs,np.array([curr_t_inputs]),axis=0)
        

    return pd.DataFrame(trend_inputs,index=df_complete_data.index,columns=['Trend_polynomial_degree_{}'.format(i+1) for i in range(trend_inputs.shape[1])])    

def get_holidays_and_release_features(df_complete_data,data_frequency,holidays_and_releases,kpi_name,training_window,\
                                      operation_mode,misc_values,training_data_start_time=None,exog_input_base_columns=None):

    # Get holiday flags
    [us_holiday_flag,chinese_holiday_flag]=get_holiday_flag(df_complete_data,data_frequency,\
                                                     holidays_and_releases,misc_values)
                                                
    # Combining all exog inputs
    exog_input=pd.DataFrame([],index=df_complete_data.index)
    
                            
    if (operation_mode!='executor' and operation_mode!='tester'):
        
        if isinstance(training_window,float):
        
            # US holidays (want atleast two instances of these in training data)
            if (us_holiday_flag.iloc[:int(training_window*len(df_complete_data))]==1).sum()>1:
                exog_input=pd.concat([exog_input,us_holiday_flag],axis=1)

            # Chinese holidays (want atleast two instances of these in training data)    
            if (chinese_holiday_flag.iloc[:int(training_window*len(df_complete_data))]==1).sum()>1:
                exog_input=pd.concat([exog_input,chinese_holiday_flag],axis=1)
                
        else:
            
            # US holidays (want atleast two instances of these in training data)
            if (us_holiday_flag.loc[training_window[0]:training_window[1]]==1).sum()>1:
                exog_input=pd.concat([exog_input,us_holiday_flag],axis=1)

            # Chinese holidays (want atleast two instances of these in training data)    
            if (chinese_holiday_flag.loc[training_window[0]:training_window[1]]==1).sum()>1:
                exog_input=pd.concat([exog_input,chinese_holiday_flag],axis=1)
    
    else:
        
        # NO CONSTRAINTS ON THE NUMBER OF INSTANCES IN EXEXUTOR MODE
        
        if exog_input_base_columns!=[''] and exog_input_base_columns!=None:
            # US holidays 
            exog_input=pd.concat([exog_input,us_holiday_flag],axis=1)

            # Chinese holidays     
            exog_input=pd.concat([exog_input,chinese_holiday_flag],axis=1)
            
	
            exog_input=exog_input.loc[:,exog_input_base_columns]    
            
    
    # This is the number of non-trend, non-seasonal regressors
    num_spec_reg=len(exog_input.columns)    
    
    
    return exog_input,num_spec_reg,us_holiday_flag,chinese_holiday_flag
    
