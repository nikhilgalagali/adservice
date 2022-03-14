import pandas as pd
import numpy as np
import logging
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset,Day,MonthBegin,Hour,Minute
import matplotlib.dates as dates
from multiprocessing import Pool
import multiprocessing
import time
import sys
import datetime as dt
import model.univariate_AD as univariate_AD
import json
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
#
def plot_data(file_id,df_actual,truth_file,data_frequency,training_window,test_window,min_plotWindow,max_plotWindow,plottingYlim):
    
    df_actual=missing_timestamp_imputation(df_actual,data_frequency)
    df_actual=df_actual.loc[training_window[0]:]
    
    truth_df = pd.read_csv(truth_file)    
    truth_df.set_index('timestamp',inplace=True)
    truth_df.index=pd.to_datetime(truth_df.index)
    
    truth_df=missing_timestamp_imputation(truth_df,data_frequency)
    truth_df=truth_df.loc[training_window[0]:]
        
    truth_df=pd.concat([truth_df,df_actual],axis=1)
        
    truth_df.loc[truth_df['label']==0,'Actual']=np.nan
    truth_df.drop('label',axis=1,inplace=True)
            
    fig=plt.figure(figsize=(15,8))
    ax=fig.add_subplot(1,1,1)
    plt.rcParams.update({'font.size':14,'axes.titlesize':14})
    
    plt.subplots_adjust(bottom=0.2)
    
    ax.plot_date(df_actual.index.to_pydatetime(),df_actual['Actual'],'o-',color='blue')    
    ax.plot_date(truth_df.index.to_pydatetime(),truth_df['Actual'],color='r',marker='o')

    df_test_zone=truth_df.iloc[[len(truth_df.loc[training_window[0]:test_window]),\
                              len(truth_df.loc[training_window[0]:test_window])],0] 

    df_test_zone[0]=min_plotWindow-plottingYlim*(max_plotWindow-min_plotWindow)
    df_test_zone[1]=max_plotWindow+plottingYlim*(max_plotWindow-min_plotWindow)

    ax.plot_date(df_test_zone.index.to_pydatetime(),df_test_zone,\
                                      '-',linewidth=2,color='black',label='',alpha=0.4) 
    
    ax.set_title(r'TRUE LABELS File {}'.format(file_id),fontsize=16)
    ax.set_xlabel('Time',fontsize=16)    
    
    # Set Y-lim        
    ax.set_ylim([min_plotWindow-plottingYlim*(max_plotWindow-min_plotWindow),\
                 max_plotWindow+plottingYlim*(max_plotWindow-min_plotWindow)])        
    # Set X-lim
    if data_frequency=='minutely':
        offset=Minute()
    elif data_frequency=='5minutely':
        offset=Minute(5)
    elif data_frequency=='daily':
        offset=Day()
    elif data_frequency=='hourly':
        offset=Hour()
    elif data_frequency=='weekly':
        offset=Day(7)
    else:
        offset=MonthBegin()

    ax.set_xlim([training_window[0]-offset,truth_df.index[-1]+offset]) 

    # Formatting the axes tick locations for both types of anomalies
    if data_frequency=='hourly':
        
        ax.xaxis.set_minor_locator(dates.DayLocator(interval=1))                
        ax.xaxis.set_major_locator(dates.DayLocator(interval=7))
        ax.yaxis.grid()

        ax.xaxis.set_major_formatter(dates.DateFormatter('%D'))

        ax.grid(b=1,axis='x',which='both') 
        ax.grid(axis='y')

    elif data_frequency=='minutely' or data_frequency=='5minutely':
              
        ax.xaxis.set_major_locator(dates.HourLocator(interval=24))
        ax.xaxis.set_minor_locator(dates.HourLocator(interval=6))
        ax.yaxis.grid()
                
        ax.xaxis.set_major_formatter(dates.DateFormatter('%H'))

        ax.grid(b=1,axis='x',which='both') 
        ax.grid(axis='y')    

    plt.savefig("figures/actual_data_{}".format(file_id)+".png")
    

def get_range_proba(predict, label, delay=7):
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0
    
    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                new_predict[pos: sp] = 1
            else:
                new_predict[pos: sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:  # anomaly in the end
        if 1 in predict[pos: min(pos + delay + 1, sp)]:
            new_predict[pos: sp] = 1
        else:
            new_predict[pos: sp] = 0

    return new_predict

def get_range_proba_trend(predict, label, delay=7):
    splits = np.where(label[1:] != label[:-1])[0] + 1    
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0
    
    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:pos + delay + 1]:
                new_predict[pos:sp] = 1
            else:
                new_predict[pos: sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:  # anomaly in the end
        if 1 in predict[pos: pos + delay + 1]:
            new_predict[pos: sp] = 1
        else:
            new_predict[pos: sp] = 0

    return new_predict


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


def label_evaluation(truth_file, result_file,data_frequency,test_window,delay=7,type_of_anomaly='POINT'):

    if result_file[-4:] != '.csv':
        data['message'] = "predictions not provided ina csv file"
        return json.dumps(data, ensure_ascii=False)
    else:
        result_df = pd.read_csv(result_file)
        
    result_df.set_index('timestamp',inplace=True)
    result_df.index=pd.to_datetime(result_df.index)
        
    try:
        
        truth_df = pd.read_csv(truth_file)    
        truth_df.set_index('timestamp',inplace=True)
        truth_df.index=pd.to_datetime(truth_df.index)
        truth_df=missing_timestamp_imputation(truth_df,data_frequency)
        truth_df=truth_df.loc[truth_df.index>test_window]
        # replace all NANs with 0
        truth_df.fillna(0,inplace=True)
        y_true=truth_df['label'].values
        
    except:
        y_true=np.zeros(result_df['label'].as_matrix().shape)
    
    result_df=missing_timestamp_imputation(result_df,data_frequency)
    result_df=result_df.loc[result_df.index>test_window]
    y_pred=result_df['label'].values
                    
    if delay>0:    
        if type_of_anomaly=='POINT':
            y_pred = get_range_proba(y_pred, y_true, delay)
        elif type_of_anomaly=='TREND':
            y_pred = get_range_proba_trend(y_pred, y_true, delay)
    
    return [y_pred,y_true]
    
def adservice_dataset_evaluation(args):

    file_id=args[0]
    operation_mode=args[1]
    model_type=args[2]
    dataset=args[3]
    perform_point_anomaly=args[4]
    perform_trend_anomaly=args[5]
    delay=args[6]

    y_pred=0
    y_true=0

    time_series_data=pd.read_csv('datasets/AIOPS_dataset/KPI_{}.csv'.format(file_id))

    if dataset=='AIOPS':

        time_series_data.columns=[0,3]
        ts=time_series_data.copy().set_index(0)
        ts.index=pd.to_datetime(ts.index)

        numData=len(time_series_data)

        if int(numData/2)>20160:
            training_window=[pd.to_datetime(time_series_data[0].iloc[-20160-int(numData/2)]),\
                             pd.to_datetime(time_series_data[0].iloc[-320-int(numData/2)])]
        else:    
            training_window=[pd.to_datetime(time_series_data[0].iloc[0]),\
                             pd.to_datetime(time_series_data[0].iloc[-320-int(numData/2)])]
            
        test_window=pd.to_datetime(time_series_data[0].iloc[-int(numData/2)])

        length_of_training_window=len(ts.loc[training_window[0]:training_window[1]])
        length_of_forecast_error_window=len(ts.loc[training_window[0]:test_window])-length_of_training_window   
        length_of_evaluation_window=len(ts.loc[training_window[0]:])-length_of_forecast_error_window-\
        length_of_training_window    

        print('File id',file_id)
        print('Total number of data points',numData)
        print('Number of training data points: ',length_of_training_window)
        print('Number of data points for forecast error estimation: ',length_of_forecast_error_window)
        print('Number of data points on which detection accuracy estimated',length_of_evaluation_window)

        # Pass the entire time series data
        n=numData-1

        app_name=str(file_id)
        kpi_name=str(file_id)

        alertingDate=dt.datetime.strptime(time_series_data.loc[time_series_data.index[n],0],'%Y-%m-%d %H:%M:%S')

        logging.debug('Anomaly detection on {} {}'.format(app_name,kpi_name))
        logging.debug('Running ML model')                   

        returnedDataFrame=\
        univariate_AD.perform_AD(operation_mode,model_type,time_series_data,app_name,kpi_name,perform_point_anomaly,delay,\
                 perform_trend_anomaly,training_window,test_window,database=False,alertingDate=alertingDate)

        predict_label_group = returnedDataFrame.iloc[0, 0]
        predict_label=returnedDataFrame.iloc[0,1]
        predict_label.to_csv('datasets/AIOPS_dataset/predict_label_{}.csv'.format(file_id), index=False)
        actual=returnedDataFrame.iloc[0,2]
        mean_square_error=returnedDataFrame.iloc[0,3]
        threshold=returnedDataFrame.iloc[0,4]
        data_frequency=returnedDataFrame.iloc[0,5]
        min_plotWindow=returnedDataFrame.iloc[0,6]
        max_plotWindow=returnedDataFrame.iloc[0,7]
        plottingYlim=returnedDataFrame.iloc[0,8]

    confidence_band_collection=['3-sigma','3.5-sigma','4-sigma','4.5-sigma','5-sigma','5.5-sigma','6-sigma']
    for ind,confidence in enumerate(confidence_band_collection):

        if dataset in ['AIOPS']:

            # Save predicted labels
            predict_label=predict_label_group[ind]
            if dataset=='AIOPS':

                predict_label.to_csv('datasets/AIOPS_dataset/predict_label_{}_{}.csv'.format(file_id,confidence),index=False)

                # Plotting true data
                plot_data(file_id,actual,'datasets/AIOPS_dataset/labels_{}.csv'.format(file_id),data_frequency,\
                          training_window,test_window,min_plotWindow,max_plotWindow,plottingYlim)

                # Perform label adjustment according to delayed detection criteria
                [y_pred,y_true]=label_evaluation('datasets/AIOPS_dataset/labels_{}.csv'.format(file_id),\
                                       'datasets/AIOPS_dataset/predict_label_{}_{}.csv'.format(file_id,confidence),\
                                       data_frequency,test_window,delay=delay,type_of_anomaly='POINT')

                ## AIOPS dataset has minutely, 5-minutely frequency, so the delay above is taken to be 7 (as in the Micfosoft paper)

            y_true_list = [y_true]
            y_pred_list = [y_pred]

            # Save results to file

            data={}
            try:

                fscore = f1_score(np.concatenate(y_true_list), np.concatenate(y_pred_list))
                precisionscore = precision_score(np.concatenate(y_true_list), np.concatenate(y_pred_list))
                recallscore = recall_score(np.concatenate(y_true_list), np.concatenate(y_pred_list))
                data['message'] = ['delay allowed: {}'.format(delay)]
                data['fscore'] = [fscore]
                data['precisionscore']=[precisionscore]
                data['recallscore']=[recallscore]
                data['file']=[file_id]
                data['true_anomalies']=[sum(y_true)]
                data['true_predictions']=[sum(y_pred)]
                data['threshold']=[confidence]
                if perform_point_anomaly:
                    data['type']=['POINT']
                elif perform_trend_anomaly:
                    data['type']=['TREND']

            except:

                data['file']=[file_id]
                data['message'] = ["failed evaluation of metrics"]

        if dataset=='AIOPS':
            pd.DataFrame(data).to_csv('output_data/AIOPS/{}_{}.csv'.format(file_id,confidence))

    y_pred=0
    y_true=0
    return [y_pred,y_true]
    
    
def adservice():
       
    format_string='\ntimestamp="%(asctime)s"\nlevel="INFO"\nmessage="%(message)s'
    logging.basicConfig(level=logging.DEBUG,filemode='a',format=format_string)            

    start_time=time.time()
    perform_point_anomaly=True
    perform_trend_anomaly=True

    # GLOBAL VARIABLE CONFIGURATION
    try:
        
        dataset=sys.argv[1]
        if dataset in ['AIOPS']:
            
            # Model type to use
            try:
                model_type=sys.argv[2]
                if not model_type in ['whistler_batch']:
                    model_type='whistler_batch'
            except:
                model_type='whistler_batch'
                logging.debug('Model type not specified')

            # Read the type of anomaly detection to be performed
            try:
                mode=sys.argv[3]
                if mode=='POINT':
                    perform_trend_anomaly=False
                elif mode=='TREND':
                    perform_point_anomaly=False
            except:
                perform_trend_anomaly=False
                logging.debug('Anomaly det type not specified. Continuing with default POINT and TREND')

            # Read the delay allowed in detection    
            try:    
                delay=int(sys.argv[4])
            except:
                delay=0
                                
            # Read the datasets to evaluate        
            try: 
                dataset_range_l=int(sys.argv[5])
                dataset_range_u=int(sys.argv[6])
            except:
                dataset_range_l=1
                dataset_range_u=2
                logging.debug('Range of datasets for evaluation not provided. Default 1')
                                        
        else:
            dataset=None
            logging.debug('Data set not specified.')
                                    
    except:

        logging.debug('Data set not specified. Continuing in non-evaluation mode')
        exit()

    logging.debug('START TIME {}'.format(start_time))    
    logging.debug('Number of CPUs {}'.format(multiprocessing.cpu_count()))
            
    if dataset in ['AIOPS']:

        which_model=[univariate_AD]
        operation_mode='validation'

        pool=Pool(5)
       
        list_of_inputs=[]

        if dataset in ['AIOPS']:
            
            for file_id in range(dataset_range_l,dataset_range_u):
                list_of_inputs.append((file_id,operation_mode,model_type,dataset,\
                                       perform_point_anomaly,perform_trend_anomaly,delay))

        for i in range(len(list_of_inputs)):
            adservice_dataset_evaluation(list_of_inputs[i])

        #pool.map(adservice_dataset_evaluation,list_of_inputs)

    elapsed_time=time.time()-start_time
    logging.debug('EXECUTION TIME {}'.format(elapsed_time))

if __name__ == "__main__":
    adservice()
