import pandas as pd
pd.set_option('display.max_colwidth',1000)
import matplotlib
matplotlib.use('agg')
import statsmodels.api as sm
import numpy as np
from scipy.stats import norm
import tools.time_series_metadata as time_series_metadata
from tools.plot_generator import generate_plot
from model.model_selection import select_best_model
import json
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
import time
import warnings
warnings.filterwarnings("ignore")
import datetime
import tools.data_transformer as data_transformer

def pointanomalyThreshold(confidence_band):
    return 2*(1-norm.cdf(float(confidence_band.split('-')[0]), loc=0, scale=1))

def perform_AD(operation_mode,model_type,data_source,model_json,kpi_name,perform_point_anomaly,delay,perform_trend_anomaly,training_window,test_window,database=False,alertingDate=datetime.datetime.now()):

    #start_time=time.time()
        
    ######################
    # Global parameters
    ######################
    if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor' or operation_mode=='fast':  # production  
        import tools.data_extractor as data_extractor
    else:
        import tools.data_extractor_nonProd as data_extractor

    autoregressive_order=0
    MA_order=0
    plottingWindow=1.0
    plottingYlim=0.2
    data_scale=0
    onlyData=False
    if operation_mode=='test':
        write_to_file=True
    else:
        write_to_file=False
    rule_based=False
    enforce_stationarity=False
    enforce_invertibility=False
    enable_seasonality=True
    confidence_band_collection=['3-sigma','3.5-sigma','4-sigma','4.5-sigma','5-sigma','5.5-sigma','6-sigma']


    # THESE ARE DEFAULT VALUES   
    
    display_trend_anomaly=True
    if operation_mode=='validation':
        display_all_anomalies=True
    else:
        display_all_anomalies=False # point
    display_all_trend_anomalies=False # trend
    
    holidays_and_releases=True
    # Used to defined the training set when user defines it from dashboard
    user_defined=False

    logarithmize=False
    model_suite=['classical']
    trend_models=['deterministic trend','local level','local linear trend']

    max_AR_order=3
    max_MA_order=1
    
    max_seasonal_AR_order=1    
    max_seasonal_MA_order=1

    start_time=0
                
    ##############################
    ## EXTRACT DATA
    ############################## 
            
    df_complete_data,df_label_data,likelihoodHashmap,model_name,kpi_name,app_name,model_json,user_defined,\
    misc_values,type_of_anomaly,trend_anomaly_window,confidence_band,threshold_probability,trendPrior,data_frequency=\
    data_extractor.load_data(operation_mode,model_type,data_source,model_json,kpi_name,\
                             database,alertingDate)      

    start_time=time.time()
    if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
        logging.debug('{}: START TIME {}'.format(kpi_name,start_time))
    else:
        logging.debug('START TIME {}'.format(start_time))    
        
    if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
        logging.debug('{}: SUCCESS LOADING DATA. NUMBER OF DATA POINTS PROVIDED {}'.\
                      format(kpi_name,len(df_complete_data)))
    else:
        logging.debug('SUCCESS LOADING DATA. NUMBER OF DATA POINTS PROVIDED {}'.\
                      format(len(df_complete_data)))          

    ##############################################################################
    ###### SET POINT ANOMALY DETECTION THRESHOLD BASED ON CONFIDENCE LEVEL DESIRED
    ##############################################################################
    threshold=pointanomalyThreshold(confidence_band)

    #####################################################
    ## UPDATE ANOMALY DETECTION TYPE IF PASSED BY INVOKER
    #####################################################         
                
    if type_of_anomaly!=None:  # If explicit invocation of a specific type of anomaly detection
        
        if type_of_anomaly=='POINT':
            
            perform_trend_anomaly=False
            
        elif type_of_anomaly=='TREND':
            
            perform_point_anomaly=False
            
            # IF ONLY TREND ANOMALY DETECTION IN TRAINER MODE, RETURN IMMEDIATELY
            if operation_mode=='trainer' or operation_mode=='tester':
                
                if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                    logging.debug('{}: TRAINER/TESTER INVOKED WITH TREND ANOMALY DETECTION. EXITING'.\
                                  format(kpi_name)) 
                else:
                    logging.debug('TRAINER/TESTER INVOKED WITH TREND ANOMALY DETECTION. EXITING')         

                param_dict = {
                    "evaluation_status": str('FAILURE'),
                    "status_description":str('TRAINER/TESTER INVOKED WITH TREND ANOMALY DETECTION')
                }

                json_string = json.dumps(param_dict, separators=(',', ':'))

                ## Output the model coefficients as a JSON string
                # print('output#'+json_string)

                logging.debug('{}: json_string {}'.format(kpi_name, json_string))

                if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                    logging.debug('{}: Ending Python script FAILURE {}'.\
                                  format(kpi_name,type_of_anomaly))  
                else:
                    logging.debug('Ending Python script FAILURE')

                return json_string           
                            
    #################################################################
    ## DATASET TRIMMING IN CASE TRAINING WINDOW PASSED AS TIME STAMPS
    ################################################################# 
        
    if not isinstance(training_window,float):
        df_complete_data.loc[:,0]=pd.to_datetime(df_complete_data.loc[:,0])
        df_complete_data=df_complete_data.loc[df_complete_data.loc[:,0]>=training_window[0]]

        if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
            df_label_data.loc[:,0]=pd.to_datetime(df_label_data.loc[:,0])
            df_label_data=df_label_data.loc[df_label_data.loc[:,0]>=training_window[0]]
                                
    #########################################################################################
    ### DATA INTEGRITY CHECK: CHECK MINIMUM DATASET PROVIDED
    #########################################################################################
        
    if len(df_complete_data)<24:# based on the min number of data needed to learn min set of parameters
        
        if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
            logging.debug('{}: FAILED SUFFICIENT DATA NOT PROVIDED. EXITING'.format(kpi_name)) 
        else:
            logging.debug('FAILED SUFFICIENT DATA NOT PROVIDED. EXITING')         
            
        param_dict = {
            "evaluation_status": str('FAILURE'),
            "status_description":str('SUFFICIENT DATA NOT PROVIDED')
        }

        json_string = json.dumps(param_dict, separators=(',', ':'))

        ## Output the model coefficients as a JSON string
        # print('output#'+json_string)

        logging.debug('{}: json_string {}'.format(kpi_name, json_string))

        if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
            logging.debug('{}: Ending Python script FAILURE {}'.format(kpi_name,type_of_anomaly))
            return json_string
        else:
            logging.debug('Ending Python script FAILURE')
        
            if operation_mode=='validation':
                
                # If batch evaluation failed because of insufficient data return None
                if model_type=='whistler_batch':
                    return pd.DataFrame([[[],[],[],[],[],[],[]]])
                            
            else:
                return     

    ##############################
    ### TRANSFORM DATA
    ##############################
    
    if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
        logging.debug('{}: TRANSFORMING DATA'.format(kpi_name))
    else:
        logging.debug('TRANSFORMING DATA')        

    df_complete_data,df_label_data,data_scale,training_window,test_window,data_frequency,\
                     cycle_period,seasonal_periods,harmonics,trend_anomaly_window,originalColumnNames,\
                     numberOfColumns,misc_values=data_transformer.data_transformer(\
                     df_complete_data,df_label_data,operation_mode,data_scale,training_window,\
                     test_window,kpi_name,perform_point_anomaly,misc_values,trend_anomaly_window,\
                     data_frequency)
        
    if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
        logging.debug('{}: SUCCESS TRANSFORMING DATA'.format(kpi_name)) 
    else:
        logging.debug('SUCCESS TRANSFORMING DATA')         

    #########################################################################################
    ### DATA INTEGRITY CHECK: CHECK MINIMUM DATASET PROVIDED FOR BATCH EVALUATION
    #########################################################################################
        
    if (model_type in ['whistler_batch']):
        
        if (data_frequency=='weekly' and (len(df_complete_data)<48)) or (data_frequency=='monthly' and (len(df_complete_data)<36)):
        
            # This is based on the minimum number of data needed to learn minimum set of parameters

            if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                logging.debug('{}: FAILED SUFFICIENT DATA NOT PROVIDED. EXITING'.format(kpi_name)) 
            else:
                logging.debug('FAILED SUFFICIENT DATA NOT PROVIDED. EXITING')         

            param_dict = {
                "evaluation_status": str('FAILURE'),
                "status_description":str('SUFFICIENT DATA NOT PROVIDED')
            }

            json_string = json.dumps(param_dict, separators=(',', ':'))

            ## Output the model coefficients as a JSON string
            # print('output#'+json_string)

            logging.debug('{}: json_string {}'.format(kpi_name, json_string))

            if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                logging.debug('{}: Ending Python script FAILURE {}'.format(kpi_name,type_of_anomaly))
                return json_string
            else:
                logging.debug('Ending Python script FAILURE')

                if operation_mode=='validation':

                    # If batch evaluation failed because of insufficient data return None
                    if model_type=='whistler_batch':
                        return pd.DataFrame([[[],[],[],[],[],[],[]]])                            
                else:
                    return
                        
        
    #############################################################
    ### DATA INTERGRITY CHECK: CHECK IF DEGENERATE DATASET PASSED
    #############################################################
        
    if data_scale[0]==-np.inf:
        
        if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
            logging.debug('{}: ONLY ZEROS PASSED. EXITING'.format(kpi_name)) 
        else:
            logging.debug('ONLY ZEROS PASSED. EXITING')         
            
        param_dict = {
            "evaluation_status": str('FAILURE'),
            "status_description":str('ONLY ZEROS PASSED')
        }


        json_string = json.dumps(param_dict, separators=(',', ':'))

        ## Output the model coefficients as a JSON string
        # print('output#'+json_string)

        logging.debug('{}: json_string {}'.format(kpi_name, json_string))

        if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
            logging.debug('{}: Ending Python script FAILURE {}'.format(kpi_name,type_of_anomaly))  
        else:
            logging.debug('Ending Python script FAILURE')
        
        return json_string
                
    ############################################################
    ### DATA INTEGRITY CHECK: DATA FREQUENCY AVAILABILITY
    ############################################################
                            
    if data_frequency==None:
        
        if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
            logging.debug('{}: DATA FREQUENCY PROVIDED NOT AMONG W, D, H, M, MIN. EXITING'.format(kpi_name)) 
            
            param_dict = {
                "evaluation_status": str('FAILURE'),
                "status_description":str('DATA FREQUENCY PROVIDED NOT AMONG W, D, H, M, MIN.')
            }
                        
        else:
            logging.debug('FAILED COULD NOT DETERMINE DATA FREQUENCY. EXITING')        
        
            param_dict = {
                "evaluation_status": str('FAILURE'),
                "status_description":str('COULD NOT DETERMINE DATA FREQUENCY')
            }

        json_string = json.dumps(param_dict, separators=(',', ':'))

        ## Output the model coefficients as a JSON string
        # print('output#'+json_string)

        logging.debug('{}: json_string {}'.format(kpi_name, json_string))        

        if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
            logging.debug('{}: Ending Python script FAILURE {}'.format(kpi_name,type_of_anomaly))  
        else:
            logging.debug('Ending Python script FAILURE')
        
        return json_string
            
    ##################################
    ## GENERATE EXOGENOUS VARIABLES      
    ##################################    

    if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
        logging.debug('{}: GENERATING EXOGENOUS VARIABLES'.format(kpi_name))
    else:
        logging.debug('GENERATING EXOGENOUS VARIABLES')
    
    if (operation_mode!='executor' and operation_mode!='tester'):
        
        # trainer,test,validation
        
        [exog_input,num_spec_reg,us_holiday_flag,chinese_holiday_flag]=\
          time_series_metadata.get_holidays_and_release_features(df_complete_data,data_frequency,holidays_and_releases,\
                                                                 kpi_name,training_window,operation_mode,misc_values)
          
        exog_input_base_columns=exog_input.columns.tolist()
        
    else:
        
        # tester,executor
        
        if perform_point_anomaly:
            [exog_input,num_spec_reg,us_holiday_flag,chinese_holiday_flag]=\
             time_series_metadata.get_holidays_and_release_features(df_complete_data,\
             data_frequency,holidays_and_releases,kpi_name,training_window,operation_mode,\
             misc_values,training_data_start_time,exog_input_base_columns)
        else:
            [exog_input,num_spec_reg,us_holiday_flag,chinese_holiday_flag]=\
             time_series_metadata.get_holidays_and_release_features(df_complete_data,\
             data_frequency,holidays_and_releases,kpi_name,training_window,operation_mode,\
             misc_values)            
                        

    if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
        logging.debug('{}: SUCCESS GENERATING EXOGENOUS VARIABLES. NAMES: {}, LIST: {}'.format(kpi_name,exog_input.columns,exog_input))  
    else:
        logging.debug('SUCCESS GENERATING EXOGENOUS VARIABLES. NAMES: {}, LIST: {}'.format(exog_input.columns,exog_input))  
        
        
    ######################################
    ### GENERATE A COPY OF ORIGINAL DATA
    ######################################
    
    # Initialize trend_boundaries as a dataframe
    trend_boundaries=df_complete_data.copy()
    trend_boundaries[:]=0.0

    # Keep a copy of the orginal data column
    for c in range(1,numberOfColumns):
        df_complete_data.loc[:,'KPI_original_{}'.format(c)]=df_complete_data.loc[:,'KPI_{}'.format(c)].copy()

    
    ##############################
    ### FEEDBACK ASSIMILATION
    ##############################

    if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
        logging.debug('{}: ASSIMILATING FEEDBACK'.format(kpi_name))  
    else:
        logging.debug('ASSIMILATING FEEDBACK') 
        
    # Any postive anomaly label provided by user    
    if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
        df_complete_data.loc[df_label_data['KPI_1']=='Y','KPI_1']=np.nan

    if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
        logging.debug('{}: SUCCESS ASSIMILATING FEEDBACK'.format(kpi_name))  
    else:
        logging.debug('SUCCESS ASSIMILATING FEEDBACK')
                
    #####################################################################
    ### PRELIMINARY OUTLIER DETECTION
    #####################################################################

    if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
        logging.debug('{}: AUTOMATIC OUTLIER DETECTION'.format(kpi_name))  
    else:
        logging.debug('AUTOMATIC OUTLIER DETECTION')    

    # Raw data
    [df_training,df_test,df_anom_training,count_of_anomalies_training]=data_transformer.prior_outlier_detection(\
                                         df_complete_data,df_label_data,us_holiday_flag,chinese_holiday_flag,\
                                         operation_mode,kpi_name,training_window,cycle_period,plottingWindow,model_type)
            
    if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
        logging.debug('{}: SUCCESS AUTOMATIC OUTLIER DETECTION'.format(kpi_name))  
    else:
        logging.debug('SUCCESS AUTOMATIC OUTLIER DETECTION')

    if operation_mode == 'trainer' or operation_mode == 'tester' or operation_mode == 'executor':
        logging.debug('{}: ZERO VALUE ANOMALY DETECTION'.format(kpi_name))
    else:
        logging.debug('ZERO VALUE ANOMALY DETECTION')

    fraction_of_zeros = float((df_training['KPI_1'] == 0.0).sum()) / (df_training['KPI_1']).count()
    df_training.loc[df_training['KPI_1'] == 0.0, 'KPI_1'] = np.nan

    if operation_mode == 'trainer' or operation_mode == 'tester' or operation_mode == 'executor':
        logging.debug('{}: SUCCESS ZERO VALUE ANOMALY DETECTION'.format(kpi_name))
    else:
        logging.debug('SUCCESS ZERO VALUE ANOMALY DETECTION')

    if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
        logging.debug('{}: COMBINING TRAINING AND TEST DATA'.format(kpi_name))  
    else:
        logging.debug('COMBINING TRAINING AND TEST DATA')        

    # combine (modified) training and test data
    df_new = pd.concat([df_training, df_test])

    if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
        logging.debug('{}: SUCCESS COMBINING TRAINING AND TEST DATA'.format(kpi_name))  
    else:
        logging.debug('SUCCESS COMBINING TRAINING AND TEST DATA')

    #############################
    ## RECORD PREPROCESSING EXECUTION TIME
    #############################

    elapsed_time_preprocessing=time.time()-start_time                        
    if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
        logging.debug('{} {}: PREPROCESSING EXECUTION TIME {}'.format(app_name,kpi_name,elapsed_time_preprocessing))
    else:
        logging.debug('PREPROCESSING EXECUTION TIME {}'.format(elapsed_time_preprocessing))
            
                        
    ########################################
    #### TRAINING (POINT ANOMALY DETECTION)
    ########################################
    if operation_mode!='tester' and operation_mode!='executor':
    
        if perform_point_anomaly:

            #-----------
            ### TRAINING
            #-----------
            start_time_pointAD=time.time()
            
            if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                logging.debug('{}: BEGIN MODEL SELECTION'.format(kpi_name))  
            else:
                logging.debug('BEGIN MODEL SELECTION') 

            ## Model selection
            [model_result,selected_trend_model,autoregressive_order,MA_order,freq_seasonal,\
             AR_seasonal_order,MA_seasonal_order,exog_input,logarithmize,classical]=\
             select_best_model(df_new,df_training,model_suite,exog_input,enable_seasonality,trend_models,\
                               max_AR_order,max_MA_order,max_seasonal_AR_order,max_seasonal_MA_order,\
                               cycle_period,seasonal_periods,harmonics,\
             enforce_invertibility,enforce_stationarity,training_window,test_window,numberOfColumns,operation_mode,kpi_name)


            if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                logging.debug('{}: SUCCESS BEGIN MODEL SELECTION'.format(kpi_name))  
            else:
                logging.debug('SUCCESS BEGIN MODEL SELECTION') 

            #-------------------    
            # Save trained model
            #-------------------

            if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                logging.debug('{}: SAVING TRAINED MODEL'.format(kpi_name))  
            else:
                logging.debug('SAVING TRAINED MODEL')

        if operation_mode=='production' or operation_mode=='test' or operation_mode=='validation':
            
            if perform_point_anomaly:             
                logging.debug(model_result.summary())
                        
        if perform_point_anomaly:
            
            if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                logging.debug('{}: SUCCESS SAVING TRAINED MODEL'.format(kpi_name))  
            else:
                logging.debug('SUCCESS SAVING TRAINED MODEL')
                
            #############################
            ## RECORD TRAINING EXECUTION TIME
            #############################

            elapsed_time_pointAD=time.time()-start_time_pointAD                        
            if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                logging.debug('{} {}: POINT ANOMALY TRAINING EXECUTION TIME {}'.format(app_name,kpi_name,elapsed_time_pointAD))
            else:
                logging.debug('POINT ANOMALY TRAINING EXECUTION TIME {}'.format(elapsed_time_pointAD))
                
    ####################################################            
    # EVALUATION/SERVE
    ####################################################
    evaluation_start_time=time.time()
    if operation_mode!='trainer':
        
        if isinstance(training_window,float):    
            prediction_start=int(training_window*len(df_new))
        else:
            prediction_start=len(df_new.loc[training_window[0]:training_window[1]])
        prediction_end=len(df_new)

        # count of anomalies in test data
        count_of_anomalies=df_new['KPI_1'].iloc[prediction_start:].isnull().sum()
        df_anom=df_new.iloc[prediction_start:].loc[df_new['KPI_1'].iloc[prediction_start:].isnull()]

        ##################################
        # POINT ANOMALY DETECTION
        ##################################
        
        if perform_point_anomaly:    # 
            
            ############################################################################
            # RECORD POINT ANOMALY SERVING START TIME
            ############################################################################
            
            start_time_pointAD=time.time()
                
            #####################################################################################    
            #### FORECASTING CODE BEGINS    
            #####################################################################################    
            # forecasting for future date: We use one-step ahead prediction
            
            if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                logging.debug('{}: EVALUATING FORECASTS'.format(kpi_name))  
            else:
                logging.debug('EVALUATING FORECASTS')

            if True:

                # Anomalies because value is zero
                if fraction_of_zeros<0.0027: # 99.63%
                    count_of_anomalies=count_of_anomalies+(df_new['KPI_1'].iloc[prediction_start:]==0.0).sum()
                    df_anom=df_anom.append(df_new.iloc[prediction_start:].\
                        loc[df_new['KPI_1'].iloc[prediction_start:]==0.0])
                df_new.loc[df_new['KPI_1']==0.0,'KPI_1']=np.nan


                if logarithmize:  # logarithmic prediction

                    #################################
                    ## Apply logarithm transformation
                    #################################

                    df_log_new = df_new.copy()
                    df_log_new['KPI_1'] = np.log(df_log_new['KPI_1'])

                    df_log_training = df_training.copy()
                    df_log_training['KPI_1'] = np.log(df_log_training['KPI_1'])

                    # -----------------------------------

                    if classical:

                        # Initialize the model used for forecasting

                        # Point anomaly detection
                        if len(exog_input.columns)>0:
                            model_new = sm.tsa.UnobservedComponents(df_log_training['KPI_1'],autoregressive=int(autoregressive_order),
                                        level=selected_trend_model,freq_seasonal=freq_seasonal,exog=exog_input)

                        else:
                            model_new = sm.tsa.UnobservedComponents(df_log_training['KPI_1'],autoregressive=int(autoregressive_order),\
                                        level=selected_trend_model,freq_seasonal=freq_seasonal)

                        if operation_mode=='tester' or operation_mode=='executor':
                            resPoint = model_new.smooth(model_params)
                        else:
                            resPoint = model_new.smooth(model_result.params)

                        for t in range(prediction_start,prediction_end):

                            # Point anomaly detection
                            f = resPoint.get_prediction(start=df_new.index[t], end=df_new.index[t])
                            resPoint=resPoint.extend(df_log_new.loc[df_new.index[t:t+1],'KPI_1'])

                            # Point anomaly detection
                            df_new.loc[df_new.index[t], 'forecast_test'] = np.exp(f.predicted_mean.iloc[0] + f.var_pred_mean[0] / 2.0)
                            df_new.loc[df_new.index[t], 'forecast_std'] = np.exp(np.sqrt(f.var_pred_mean[0]))
                            df_new.loc[df_new.index[t], 'forecast_test_LL'] = np.exp(f.conf_int(alpha=threshold).iloc[0, 0])
                            df_new.loc[df_new.index[t], 'forecast_test_UL'] = np.exp(f.conf_int(alpha=threshold).iloc[0, 1])

                            df_new.loc[df_new.index[t], 'p_value'] = f.t_test(value=df_log_new.loc[df_new.index[t], 'KPI_1'])[1]

                            # Anomalies for confidence band collection
                            if operation_mode=='validation':

                                for confidence in confidence_band_collection:

                                    ll = np.exp(f.conf_int(alpha=pointanomalyThreshold(confidence)).iloc[0, 0])
                                    ul = np.exp(f.conf_int(alpha=pointanomalyThreshold(confidence)).iloc[0, 1])

                                    if (df_new.loc[df_new.index[t],'KPI_1']>ul) or (df_new.loc[df_new.index[t],'KPI_1']<ll):
                                        df_new.loc[df_new.index[t:t+1],'KPI_{}'.format(confidence)]=np.nan
                                    else:
                                        df_new.loc[df_new.index[t:t+1],'KPI_{}'.format(confidence)]=df_new.loc[df_new.index[t:t+1],'KPI_1']

                            # ML based anomalies
                            if (operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor') and \
                                    (df_label_data.loc[df_new.index[t]]!='N')[0]:

                                if (df_new.loc[df_new.index[t],'KPI_1']>df_new.loc[df_new.index[t],'forecast_test_UL']) or \
                                (df_new.loc[df_new.index[t],'KPI_1']<df_new.loc[df_new.index[t],'forecast_test_LL']):
                                    count_of_anomalies=count_of_anomalies+1
                                    if count_of_anomalies==1:
                                        df_anom=df_new.loc[df_new.index[t:t+1]]
                                    else:
                                        df_anom=df_anom.append(df_new.loc[df_new.index[t:t+1]])
                                    df_new.loc[df_new.index[t:t+1],'KPI_1']=np.nan


                            else:


                                if (df_new.loc[df_new.index[t],'KPI_1']>df_new.loc[df_new.index[t],'forecast_test_UL']) or \
                                (df_new.loc[df_new.index[t],'KPI_1']<df_new.loc[df_new.index[t],'forecast_test_LL']):
                                    count_of_anomalies=count_of_anomalies+1
                                    if count_of_anomalies==1:
                                        df_anom=df_new.loc[df_new.index[t:t+1]]
                                    else:
                                        df_anom=df_anom.append(df_new.loc[df_new.index[t:t+1]])
                                    df_new.loc[df_new.index[t:t+1],'KPI_1']=np.nan

                    else:

                        _,_,predictDataset = createTFDataset(df_log_new,exog_input,df_log_training,[],\
                                                             trainStart=None,valStart=None,predictStart=prediction_start)

                        predicted_mean = np.array([])
                        predicted_stddev = np.array([])

                        count = 0
                        for inputs, label in predictDataset:

                            #print('PREDICTION')
                            #print(inputs)

                            model = deepnetwork.defineModel(inputs.shape[1], inputs.shape[2],
                                                            model_result.layers[1].units,
                                                            init_state=model_result.layers[1].states)
                            model.set_weights(model_result.get_weights())

                            m = model(inputs).numpy()
                            s = np.ones(m.shape) * np.std(label - m)

                            print(inputs.shape)
                            count += len(inputs)

                            predicted_mean = np.append(predicted_mean, m)
                            predicted_stddev = np.append(predicted_stddev, s)

                        #print('Mean',predicted_mean)
                        #print('Standard deviation',s)

                        for t in range(prediction_start, prediction_end):

                            df_new.loc[df_new.index[t], 'forecast_test'] = np.exp(predicted_mean[t - prediction_start] + np.square(predicted_stddev[t - prediction_start]) / 2.0)
                            df_new.loc[df_new.index[t], 'forecast_std'] = np.exp(predicted_stddev[t - prediction_start])
                            df_new.loc[df_new.index[t], 'forecast_test_LL'] = np.exp(norm.interval(1 - threshold, loc=predicted_mean[t - prediction_start],scale=predicted_stddev[t - prediction_start])[0])
                            df_new.loc[df_new.index[t], 'forecast_test_UL'] = np.exp(norm.interval(1 - threshold, loc=predicted_mean[t - prediction_start],scale=predicted_stddev[t - prediction_start])[1])
                            df_new.loc[df_new.index[t], 'p_value'] = 1.0  # just setting a dummy for now

                            # Anomalies for confidence band collection
                            if operation_mode=='validation':

                                for confidence in confidence_band_collection:

                                    ll = np.exp(norm.interval(1 - pointanomalyThreshold(confidence),loc=predicted_mean[t - prediction_start],scale=predicted_stddev[t - prediction_start])[0])
                                    ul = np.exp(norm.interval(1 - pointanomalyThreshold(confidence),loc=predicted_mean[t - prediction_start],scale=predicted_stddev[t - prediction_start])[1])

                                    if (df_new.loc[df_new.index[t],'KPI_1']>ul) or (df_new.loc[df_new.index[t],'KPI_1']<ll):
                                        df_new.loc[df_new.index[t:t+1],'KPI_{}'.format(confidence)]=np.nan
                                    else:
                                        df_new.loc[df_new.index[t:t+1],'KPI_{}'.format(confidence)]=df_new.loc[df_new.index[t:t+1],'KPI_1']

                            # ML based anomalies
                            if (
                                    operation_mode == 'trainer' or operation_mode == 'tester' or operation_mode == 'executor') and \
                                    (df_label_data.loc[df_new.index[t]] != 'N')[0]:

                                if (df_new.loc[df_new.index[t], 'KPI_1'] > df_new.loc[
                                    df_new.index[t], 'forecast_test_UL']) or \
                                        (df_new.loc[df_new.index[t], 'KPI_1'] < df_new.loc[
                                            df_new.index[t], 'forecast_test_LL']):
                                    count_of_anomalies = count_of_anomalies + 1
                                    if count_of_anomalies == 1:
                                        df_anom = df_new.loc[df_new.index[t:t + 1]]
                                    else:
                                        df_anom = df_anom.append(df_new.loc[df_new.index[t:t + 1]])
                                    df_new.loc[df_new.index[t:t + 1], 'KPI_1'] = np.nan

                            else:

                                if (df_new.loc[df_new.index[t], 'KPI_1'] > df_new.loc[
                                    df_new.index[t], 'forecast_test_UL']) or \
                                        (df_new.loc[df_new.index[t], 'KPI_1'] < df_new.loc[
                                            df_new.index[t], 'forecast_test_LL']):
                                    count_of_anomalies = count_of_anomalies + 1
                                    if count_of_anomalies == 1:
                                        df_anom = df_new.loc[df_new.index[t:t + 1]]
                                    else:
                                        df_anom = df_anom.append(df_new.loc[df_new.index[t:t + 1]])
                                    df_new.loc[df_new.index[t:t + 1], 'KPI_1'] = np.nan

                else:  # non-logarithmic prediction

                    if classical:

                        # Point anomaly detection
                        if len(exog_input.columns)>0:
                            model_new = sm.tsa.UnobservedComponents(df_training['KPI_1'],autoregressive=int(autoregressive_order),
                                        level=selected_trend_model,freq_seasonal=freq_seasonal,exog=exog_input)

                        else:
                            model_new = sm.tsa.UnobservedComponents(df_training['KPI_1'],autoregressive=int(autoregressive_order),\
                                        level=selected_trend_model,freq_seasonal=freq_seasonal)

                        if operation_mode=='tester' or operation_mode=='executor':
                            resPoint = model_new.smooth(model_params)
                        else:
                            resPoint = model_new.smooth(model_result.params)

                        for t in range(prediction_start,prediction_end):

                            # Point anomaly detection
                            f = resPoint.get_prediction(start=df_new.index[t], end=df_new.index[t])
                            resPoint=resPoint.extend(df_new.loc[df_new.index[t:t+1],'KPI_1'])

                            # Regular prediction
                            df_new.loc[df_new.index[t],'forecast_test']=f.predicted_mean.iloc[0]
                            df_new.loc[df_new.index[t],'forecast_std']=np.sqrt(f.var_pred_mean[0])
                            df_new.loc[df_new.index[t],'forecast_test_LL']=f.conf_int(alpha=threshold).iloc[0,0]
                            df_new.loc[df_new.index[t],'forecast_test_UL']=f.conf_int(alpha=threshold).iloc[0,1]

                            df_new.loc[df_new.index[t],'p_value']=\
                            f.t_test(value=df_new.loc[df_new.index[t],'KPI_1'])[1]

                            # Anomalies for confidence band collection
                            if operation_mode=='validation':

                                for confidence in confidence_band_collection:

                                    # point anomaly detection
                                    ll = f.conf_int(alpha=pointanomalyThreshold(confidence)).iloc[0, 0]
                                    ul = f.conf_int(alpha=pointanomalyThreshold(confidence)).iloc[0, 1]

                                    if (df_new.loc[df_new.index[t],'KPI_1']>ul) or (df_new.loc[df_new.index[t],'KPI_1']<ll):
                                        df_new.loc[df_new.index[t:t+1],'KPI_{}'.format(confidence)]=np.nan
                                    else:
                                        df_new.loc[df_new.index[t:t+1],'KPI_{}'.format(confidence)]=df_new.loc[df_new.index[t:t+1],'KPI_1']

                            # ML based anomalies
                            if (operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor') and \
                                    (df_label_data.loc[df_new.index[t]]!='N')[0]:

                                if (df_new.loc[df_new.index[t],'KPI_1']>df_new.loc[df_new.index[t],'forecast_test_UL']) or \
                                (df_new.loc[df_new.index[t],'KPI_1']<df_new.loc[df_new.index[t],'forecast_test_LL']):
                                    count_of_anomalies=count_of_anomalies+1
                                    if count_of_anomalies==1:
                                        df_anom=df_new.loc[df_new.index[t:t+1]]
                                    else:
                                        df_anom=df_anom.append(df_new.loc[df_new.index[t:t+1]])
                                    df_new.loc[df_new.index[t:t+1],'KPI_1']=np.nan


                            else:

                                if (df_new.loc[df_new.index[t],'KPI_1']>df_new.loc[df_new.index[t],'forecast_test_UL']) or \
                                (df_new.loc[df_new.index[t],'KPI_1']<df_new.loc[df_new.index[t],'forecast_test_LL']):
                                    count_of_anomalies=count_of_anomalies+1
                                    if count_of_anomalies==1:
                                        df_anom=df_new.loc[df_new.index[t:t+1]]
                                    else:
                                        df_anom=df_anom.append(df_new.loc[df_new.index[t:t+1]])
                                    df_new.loc[df_new.index[t:t+1],'KPI_1']=np.nan

                    else:

                        _,_,predictDataset = createTFDataset(df_new,exog_input,df_training,[],\
                                                             trainStart=None,valStart=None,predictStart=prediction_start)

                        predicted_mean = np.array([])
                        predicted_stddev = np.array([])

                        count = 0
                        for inputs, label in predictDataset:


                            model = deepnetwork.defineModel(inputs.shape[1], inputs.shape[2],
                                                            model_result.layers[1].units,
                                                            init_state=model_result.layers[1].states)
                            model.set_weights(model_result.get_weights())


                            m = model(inputs).numpy()
                            s = np.ones(m.shape) * np.std(label - m)

                            print(inputs.shape)
                            count += len(inputs)

                            predicted_mean = np.append(predicted_mean, m)
                            predicted_stddev = np.append(predicted_stddev, s)

                        for t in range(prediction_start, prediction_end):

                            if len(misc_values) == 0 or data_frequency != 'daily':
                                df_new.loc[df_new.index[t], 'forecast_test'] = predicted_mean[t - prediction_start]
                                df_new.loc[df_new.index[t], 'forecast_std'] = predicted_stddev[t - prediction_start]
                                df_new.loc[df_new.index[t], 'forecast_test_LL'] = norm.interval(1 - threshold, \
                                    loc=predicted_mean[t - prediction_start],scale=predicted_stddev[t - prediction_start])[0]
                                df_new.loc[df_new.index[t], 'forecast_test_UL'] =norm.interval(1 - threshold,\
                                    loc=predicted_mean[t - prediction_start],scale=predicted_stddev[t - prediction_start])[1]

                            df_new.loc[df_new.index[t], 'p_value'] = 1.0  # just setting a dummy for now

                            # Anomalies for confidence band collection
                            if operation_mode=='validation':

                                for confidence in confidence_band_collection:

                                    ll = norm.interval(1 - pointanomalyThreshold(confidence),\
                                                       loc=predicted_mean[t - prediction_start],scale=predicted_stddev[t - prediction_start])[0]
                                    ul = norm.interval(1 - pointanomalyThreshold(confidence),\
                                                       loc=predicted_mean[t - prediction_start],scale=predicted_stddev[t - prediction_start])[1]

                                    if (df_new.loc[df_new.index[t],'KPI_1']>ul) or (df_new.loc[df_new.index[t],'KPI_1']<ll):
                                        df_new.loc[df_new.index[t:t+1],'KPI_{}'.format(confidence)]=np.nan
                                    else:
                                        df_new.loc[df_new.index[t:t+1],'KPI_{}'.format(confidence)]=df_new.loc[df_new.index[t:t+1],'KPI_1']

                            # ML based anomalies
                            if (
                                    operation_mode == 'trainer' or operation_mode == 'tester' or operation_mode == 'executor') and \
                                    (df_label_data.loc[df_new.index[t]] != 'N')[0]:

                                if (df_new.loc[df_new.index[t], 'KPI_1'] > df_new.loc[
                                    df_new.index[t], 'forecast_test_UL']) or \
                                        (df_new.loc[df_new.index[t], 'KPI_1'] < df_new.loc[
                                            df_new.index[t], 'forecast_test_LL']):
                                    count_of_anomalies = count_of_anomalies + 1
                                    if count_of_anomalies == 1:
                                        df_anom = df_new.loc[df_new.index[t:t + 1]]
                                    else:
                                        df_anom = df_anom.append(df_new.loc[df_new.index[t:t + 1]])
                                    df_new.loc[df_new.index[t:t + 1], 'KPI_1'] = np.nan

                            else:

                                if (df_new.loc[df_new.index[t], 'KPI_1'] > df_new.loc[
                                    df_new.index[t], 'forecast_test_UL']) or \
                                        (df_new.loc[df_new.index[t], 'KPI_1'] < df_new.loc[
                                            df_new.index[t], 'forecast_test_LL']):
                                    count_of_anomalies = count_of_anomalies + 1
                                    if count_of_anomalies == 1:
                                        df_anom = df_new.loc[df_new.index[t:t + 1]]
                                    else:
                                        df_anom = df_anom.append(df_new.loc[df_new.index[t:t + 1]])
                                    df_new.loc[df_new.index[t:t + 1], 'KPI_1'] = np.nan


            if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                logging.debug('{}: SUCCESS EVALUATING FORECASTS'.format(kpi_name))  
            else:
                logging.debug('SUCCESS EVALUATING FORECASTS')

            #####################################################################################    
            #### FORECASTING CODE ENDS  
            #####################################################################################                 
                                
            df_new.rename(columns={'KPI_original_1':'Actual','forecast_test':'ML prediction'},inplace=True)
            # Setting ML prediction of zero values to be zero
            df_new.loc[df_new['Actual']==0.0,'ML prediction']=0.0

            ######################################
            ### TEST ERROR COMPUTATION
            ######################################
            
            if operation_mode=='tester' or operation_mode=='validation':

                if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                    logging.debug('{}: EVALUATING TEST ERROR'.format(kpi_name))  
                else:
                    logging.debug('EVALUATING TEST ERROR')
                                
                # Error computation
                if isinstance(training_window,float):
                    df_test_error=np.abs(df_new.loc[df_new.index[int(training_window*len(df_new)):],'ML prediction']-df_new.loc[df_new.index[int(training_window*len(df_new)):],'Actual'])
                else:
                    rescaling_factor = 10**data_scale[0]/10**np.floor(np.log10(np.abs(df_new.loc[df_new.index > test_window, 'Actual']).max()*10**data_scale[0]))
                    df_test_error=np.square((df_new.loc[df_new.index>test_window,'ML prediction']-df_new.loc[df_new.index>test_window,'Actual'])*rescaling_factor)

                df_forecast_error=0

                error_dict = {
                "evaluation_status": str('SUCCESS'),
                "model_name": model_name,
                "kpi_name":kpi_name,
                "app_name":app_name,
                "forecast_error": df_forecast_error,
                "test_error": df_test_error.mean()
                }
                json_string = json.dumps(error_dict,separators=(',', ':'))
                # print('output#'+json_string)

                logging.debug('{}: error_json {}'.format(kpi_name, error_dict))

                if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                    logging.debug('{}: SUCCESS EVALUATING TEST ERROR'.format(kpi_name))  
                else:
                    logging.debug('SUCCESS EVALUATING TEST ERROR')
            
            #############################
            ## RECORD POINT ANOMALY SERVING EXECUTION TIME
            #############################

            elapsed_time_pointAD=time.time()-start_time_pointAD                        
            if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                logging.debug('{} {}: POINT ANOMALY SERVING EXECUTION TIME {}'.format(app_name,kpi_name,elapsed_time_pointAD))
            else:
                logging.debug('POINT ANOMALY SERVING EXECUTION TIME {}'.format(elapsed_time_pointAD))
                        
        if operation_mode!='tester':
                        
            df_changepoint=pd.DataFrame(np.zeros((len(df_new)-1,1)),index=df_new.index[:-1],columns=['probability'])
            changepoint_desc=df_changepoint
            list_of_trend_anomalies=[]
            test_meanTrend = []
            test_stdTrend = []
            start_time_postprocessing=time.time()
            
            ###########
            # DECISION
            ###########
            
            if perform_point_anomaly:
                decision_point_anomaly=(not np.isnan(df_new.loc[df_new.index[-cycle_period-1:-1],'KPI_1']).all()) and \
                            np.isnan(df_new.loc[df_new.index[-1],'KPI_1'])
            else:
                decision_point_anomaly=False
            
            ##########################
            ### PLOTTING
            ##########################

            if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                logging.debug('{}: PLOTTING'.format(kpi_name))  
            else:
                logging.debug('PLOTTING')
                        
            plt,min_plotWindow,max_plotWindow=generate_plot(operation_mode,model_type,df_new,df_training,df_test,\
                      df_anom_training,df_anom,data_source,kpi_name,numberOfColumns,data_frequency,training_window,\
                      test_window,plottingWindow,plottingYlim,decision_point_anomaly,perform_point_anomaly,\
                      perform_trend_anomaly,count_of_anomalies_training,count_of_anomalies,display_trend_anomaly,\
                      display_all_anomalies,display_all_trend_anomalies,trend_boundaries,confidence_band,threshold,\
                      threshold_probability,onlyData,data_scale,rule_based,changepoint_desc,trend_anomaly_window,\
                                                            list_of_trend_anomalies)

            if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                logging.debug('{}: SUCCESS PLOTTING'.format(kpi_name))  
            else:
                logging.debug('SUCCESS PLOTTING')

            ################################################
            ### GENERATION OF FINAL RESULTS
            ################################################

            if True:

                if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                    logging.debug('{}: GENERATING FINAL RESULTS'.format(kpi_name))  
                else:
                    logging.debug('GENERATING FINAL RESULTS')
                
                ################
                # Model summary
                ################

                logging.debug('\n\nModel summary\n============\n')
                logging.debug('Total number of data points {}'.format(len(df_complete_data)))
                logging.debug('Total number of training data points {}'.format(len(df_training)))
                logging.debug('Total number of prediction data points {}'.format(len(df_test)))
                logging.debug('Data frequency: {}'.format(data_frequency))
                logging.debug('Plotting window size (fraction of training): {}'.format(plottingWindow))
                
                if perform_point_anomaly:
                    
                    logging.debug('Selected trend model: {}'.format(selected_trend_model))
                    logging.debug('Selected autoregressive order: {}'.format(autoregressive_order))
                    logging.debug('Selected moving average order: {}'.format(MA_order))
                    logging.debug('Possible seasonal periods for point anomaly detection: {}'.format(seasonal_periods))
                    logging.debug('Selected seasonal and harmonics for point anomaly detection: {}'.format(freq_seasonal))
                    logging.debug('Selected seasonal autoregressive order: {}'.format(AR_seasonal_order))
                    logging.debug('Selected seasonal moving average order: {}'.format(MA_seasonal_order))               
                    logging.debug('Point anomaly threshold: {}'.format(confidence_band))
                    logging.debug('Logarithmize: {}'.format(logarithmize))
                
                    logging.debug('p-value: {}'.format(round(1-df_new['p_value'].iloc[-1],3)))
                    logging.debug('{}'.format(kpi_name))

                ##############
                
                ############################################    
                # save the anomaly prediction plot to a file
                ############################################
                if operation_mode=='validation':

                            
                        if onlyData==False:
                            plt.savefig("figures/univariateML"+alertingDate.strftime('%d-%b-%Y %H:%M')+'_'+app_name+kpi_name+".png")
                        else:
                            plt.savefig("figures/univariateML"+alertingDate.strftime('%d-%b-%Y %H:%M')+'_'+app_name+kpi_name+".png")

                        ############################################
                        # return anomaly status at last date
                        ############################################
                            
                        if perform_point_anomaly:     # only point anomaly detection
                            
                            if model_type=='whistler_batch':
                            
                                # Saving prediction results                            
                                predict_label=np.isnan(df_new['KPI_1'])
                                predict_label.loc[np.isnan(df_new['Actual'])]=False
                                # Anomalies in data=0.0
                                if fraction_of_zeros>=0.0027:
                                    predict_label.loc[df_new['Actual']==0.0]=False
                                predict_label=pd.concat([predict_label,df_new['p_value']],\
                                                            axis=1)
                                predict_label=predict_label.reset_index()
                                
                                predict_label.columns=['timestamp','label','score']
                                    
                                predict_label['label']=predict_label['label'].map(lambda x:int(x))
                                actual=df_new['Actual'].to_frame()

                                thres=threshold

                                predict_label_group=[]
                                for confidence in confidence_band_collection:

                                    # Saving prediction results for each confidence interval in confidence_band_collection
                                    predict_label_conf = np.isnan(df_new['KPI_{}'.format(confidence)])
                                    predict_label_conf.loc[np.isnan(df_new['Actual'])] = False
                                    # Anomalies in data=0.0
                                    if fraction_of_zeros >= 0.0027:
                                        predict_label_conf.loc[df_new['Actual'] == 0.0] = False

                                    predict_label_conf = predict_label_conf.reset_index()

                                    predict_label_conf.columns = ['timestamp','label']

                                    predict_label_conf['label'] = predict_label_conf['label'].map(lambda x: int(x))

                                    predict_label_group.append(predict_label_conf)
                                
                                return pd.DataFrame([[predict_label_group,predict_label,actual,(np.sqrt(df_test_error.mean()),np.sqrt(df_test_error).max()),confidence_band,data_frequency,min_plotWindow[0],\
                                        max_plotWindow[0],plottingYlim]])
                            
                            else:
                                
                                return pd.DataFrame([[df_new.index[-1],\
                                       np.isnan(df_new.loc[df_new.index[-1],'KPI_1']),\
                                       np.nan,trendPrior,kpi_name]],\
                                       columns=['Date','Anomaly','Time of trend anomaly',\
                                                'Trend anomaly prior probability','KPI'])
                        
                        elif perform_trend_anomaly:   # only trend anomaly detection
                            
                            # Saving prediction results
                            
                            if model_type=='whistler_batch':
                            
                                predict_label=np.isnan(trend_boundaries['KPI_1'])
                                predict_label.iloc[-1]=False
                                predict_label.loc[np.isnan(df_new['Actual'])]=False
                                predict_label=predict_label.reset_index()        

                                predict_label.columns=['timestamp','label']                            
                                predict_label['label']=predict_label['label'].map(lambda x:int(x))
                                actual=df_new['Actual'].to_frame()                            
                                
                                return pd.DataFrame([[None,predict_label,actual,None,threshold_probability,data_frequency,\
                                                      min_plotWindow[0],max_plotWindow[0],plottingYlim]])
                            
                            else:
                                
                                if len(list_of_trend_anomalies)>0:
                                    return pd.DataFrame([[df_new.index[-1],\
                                    np.nan,list_of_trend_anomalies.index[-1],trendPrior,kpi_name]],\
                                    columns=['Date','Anomaly','Time of trend anomaly','Trend anomaly prior probability','KPI'])
                                else:
                                    return pd.DataFrame([[df_new.index[-1],\
                                    np.nan,np.nan,trendPrior,kpi_name]],\
                                    columns=['Date','Anomaly','Time of trend anomaly','Trend anomaly prior probability','KPI'])

                            
                else:
                    
                    #print time.time()-start_time

                    if database:
                        if onlyData==False:
                            plt.savefig("figures/"+alertingDate.strftime('%d-%b-%Y %H:%M')+'_'+app_name+kpi_name+".png")
                        else:
                            plt.savefig("figures/onlyData/data_"+alertingDate.strftime('%d-%b-%Y %H:%M')+'_'+app_name+kpi_name+".png")
                           
                    else:
                        if onlyData==False:
                            plt.savefig("figures/"+app_name[:-4]+".png")
                        else:
                            plt.savefig("figures/data_"+app_name[:-4]+".png")

                if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                    logging.debug('{}: SUCCESS GENERATING FINAL RESULTS'.format(kpi_name))  
                else:
                    logging.debug('SUCCESS GENERATING FINAL RESULTS')
    
            #############################
            ## RECORD POSTPROCESSING EXECUTION TIME
            #############################

            elapsed_time_postprocessing=time.time()-start_time_postprocessing                       
            if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                logging.debug('{} {}: POSTPROCESSING EXECUTION TIME {}'.format(app_name,kpi_name,elapsed_time_postprocessing))
            else:
                logging.debug('POSTPROCESSING EXECUTION TIME {}'.format(elapsed_time_postprocessing))
    

    #############################
    ## RECORD POINT ANOMALY EXECUTION TIME
    #############################    
    if perform_point_anomaly:
        
        if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
            if operation_mode=='executor':
                logging.debug('{} {}: POINT ANOMALY EXECUTION TIME {}'.\
                              format(app_name,kpi_name,elapsed_time_preprocessing+elapsed_time_pointAD+elapsed_time_postprocessing))
            else:
                logging.debug('{} {}: POINT ANOMALY EXECUTION TIME {}'.\
                              format(app_name,kpi_name,elapsed_time_preprocessing+elapsed_time_pointAD))
        else:
            logging.debug('POINT ANOMALY EXECUTION TIME {}'.\
                          format(elapsed_time_preprocessing+elapsed_time_pointAD+elapsed_time_postprocessing))
            
    #############################
    ## RECORD OVERALL EXECUTION TIME
    #############################
    
    elapsed_time=time.time()-start_time
    if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
        logging.debug('{} {}: EXECUTION TIME {}'.format(app_name,kpi_name,elapsed_time))
    else:
        logging.debug('EXECUTION TIME {}'.format(elapsed_time))
    
    #############################
    # LOG EXIT MESSAGE        
    #############################
                                                
    if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
        logging.debug('{} {}: Ending Python script SUCCESS {}'.format(app_name,kpi_name,type_of_anomaly))  
    else:
        logging.debug('Ending Python script SUCCESS')
        
    ############################################
    # RETURN OUTPUT JSON BACK TO MAIN_UNIVARIATE
    ############################################
    return json_string    
