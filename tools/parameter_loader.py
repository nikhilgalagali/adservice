from __future__ import division
import warnings
import json
import pandas as pd
import logging
warnings.filterwarnings("ignore")
    
def load_parameters(operation_mode,model_json,kpi_name,type_of_anomaly):
        
    # Default initialization
    loader_status=True
    json_string=None
    freq_seasonal=None
    selected_trend_model=None
    autoregressive_order=None
    cycle_period=None
    model_params=None
    model_summary=None
    training_data_start_time=None
    data_scale=None
    exog_input_base_columns=None
                
    if operation_mode=='tester':

        logging.debug('{}: LOADING PARAMETERS FOR TESTER EXECUTION'.format(kpi_name))

        try:

            prev_evaluation_status=model_json['evaluation_status']

            if prev_evaluation_status=='FAILURE':
                logging.debug('{}: MODEL JSON NOT SUPPLIED'.format(kpi_name))

                param_dict = {
                    "evaluation_status": str('FAILURE'),
                    "status_description":str('MODEL JSON SUPPLIED FROM FAILED TRAINING'),
                    "model_json":model_json
                    
                }

                json_string = json.dumps(param_dict, separators=(',', ':'))

                ## Output the model coefficients as a JSON string
                print('output#'+json_string)

                logging.debug('{}: json_string {}'.format(kpi_name, json_string))

                if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                    logging.debug('{}: Ending Python script FAILURE {}'.format(kpi_name,type_of_anomaly))  
                else:
                    logging.debug('Ending Python script FAILURE')
                    
                loader_status=False    

        except:

                logging.debug('{}: PREVIOUS EVAL STATUS NOT PROVIDED'.format(kpi_name))

                param_dict = {
                    "evaluation_status": str('FAILURE'),
                    "status_description":str('PREVIOUS EVAL STATUS NOT PROVIDED'),
                    "model_json":model_json
                }

                json_string = json.dumps(param_dict, separators=(',', ':'))

                ## Output the model coefficients as a JSON string
                print('output#'+json_string)

                logging.debug('{}: json_string {}'.format(kpi_name, json_string))

                if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                    logging.debug('{}: Ending Python script FAILURE {}'.format(kpi_name,type_of_anomaly))  
                else:
                    logging.debug('Ending Python script FAILURE')

                loader_status=False    

                    
        if loader_status:
            
            cycle_period=model_json['cycle_period']
            model_params=model_json['model_params']
            selected_trend_model=model_json['selected_trend_model']
            freq_seasonal=model_json['freq_seasonal']
            if freq_seasonal!=None:
                for i in range(len(freq_seasonal)):
                    freq_seasonal[i] = { str(k): v for k, v in freq_seasonal[i].items() }
                        
            model_summary=model_json['model_summary'].replace('+',' ')
            autoregressive_order=model_json['autoregressive_order']

            training_data_start_time=pd.to_datetime(model_json['training_data_start_time'].replace('+',' '))
            data_scale=model_json['data_scale']
            exog_input_base_columns=model_json['exog_input_base_columns'].split('+')

            logging.debug('{}: model_params {}'.format(kpi_name,model_params))    


    elif operation_mode=='executor':

        logging.debug('{}: LOADING PARAMETERS FOR EXECUTOR EXECUTION'.format(kpi_name))

        try:
            prev_evaluation_status=model_json['evaluation_status']

            if prev_evaluation_status=='FAILURE':
                logging.debug('{}: MODEL JSON NOT SUPPLIED'.format(kpi_name))

                param_dict = {
                    "evaluation_status": str('FAILURE'),
                    "status_description":str('MODEL JSON SUPPLIED FROM FAILED TRAINING'),
                    "model_json":model_json
                }

                json_string = json.dumps(param_dict, separators=(',', ':'))

                ## Output the model coefficients as a JSON string
                print('output#'+json_string)

                logging.debug('{}: json_string {}'.format(kpi_name, json_string))

                if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                    logging.debug('{}: Ending Python script FAILURE {}'.format(kpi_name,type_of_anomaly))  
                else:
                    logging.debug('Ending Python script FAILURE')
                    
                loader_status=False    

        except:

                logging.debug('{}: PREVIOUS EVAL STATUS NOT PROVIDED'.format(kpi_name))

                param_dict = {
                    "evaluation_status": str('FAILURE'),
                    "status_description":str('PREVIOUS EVAL STATUS NOT PROVIDED'),
                    "model_json":model_json
                }

                json_string = json.dumps(param_dict, separators=(',', ':'))

                ## Output the model coefficients as a JSON string
                print('output#'+json_string)

                logging.debug('{}: json_string {}'.format(kpi_name, json_string))

                if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                    logging.debug('{}: Ending Python script FAILURE {}'.format(kpi_name,type_of_anomaly))  
                else:
                    logging.debug('Ending Python script FAILURE')
                    
                loader_status=False    

        
        if loader_status:
            
            cycle_period=model_json['cycle_period']
            model_params=model_json['model_params']
            selected_trend_model=str(model_json['selected_trend_model'])
            freq_seasonal=model_json['freq_seasonal']
            # Convert to non-unicode
            if freq_seasonal!=None:
                for i in range(len(freq_seasonal)):
                    freq_seasonal[i] = { str(k): v for k, v in freq_seasonal[i].items() }


            model_summary=model_json['model_summary'].replace('+',' ')
            autoregressive_order=model_json['autoregressive_order']
            training_data_start_time=pd.to_datetime(model_json['training_data_start_time'].replace('+',' '))
            data_scale=model_json['data_scale']
            exog_input_base_columns=model_json['exog_input_base_columns'].split('+')


            logging.debug('{}: model_params {}'.format(kpi_name,model_params))    
        
        
    return loader_status,json_string,freq_seasonal,selected_trend_model,autoregressive_order,cycle_period,\
           model_params,model_summary,training_data_start_time,data_scale,exog_input_base_columns
