from __future__ import division
import warnings
import pandas as pd
import json
import logging
import base64

warnings.filterwarnings("ignore")

class data_extractor(object):
    def __init__(self, app_name, kpi_name, date):
        self.name = app_name
        self.kpi_name=kpi_name
        self.data = None
        self.date = date
        self.weights = []
        self.credentials = None
        self.db_properties = None
        self.filename = None
        self.ip = None
        self.port = None
        self.SID = None
        self.dsn_tns = None
        self.properties = None
        self.res = None
        self.cur = None
        self.df_list = None
        self.thresh = None
        self.df = None
        self.final_weights = None
        self.result = None
        self.final = None
        self.first_level = None
        self.second_level = None
        self.third_level = None
        self.prev_deviations = None
        self.deviations = None
        self.new_res = None
        self.domain = None
        self.dimension = None
    
def load_data(operation_mode,model_type,data_source,model_json,kpi_name,database,\
              alertingDate):

    # Read input data into a data frame
    # Convert the date column into date type and index
    if database==False:

        # production modes
        if operation_mode=='fast':

            data_arg=base64.b64decode(data_source).decode('utf-8')
            data=json.loads(data_arg.replace('#',' ').replace('\'',' '))
            
            logging.debug('data is {}'.format(data))

            # KPI values
            df_complete_data = pd.DataFrame(adhoc_json_to_list(data), columns=['Date','value'])            
            
            data_frequency=data['frequency']
            df_label_data=None
            model_name=None
            kpi_name=None
            app_name=None
            model_json=None
            user_defined=None
            misc_values=None
            type_of_anomaly=None
            trend_anomaly_window=None
            threshold_probability=None
            confidence_band=None
            trendPrior=None

            # trend likelihood values
            try:
                for kpi in data['kpiList']:
                    likelihoodHashmap=json.loads(base64.b64decode(kpi['trendADLikelihood']).decode('utf-8'))
            except:
                likelihoodHashmap={}
            logging.debug('{}: TREND LIKELIHOOD VALUES RECEIVED {}'.format(kpi_name,likelihoodHashmap))
                            
        elif operation_mode=='trainer':

            data_arg=base64.b64decode(data_source).decode('utf-8')
            data=json.loads(data_arg.replace('#',' ').replace('\'',' '))                        
            model_name = data['modelName'] 
            for kpi in data['kpiList']:
                kpi_name=kpi['kpiName']

            try:
                for kpi in data['kpiList']:
                    app_name=kpi['appName']      
            except:
                app_name=None
                logging.debug('{}: APP NAME NOT PROVIDED'.format(kpi_name))

            # Read data frequency provided    
            try:
                for kpi in data['kpiList']:
                    data_frequency=kpi['kpiFreq']      
            except:
                data_frequency=None
                logging.debug('{}: DATA FREQUENCY NOT PROVIDED'.format(kpi_name))
                
                
            #try:
            #    for kpi in data['kpiList']:
            #        trend_anomaly_window=kpi['trendWindow']      
            #except:
            #    trend_anomaly_window=None
            #    logging.debug('{}: TREND WINDOW NOT PROVIDED'.format(kpi_name))  
            trend_anomaly_window=None
                
            # Read trend threshold probability   
            try:
                for kpi in data['kpiList']:
                    threshold_probability=kpi['trendThreshold']
                assert 0<=threshold_probability<=1
            except:
                threshold_probability=0.95
            logging.debug('{}: THRESHOLD PROBABILITY NOT PROVIDED OR INCORRECT FORMAT OR OUTSIDE RANGE. DEFAULT SET'.format(kpi_name))

            # Read point threshold probability
            try:
                for kpi in data['kpiList']:
                    if kpi['pointThreshold'] == 1.0:
                        confidence_band = '1-sigma'
                    elif kpi['pointThreshold'] == 2.0:
                        confidence_band = '2-sigma'
                    elif kpi['pointThreshold'] == 2.5:
                        confidence_band = '2.5-sigma'
                    elif kpi['pointThreshold'] == 3.0:
                        confidence_band = '3-sigma'
                    elif kpi['pointThreshold'] == 3.5:
                        confidence_band = '3.5-sigma'
                    elif kpi['pointThreshold'] == 4.0:
                        confidence_band = '4-sigma'
                    elif kpi['pointThreshold'] == 4.5:
                        confidence_band = '4.5-sigma'
                assert kpi['pointThreshold'] in [1, 2, 2.5, 3, 3.5, 4, 4.5]
            except:
                confidence_band = '3-sigma'
                logging.debug('{}: THRESHOLD NOT PROVIDED. DEFAULT SET'.format(kpi_name))


            # Read trend prior probability    
            try:
                for kpi in data['kpiList']:
                    trendPrior=kpi['trendPrior']      
            except:
                trendPrior=0.5
                logging.debug('{}: TREND PRIOR NOT PROVIDED. DEFAULT SET'.format(kpi_name))

            # trend likelihood values
            try:
                for kpi in data['kpiList']:
                    likelihoodHashmap = json.loads(base64.b64decode(kpi['trendADLikelihood']).decode('utf-8'))
            except:
                likelihoodHashmap = {}
            logging.debug('{}: TREND LIKELIHOOD VALUES RECEIVED {}'.format(kpi_name, likelihoodHashmap))


            try:                
                type_of_anomaly=data['type_of_anomaly']                
                logging.debug('{}: TYPE OF ANOMALY PROVIDED: {}'.format(kpi_name,type_of_anomaly))
            except:                
                type_of_anomaly=None
                logging.debug('{}: TYPE OF ANOMALY NOT PROVIDED. DEFAULT (POINT AND TREND) SET'.format(kpi_name))
                

            logging.debug('{}: KPI name is {}'.format(kpi_name, kpi_name))
            logging.debug('{}: data is {}'.format(kpi_name, data))

            # KPI values
            df_complete_data = pd.DataFrame(json_to_list(data), columns=['Date','KPI'])            
            df_complete_data.sort_values(by='Date',inplace=True)

            # labels
            df_label_data = pd.DataFrame(labels_from_json(data),columns=['Date','Labels'])
            df_label_data.sort_values(by='Date',inplace=True)

            # model_json created only after training
            model_json=None
            
            # dataset defined by user
            user_defined = bool(data['kpiList'][0]['userOverride'])
            
            # miscellaneous values such previous Christmas, Day-after-Thanksgiving, major iPhone release,
            # major iOS release
            misc_values={}
            try:
                misc_values['prev_christmas']=data['kpiList'][0]['prev_christmas']
                misc_values['prev_DAT']=data['kpiList'][0]['prev_DAT']
                misc_values['prev_major_iOS_release']=data['kpiList'][0]['prev_major_iOS_release']
                misc_values['prev_major_iPhone_release']=data['kpiList'][0]['prev_major_iPhone_release']
                
                logging.debug('{}: SUCCESSFULLY READ PREVIOUS CHRISTMAS, THANKSGIVING, iOS RELEASE, AND iPHONE RELEASE DATA'.format(kpi_name))
                
            except:
                                
                logging.debug('{}: PREVIOUS CHRISTMAS, THANKSGIVING, iOS RELEASE, AND iPHONE RELEASE DATA NOT PROVIDED, PROCEEDING WITHOUT THEM'.format(kpi_name))
                
        elif operation_mode=='tester':

            data_arg=base64.b64decode(data_source).decode('utf-8')
            data=json.loads(data_arg.replace('#',' ').replace('\'',' '))
            model_name = data['modelName']
            for kpi in data['kpiList']:
                kpi_name=kpi['kpiName']

            try:
                for kpi in data['kpiList']:
                    app_name=kpi['appName']      
            except:
                app_name=None
                logging.debug('{}: APP NAME NOT PROVIDED'.format(kpi_name))

            # Read data frequency provided    
            try:
                for kpi in data['kpiList']:
                    data_frequency=kpi['kpiFreq']      
            except:
                data_frequency=None
                logging.debug('{}: DATA FREQUENCY NOT PROVIDED'.format(kpi_name))
                
                
            #try:
            #    for kpi in data['kpiList']:
            #        trend_anomaly_window=kpi['trendWindow']      
            #except:
            #    trend_anomaly_window=None
            #    logging.debug('{}: TREND WINDOW NOT PROVIDED'.format(kpi_name)) 
            trend_anomaly_window=None
            
            # Read trend threshold probability   
            try:
                for kpi in data['kpiList']:
                    threshold_probability=kpi['trendThreshold']
                assert 0<=threshold_probability<=1
            except:
                threshold_probability=0.95
            logging.debug('{}: THRESHOLD PROBABILITY NOT PROVIDED OR INCORRECT FORMAT OR OUTSIDE RANGE. DEFAULT SET'.format(kpi_name))

            # Read point threshold probability
            try:
                for kpi in data['kpiList']:
                    if kpi['pointThreshold'] == 1.0:
                        confidence_band = '1-sigma'
                    elif kpi['pointThreshold'] == 2.0:
                        confidence_band = '2-sigma'
                    elif kpi['pointThreshold'] == 2.5:
                        confidence_band = '2.5-sigma'
                    elif kpi['pointThreshold'] == 3.0:
                        confidence_band = '3-sigma'
                    elif kpi['pointThreshold'] == 3.5:
                        confidence_band = '3.5-sigma'
                    elif kpi['pointThreshold'] == 4.0:
                        confidence_band = '4-sigma'
                    elif kpi['pointThreshold'] == 4.5:
                        confidence_band = '4.5-sigma'
                assert kpi['pointThreshold'] in [1, 2, 2.5, 3, 3.5, 4, 4.5]
            except:
                confidence_band = '3-sigma'
                logging.debug('{}: THRESHOLD NOT PROVIDED. DEFAULT SET'.format(kpi_name))

            # Read trend prior probability    
            try:
                for kpi in data['kpiList']:
                    trendPrior=kpi['trendPrior']      
            except:
                trendPrior=0.5
                logging.debug('{}: TREND PRIOR NOT PROVIDED. DEFAULT SET'.format(kpi_name))

            # trend likelihood values
            try:
                for kpi in data['kpiList']:
                    likelihoodHashmap = json.loads(base64.b64decode(kpi['trendADLikelihood']).decode('utf-8'))
            except:
                likelihoodHashmap = {}
            logging.debug('{}: TREND LIKELIHOOD VALUES RECEIVED {}'.format(kpi_name, likelihoodHashmap))

            try:
                
                type_of_anomaly=data['type_of_anomaly']                
                logging.debug('{}: TYPE OF ANOMALY PROVIDED: {}'.format(kpi_name,type_of_anomaly))
                
            except:
                
                type_of_anomaly=None
                logging.debug('{}: TYPE OF ANOMALY NOT PROVIDED. DEFAULT (POINT AND TREND) SET'.format(kpi_name))
            
            
            logging.debug('{}: KPI name is {}'.format(kpi_name, kpi_name))
            logging.debug('{}: data is {}'.format(kpi_name, data))

            # Reading a JSON from file
            if type_of_anomaly=='TREND':
                model_json=None
            else:    
                model_arg=base64.b64decode(model_json).decode('utf-8')
                model_raw = model_arg.replace('\'',' ')  
                model_json=json.loads(model_raw)

            logging.debug('{}: model_json is {}'.format(kpi_name, model_json))

            # KPI values
            df_complete_data = pd.DataFrame(json_to_list(data), columns=['Date','KPI'])
            df_complete_data.sort_values(by='Date',inplace=True)

            # labels
            df_label_data = pd.DataFrame(labels_from_json(data),columns=['Date','Labels'])
            df_label_data.sort_values(by='Date',inplace=True)

            # dataset defined by user
            user_defined = bool(data['kpiList'][0]['userOverride'])

        elif operation_mode=='executor':

            data_arg=base64.b64decode(data_source).decode('utf-8')
            data=json.loads(data_arg.replace('#',' ').replace('\'',' '))
            
            # Read model name
            model_name = data['modelName']
            
            # Read KPI name
            for kpi in data['kpiList']:
                kpi_name=kpi['kpiName']
                
            # Read App name    
            try:
                for kpi in data['kpiList']:
                    app_name=kpi['appName']      
            except:
                app_name=None
                logging.debug('{}: APP NAME NOT PROVIDED'.format(kpi_name))

            # Read data frequency provided    
            try:
                for kpi in data['kpiList']:
                    data_frequency=kpi['kpiFreq']      
            except:
                data_frequency=None
                logging.debug('{}: DATA FREQUENCY NOT PROVIDED'.format(kpi_name))
                
                
                
            # Read trend window    
            #try:
            #    for kpi in data['kpiList']:
            #        trend_anomaly_window=kpi['trendWindow']      
            #except:
            #    trend_anomaly_window=None
            #    logging.debug('{}: TREND WINDOW NOT PROVIDED'.format(kpi_name)) 
            trend_anomaly_window=None  # default will be determined later
                
            # Read trend threshold probability   
            try:
                for kpi in data['kpiList']:
                    threshold_probability=kpi['trendThreshold']
                assert 0<=threshold_probability<=1
            except:
                threshold_probability=0.95
            logging.debug('{}: THRESHOLD PROBABILITY NOT PROVIDED OR INCORRECT FORMAT OR OUTSIDE RANGE. DEFAULT SET'.format(kpi_name))

            # Read point threshold probability
            try:
                for kpi in data['kpiList']:
                    if kpi['pointThreshold'] == 1.0:
                        confidence_band = '1-sigma'
                    elif kpi['pointThreshold'] == 2.0:
                        confidence_band = '2-sigma'
                    elif kpi['pointThreshold'] == 2.5:
                        confidence_band = '2.5-sigma'
                    elif kpi['pointThreshold'] == 3.0:
                        confidence_band = '3-sigma'
                    elif kpi['pointThreshold'] == 3.5:
                        confidence_band = '3.5-sigma'
                    elif kpi['pointThreshold'] == 4.0:
                        confidence_band = '4-sigma'
                    elif kpi['pointThreshold'] == 4.5:
                        confidence_band = '4.5-sigma'
                assert kpi['pointThreshold'] in [1, 2, 2.5, 3, 3.5, 4, 4.5]
            except:
                confidence_band = '3-sigma'
                logging.debug('{}: THRESHOLD NOT PROVIDED. DEFAULT SET'.format(kpi_name))

            # Read trend prior probability    
            try:
                for kpi in data['kpiList']:
                    trendPrior=kpi['trendPrior']      
            except:
                trendPrior=0.5
                logging.debug('{}: TREND PRIOR NOT PROVIDED. DEFAULT SET'.format(kpi_name))

            # trend likelihood values
            try:
                for kpi in data['kpiList']:
                    likelihoodHashmap = json.loads(base64.b64decode(kpi['trendADLikelihood']).decode('utf-8'))
            except:
                likelihoodHashmap = {}
            logging.debug('{}: TREND LIKELIHOOD VALUES RECEIVED {}'.format(kpi_name, likelihoodHashmap))

            # Read type of anomaly    
            try:                
                type_of_anomaly=data['type_of_anomaly']                
                logging.debug('{}: TYPE OF ANOMALY PROVIDED: {}'.format(kpi_name,type_of_anomaly))
                
            except:                
                type_of_anomaly=None
                logging.debug('{}: TYPE OF ANOMALY NOT PROVIDED. DEFAULT (POINT AND TREND) SET'.format(kpi_name))
            
            
            logging.debug('{}: KPI name is {}'.format(kpi_name, kpi_name))
            logging.debug('{}: data is {}'.format(kpi_name, data))

            time_series_file_name=kpi_name

            # KPI values
            df_complete_data = pd.DataFrame(json_to_list(data), columns=['Date','KPI'])
            df_complete_data.sort_values(by='Date',inplace=True)

            # labels
            df_label_data = pd.DataFrame(labels_from_json(data),columns=['Date','Labels'])
            df_label_data.sort_values(by='Date',inplace=True)


            # dataset defined by user
            user_defined = bool(data['kpiList'][0]['userOverride'])

            # input : time series data file
            # output : forecast value for today
            # Reading a JSON from file
            if type_of_anomaly=='TREND':
                model_json=None
            else:                    
                model_arg=base64.b64decode(model_json).deocde('utf-8')
                model_raw = model_arg.replace('\'',' ')            
                model_json=json.loads(model_raw)

            logging.debug('{}: model_json is {}'.format(kpi_name,model_json))            

        elif operation_mode=='validation':
            
            # This is the mode where validation is performed on passed data file
            
            logging.debug('KPI name is {}'.format(kpi_name))
            
            df_complete_data = data_source.copy()
            df_complete_data[0]=pd.to_datetime(df_complete_data[0])
            df_complete_data.set_index(0,inplace=True)
            df_complete_data=df_complete_data.loc[:alertingDate]
            df_complete_data.reset_index(inplace=True)
            df_complete_data.sort_values(by=[0],inplace=True)
            

            
            df_label_data=None
            model_name=None
            kpi_name=kpi_name
            app_name=model_json
            trend_anomaly_window=None
            threshold_probability=0.95
            trendPrior=0.5
            type_of_anomaly=None
            model_json=None
            user_defined=None
            misc_values={}
            data_frequency=None
            likelihoodHashmap={}
            confidence_band='4-sigma'
            
            logging.debug('{}: DATA READ FROM FILE'.format(kpi_name))
            
        
        else:

            # This is the mode where test is performed on passed data file
            
            logging.debug('KPI name is {}'.format(kpi_name))

            df_complete_data = pd.read_csv(data_source,header=None)
            df_complete_data.sort_values(by=[0],inplace=True)

            df_label_data=None
            model_name=None
            kpi_name=kpi_name
            app_name=model_json
            trend_anomaly_window=None
            threshold_probability=0.95
            trendPrior=0.5
            type_of_anomaly=None
            model_json=None
            user_defined=None
            misc_values={}
            data_frequency=None
            likelihoodHashmap={}
            confidence_band='3-sigma'

            logging.debug('{}: DATA READ FROM FILE'.format(kpi_name))

    else:
        # iReporter validation
        # test and validation modes called with connection to database    
        logging.debug('KPI name is {}'.format(kpi_name))
        
        df_complete_data = data_extractor(model_json,kpi_name,alertingDate).\
        get_data(model_type)
        
        print(df_complete_data.head())
                
        df_complete_data = df_complete_data.loc[:,df_complete_data.columns[[0,3]]]
        df_complete_data.sort_values(by=[0],inplace=True)

        df_label_data=None
        model_name=None
        kpi_name=kpi_name
        app_name=model_json
        trend_anomaly_window=None
        threshold_probability=0.95
        trendPrior=0.5
        type_of_anomaly=None
        model_json=None
        user_defined=None
        misc_values={}
        data_frequency=None
        likelihoodHashmap={}
        confidence_band='3-sigma'
        
        logging.debug('{}: DATA SUCCESSFULLY READ FROM DATABASE'.format(kpi_name))        

    return df_complete_data,df_label_data,likelihoodHashmap,model_name,kpi_name,app_name,model_json,user_defined,\
           misc_values,type_of_anomaly,trend_anomaly_window,confidence_band,threshold_probability,trendPrior,data_frequency

def adhoc_json_to_list(data):
    my_list = []
    for payload in data['timeseriesdata']:
        try:
            my_list.append((payload['timestamp'],payload['value']))
        except:
            my_list.append((payload['timestamp'],None))            
    return my_list


def json_to_list(data):
    my_list = []
    for kpiName in data['kpiList']:
        for payload in kpiName['valuePayload']:
            my_list.append((payload['date'],payload['value']))
    return my_list

def labels_from_json(data):
    my_list = []
    for kpiName in data['kpiList']:
        for payload in kpiName['labelPayload']:
            my_list.append((payload['date'],payload['value']))
    return my_list         
        
        
        