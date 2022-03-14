import matplotlib
matplotlib.use('agg')
import statsmodels.api as sm
import logging
import copy
import numpy as np
np.random.seed(1000) # for reproducibility
from tools.prediction_error import prediction_error
import warnings
warnings.filterwarnings("ignore")

def get_powerset(list_seas,list_har,n,seasonal_periods,harmonics):
    
    if n==1:
        list_seas.append([seasonal_periods[n-1]])
        list_har.append([harmonics[n-1]])
    else:
        get_powerset(list_seas,list_har,n-1,seasonal_periods,harmonics)
        
        s_copy=copy.deepcopy(list_seas)
        h_copy=copy.deepcopy(list_har)
            
        for l in range(len(list_seas)):
            list_seas[l].append(seasonal_periods[n-1])
        list_seas.append([seasonal_periods[n-1]])
        
        list_seas.extend(s_copy)     

        for l in range(len(list_har)):
            list_har[l].append(harmonics[n-1])
        list_har.append([harmonics[n-1]])
        
        list_har.extend(h_copy)

        
def select_best_model(df_new,df_training,model_suite,exog_input,enable_seasonality,trend_models,max_AR_order,max_MA_order,
                     max_seasonal_AR_order,max_seasonal_MA_order,cycle_period,seasonal_periods,harmonics, 
                     enforce_invertibility,enforce_stationarity,training_window,test_window,\
                     numberOfColumns,operation_mode,kpi_name):

        ### TRAINING
        
        ## Model selection
        model_number=0
        min_model_sel_criterion=1E15
        exog_input_base=exog_input.copy()
        current_best_model=None
        selected_width=None

        # Generate combinations of possible seasonalities
        comb_seasonalities=[]
        comb_harmonics=[]

        # NAN all zero values
        df_new=df_new.copy()
        df_new.loc[df_new['KPI_1']==0.0,'KPI_1']=np.nan

        if len(seasonal_periods)>0:
            get_powerset(comb_seasonalities,comb_harmonics,len(seasonal_periods),seasonal_periods,harmonics)
        
        comb_seasonalities.append([])
        comb_harmonics.append([])

        if 'classical' in model_suite:

            # CLASSICAL TIME SERIES MODELS WITH RAW DATA
            
            for trend_model in trend_models:# model selection runs for all polynomial of degree < max_trend

                # This function calculates the regressors needed to model the trend in time series
                for p in range(max_AR_order):
                    for q in range(max_MA_order):

                        for (seas,harm) in zip(comb_seasonalities,comb_harmonics):

                            # Construct list of periods and harmonics
                            freq_seasonal=[]
                            for s in range(len(seas)):
                                freq_seasonal.append({'period':seas[s],'harmonics':harm[s]})

                            if len(freq_seasonal)==0:
                                freq_seasonal=None

                            exog=exog_input_base

                            if isinstance(training_window,float):
                                exog_training=exog.iloc[:int(training_window*len(df_new))]
                            else:
                                exog_training=exog.loc[training_window[0]:training_window[1]]


                            for p_seasonal in range(max_seasonal_AR_order):
                                for q_seasonal in range(max_seasonal_MA_order):

                                    if len(exog_training.columns)>0:

                                        model = sm.tsa.UnobservedComponents(\
                                                df_training['KPI_1'],autoregressive=p,\
                                                level=trend_model,exog=exog_training,freq_seasonal=freq_seasonal)


                                    else:

                                        model = sm.tsa.UnobservedComponents(\
                                                df_training['KPI_1'],autoregressive=p,\
                                                level=trend_model,freq_seasonal=freq_seasonal)


                                    # Fit the data
                                    flag_model=1
                                    try:

                                        model_result = model.fit(disp=0)

                                        try:
                                            params_list=model_result.params
                                        except:
                                            flag_model=0

                                    except:
                                        flag_model=0

                                    # Evaluate model selection criterion
                                    if flag_model:

                                        model_sel_criterion=prediction_error(df_new,df_training,\
                                                training_window,test_window,AIC=False,model_result=model_result,\
                                                AR_order=p,diff_order=0,MA_order=q,\
                                                AR_seasonal_order=p_seasonal,freq_seasonal=freq_seasonal,\
                                                MA_seasonal_order=q_seasonal,cycle_period=cycle_period,\
                                                enforce_invertibility=enforce_invertibility,\
                                                enforce_stationarity=enforce_stationarity,\
                                                trend_model=trend_model,\
                                                deterministic_order=None,exog_input=exog,\
                                                numberOfColumns=numberOfColumns,logarithmize=False,classical=True)

                                        logging.debug(
                                            'Model selection: non-logarithmized, seasonal={} trend_model="{}" AR_order={} MA_order={} seas="{}" harms="{}" seasonal_AR_order={} seasonal_MA_order={} pred_error={}'.
                                            format(freq_seasonal,trend_model, p, q, seas, harm, p_seasonal, q_seasonal, model_sel_criterion)
                                        )

                                        if model_sel_criterion<min_model_sel_criterion:
                                            current_best_model=model_result
                                            selected_trend_model=trend_model
                                            AR_order=p
                                            MA_order=q
                                            diff_seasonal_order=freq_seasonal
                                            AR_seasonal_order=p_seasonal
                                            MA_seasonal_order=q_seasonal
                                            min_model_sel_criterion=model_sel_criterion
                                            exog_input=exog
                                            exog_input_training=exog_training
                                            logarithmize=False
                                            classical=True
                                            selected_width=None

                                    model_number=model_number+1

                                    if operation_mode=='trainer' or operation_mode=='tester' or operation_mode=='executor':
                                        logging.debug('{}: BEGIN MODEL SELECTION. EVALUATION OF MODEL NUMBER {}'.\
                                                      format(kpi_name,model_number))
                                    else:
                                        logging.debug('{}: BEGIN MODEL SELECTION. EVALUATION OF MODEL NUMBER {}'.format(kpi_name,model_number))
            
            # CLASSICAL TIME SERIES MODELS WITH LOGARITHMIZED DATA

            # Check if all training data points are >=0 and hence amenable to logarithm transformation

            if ((df_training['KPI_original_1']>=0.0) | (df_training['KPI_original_1'].isnull())).all():

                # Data
                df_log_new=df_new.copy()
                df_log_training=df_training.copy()

                df_log_new['KPI_1']=np.log(df_log_new['KPI_1'])
                df_log_training['KPI_1']=np.log(df_log_training['KPI_1'])

                for trend_model in trend_models:  # model selection runs for all polynomial of degree < max_trend

                    # This function calculates the regressors needed to model the trend in time series
                    for p in range(max_AR_order):
                        for q in range(max_MA_order):

                            for (seas, harm) in zip(comb_seasonalities, comb_harmonics):

                                # Construct list of periods and harmonics
                                freq_seasonal = []
                                for s in range(len(seas)):
                                    freq_seasonal.append({'period': seas[s], 'harmonics': harm[s]})

                                if len(freq_seasonal) == 0:
                                    freq_seasonal = None

                                exog = exog_input_base

                                if isinstance(training_window, float):
                                    exog_training = exog.iloc[:int(training_window * len(df_new))]
                                else:
                                    exog_training = exog.loc[training_window[0]:training_window[1]]

                                for p_seasonal in range(max_seasonal_AR_order):
                                    for q_seasonal in range(max_seasonal_MA_order):

                                        if len(exog_training.columns) > 0:

                                            model = sm.tsa.UnobservedComponents( \
                                                df_log_training['KPI_1'], autoregressive=p, \
                                                level=trend_model, exog=exog_training, freq_seasonal=freq_seasonal)


                                        else:

                                            model = sm.tsa.UnobservedComponents( \
                                                df_log_training['KPI_1'], autoregressive=p, \
                                                level=trend_model, freq_seasonal=freq_seasonal)

                                        # Fit the data
                                        flag_model = 1
                                        try:
                                            model_result = model.fit(disp=0)

                                            try:
                                                params_list = model_result.params
                                            except:
                                                flag_model = 0

                                        except:
                                            flag_model = 0

                                        # Evaluate model selection criterion
                                        if flag_model:

                                            model_sel_criterion = prediction_error(df_log_new,df_log_training, \
                                                                                   training_window, test_window, AIC=False,
                                                                                   model_result=model_result, \
                                                                                   AR_order=p, diff_order=0, MA_order=q, \
                                                                                   AR_seasonal_order=p_seasonal,
                                                                                   freq_seasonal=freq_seasonal, \
                                                                                   MA_seasonal_order=q_seasonal,
                                                                                   cycle_period=cycle_period, \
                                                                                   enforce_invertibility=enforce_invertibility, \
                                                                                   enforce_stationarity=enforce_stationarity, \
                                                                                   trend_model=trend_model, \
                                                                                   deterministic_order=None, exog_input=exog, \
                                                                                   numberOfColumns=numberOfColumns,logarithmize=True,classical=True)

                                            logging.debug(
                                                'Model selection logarithmized, seasonal {} trend_model="{}" AR_order={} MA_order={} seas="{}" harms="{}" seasonal_AR_order={} seasonal_MA_order={} pred_error={}'.
                                                    format(freq_seasonal,trend_model, p, q, seas, harm, p_seasonal, q_seasonal,model_sel_criterion)
                                            )

                                            if model_sel_criterion < min_model_sel_criterion:
                                                current_best_model = model_result
                                                selected_trend_model = trend_model
                                                AR_order = p
                                                MA_order = q
                                                diff_seasonal_order = freq_seasonal
                                                AR_seasonal_order = p_seasonal
                                                MA_seasonal_order = q_seasonal
                                                min_model_sel_criterion = model_sel_criterion
                                                exog_input = exog
                                                exog_input_training = exog_training
                                                logarithmize=True
                                                classical=True
                                                selected_width=None

                                        model_number = model_number + 1

                                        if operation_mode == 'trainer' or operation_mode == 'tester' or operation_mode == 'executor':
                                            logging.debug('{}: BEGIN MODEL SELECTION. EVALUATION OF MODEL NUMBER {}'. \
                                                          format(kpi_name, model_number))
                                        else:
                                            logging.debug(
                                                '{}: BEGIN MODEL SELECTION. EVALUATION OF MODEL NUMBER {}'.format(kpi_name,
                                                                                                                  model_number))

        logging.debug(
            '{} Selected model: classical={} logarithmize={} width {} trend_model="{}" AR_order={} MA_order={} seas="{}" seasonal_AR_order={} seasonal_MA_order={} pred_error={}'.
                format(kpi_name,classical,logarithmize,selected_width,selected_trend_model, AR_order, MA_order, diff_seasonal_order, AR_seasonal_order,
                       MA_seasonal_order,min_model_sel_criterion))

        # PARAMETER ESTIMATION
        if classical:

            if logarithmize:

                if len(exog_input_training.columns)>0:

                    model = sm.tsa.UnobservedComponents(\
                                   df_log_training['KPI_1'],autoregressive=AR_order,\
                                   level=selected_trend_model,exog=exog_input_training,freq_seasonal=diff_seasonal_order)


                else:


                    model = sm.tsa.UnobservedComponents(\
                                   df_log_training['KPI_1'],autoregressive=AR_order,\
                                   level=selected_trend_model,freq_seasonal=diff_seasonal_order)

            else:

                if len(exog_input_training.columns) > 0:

                    model = sm.tsa.UnobservedComponents( \
                        df_training['KPI_1'], autoregressive=AR_order, \
                        level=selected_trend_model, exog=exog_input_training, freq_seasonal=diff_seasonal_order)


                else:

                    model = sm.tsa.UnobservedComponents( \
                        df_training['KPI_1'], autoregressive=AR_order, \
                        level=selected_trend_model, freq_seasonal=diff_seasonal_order)

                

            # Fit the data
            model_result = model.fit(disp=0)
            current_best_model=model_result

        return [current_best_model,selected_trend_model,AR_order,MA_order,diff_seasonal_order,AR_seasonal_order,MA_seasonal_order,
                exog_input,logarithmize,classical]
