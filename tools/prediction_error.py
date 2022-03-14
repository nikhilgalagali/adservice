import sys
import pandas as pd
import matplotlib
matplotlib.use('agg')
import statsmodels.api as sm
import numpy as np

def prediction_error(df_new,df_training,training_window,test_window,AIC,model_result,\
                     AR_order,diff_order,MA_order,\
                     AR_seasonal_order,freq_seasonal,MA_seasonal_order,\
                     cycle_period,enforce_invertibility,enforce_stationarity,\
                     trend_model,deterministic_order,exog_input,numberOfColumns,logarithmize,classical):

    avg_pred_error=0.0
    
    # forecasting for future date: We use one-step ahead prediction
    
    if isinstance(training_window,float):
        
        # the test set is all points after the training window excluding the last data point
        prediction_start=int(training_window*len(df_new))
        prediction_end=len(df_new)-1
        
    else:
        
        prediction_start=len(df_new.loc[training_window[0]:training_window[1]])
        
        if test_window==None:    
            prediction_end=len(df_new)-1
        else:
            prediction_end=len(df_new.loc[training_window[0]:test_window])

    if numberOfColumns<=2:

        if classical:  # Univariate classical model


            if not AIC:

                if len(exog_input.columns)>0:

                    model_new = sm.tsa.UnobservedComponents(\
                                df_new['KPI_1'],exog=exog_input,autoregressive=AR_order,\
                                                level=trend_model,freq_seasonal=freq_seasonal)
                else:

                    model_new = sm.tsa.UnobservedComponents(df_new['KPI_1'],autoregressive=AR_order,\
                                                level=trend_model,freq_seasonal=freq_seasonal)

                resB = model_new.filter(model_result.params)

                for t in range(prediction_start,prediction_end):

                    f=resB.get_prediction(start=df_new.index[t],end=df_new.index[t])

                    # prediction error calculated over all non-NAN and non-zero data points
                    if (not np.isnan(df_new.loc[df_new.index[t],'KPI_original_1'])) and \
                    (not (df_new.loc[df_new.index[t],'KPI_original_1']==0.0)):

                        if logarithmize:

                            avg_pred_error = avg_pred_error + \
                                             np.square(df_new.loc[df_new.index[t], 'KPI_original_1'] - \
                                                       np.exp(f.predicted_mean.iloc[0] + f.var_pred_mean[0] / 2.0))

                        else:

                            avg_pred_error=avg_pred_error+np.square(df_new.loc[df_new.index[t],'KPI_original_1']-f.predicted_mean.values)

                return avg_pred_error


            
            else:
            
                return model_result.aic