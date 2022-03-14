import tools.plot_settings as plot_settings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset,Day,MonthBegin,Hour,Minute
            
def generate_plot(operation_mode,model_type,df_new,df_training,df_test,df_anom_training,df_anom,time_series_file_name,kpi_name,numberOfColumns,data_frequency,training_window,test_window,plottingWindow,plottingYlim,decision_point_anomaly,perform_point_anomaly,perform_trend_anomaly,count_of_anomalies_training,count_of_anomalies,display_trend_anomaly,display_all_anomalies,display_all_trend_anomalies,trend_boundaries,confidence_band,threshold,threshold_probability,onlyData,data_scale,rule_based,changepoint_desc,trend_anomaly_window,list_of_trend_anomalies):
    
    fig=plt.figure(figsize=(15,8))
    plt.rcParams.update({'font.size':14,'axes.titlesize':14})

    plt.subplots_adjust(bottom=0.2)

    ax=[]

    ######################################
    # PLOT DATA AND PREDICTIONS
    ######################################
        
    for c in range(1,numberOfColumns):

        ax.append(fig.add_subplot(numberOfColumns-1,1,c))    
        
        if isinstance(training_window,float):  # training window in terms of float fraction

            # SR model does not produce predictions
            if (not onlyData) and (model_type!='sr') and (model_type!='sr_batch'):

                s=df_new[['Actual']].\
                iloc[int(training_window*len(df_new))-int(plottingWindow*training_window*len(df_new)):]

                ax[c-1].plot_date(s.index.to_pydatetime(),s,'o-',color='blue',label='Data ('+data_frequency+')')

                # If there is a point anomaly
                if (operation_mode=='test' or operation_mode=='validation'):

                    if perform_point_anomaly:

                        s=df_new[['ML prediction']].\
                        iloc[int(training_window*len(df_new))-int(plottingWindow*training_window*len(df_new)):]

                        ax[c-1].plot_date(s.index.to_pydatetime(),s,'o-',color='green',label='Model prediction') 

                        # Confidence interval
                                                
                        ax[c-1].fill_between(df_new.index[int(training_window*len(df_new)):],\
                                        df_new.loc[df_new.index[int(training_window*len(df_new)):],'forecast_test_LL'],
                                        df_new.loc[df_new.index[int(training_window*len(df_new)):],'forecast_test_UL'],color='g',alpha=0.2,\
                                             label='{} band'.format(confidence_band)) 

                else:    

                    if decision_point_anomaly:

                        s=df_new[['ML prediction']].\
                        iloc[int(training_window*len(df_new))-int(plottingWindow*training_window*len(df_new)):]

                        ax[c-1].plot_date(s.index.to_pydatetime(),s,'o-',color='green',label='Model prediction') 

                        # Confidence interval

                        ax[c-1].fill_between(df_new.index[int(training_window*len(df_new)):],\
                                        df_new.loc[df_new.index[int(training_window*len(df_new)):],'forecast_test_LL'],
                                        df_new.loc[df_new.index[int(training_window*len(df_new)):],'forecast_test_UL'],color='g',alpha=0.2,\
                                             label='{} band'.format(confidence_band))            


            else:

                s=df_new[['Actual']].\
                iloc[int(training_window*len(df_new))-int(plottingWindow*training_window*len(df_new)):]

                ax[c-1].plot_date(s.index.to_pydatetime(),s,'o-',color='blue',label=kpi_name.replace('.',' ').replace('_',' ')+ '('+data_frequency+')')        

        else:      # training window in terms of time windows

            # SR model does not produce predictions
            if (not onlyData) and (model_type!='sr') and (model_type!='sr_batch'):

                s=df_new[['Actual']]

                ax[c-1].plot_date(s.index.to_pydatetime(),s,'o-',color='blue',label='Data ('+data_frequency+')')

                # If there is a point anomaly
                if (operation_mode=='test' or operation_mode=='validation'):

                    if perform_point_anomaly:

                        s=df_new.loc[df_new.index>test_window,'ML prediction']

                        ax[c-1].plot_date(s.index.to_pydatetime(),s,'o-',color='green',label='Model prediction') 

                        # Confidence interval
                        
                        ax[c-1].fill_between(df_new.loc[df_new.index>test_window].index,\
                                   df_new.loc[df_new.index>test_window,'forecast_test_LL'],\
                                   df_new.loc[df_new.index>test_window,'forecast_test_UL'],\
                                   color='g',alpha=0.2,label='{} band'.format(confidence_band))


                else:    

                    if decision_point_anomaly:

                        s=df_new[['ML prediction']]

                        ax[c-1].plot_date(s.index.to_pydatetime(),s,'o-',color='green',label='Model prediction') 

                        # Confidence interval

                        ax[c-1].fill_between(df_new.index[len(df_training):],\
                                   df_new.loc[df_new.index[len(df_training):],'forecast_test_LL'],\
                                   df_new.loc[df_new.index[len(df_training):],'forecast_test_UL'],\
                                   color='g',alpha=0.2,label='{} band'.format(confidence_band)) 

            else:

                s=df_new[['Actual']]

                ax[c-1].plot_date(s.index.to_pydatetime(),s,'o-',color='blue',label=kpi_name.replace('.',' ').replace('_',' ')+ '('+data_frequency+')')                    

    #######################################################
    ## DETERMINE PLOTTING METADATA
    #######################################################
                
    # Determining plotting boundaries 
    min_value_anom=[1E12 for c in range(1,numberOfColumns)]
    max_value_anom=[-1E12 for c in range(1,numberOfColumns)]
    if count_of_anomalies>0:
        for c in range(1,numberOfColumns):

            min_value_anom[c-1]=df_anom['KPI_{}'.format(c)].min()
            max_value_anom[c-1]=df_anom['KPI_{}'.format(c)].max()   

    min_value_anom_training=[1E12 for c in range(1,numberOfColumns)]
    max_value_anom_training=[-1E12 for c in range(1,numberOfColumns)] 

    if isinstance(training_window,float):

        if count_of_anomalies_training>0:
            for c in range(1,numberOfColumns):        

                min_value_anom_training[c-1]=df_anom_training.loc[df_new.index[int(training_window*len(df_new))-int(plottingWindow*training_window*len(df_new))]:,'KPI_{}'.format(c)].min()
                max_value_anom_training[c-1]=df_anom_training.loc[df_new.index[int(training_window*len(df_new))-int(plottingWindow*training_window*len(df_new))]:,'KPI_{}'.format(c)].max()    

        min_plotWindow=[1E12 for c in range(1,numberOfColumns)]
        max_plotWindow=[-1E12 for c in range(1,numberOfColumns)]

        if perform_point_anomaly and model_type not in ['sr','sr_batch']:

            for c in range(1,numberOfColumns):
                min_plotWindow[c-1]=min(df_new.loc[df_new.index[int(training_window*len(df_new))-int(plottingWindow*training_window*len(df_new)):],'forecast_test_LL'].min(),\
                                   df_new.loc[df_new.index[int(training_window*len(df_new))-int(plottingWindow*training_window*len(df_new)):],'Actual'].min(),\
                                   min_value_anom[c-1],min_value_anom_training[c-1])

                max_plotWindow[c-1]=max(df_new.loc[df_new.index[int(training_window*len(df_new))-int(plottingWindow*training_window*len(df_new)):],'forecast_test_UL'].max(),\
                                   df_new.loc[df_new.index[int(training_window*len(df_new))-int(plottingWindow*training_window*len(df_new)):],'Actual'].max(),max_value_anom[c-1],max_value_anom_training[c-1])

        else:

            for c in range(1,numberOfColumns):
                min_plotWindow[c-1]=min(df_new.loc[df_new.index[int(training_window*len(df_new))-int(plottingWindow*training_window*len(df_new)):],'Actual'].min(),min_value_anom[c-1],min_value_anom_training[c-1])

                max_plotWindow[c-1]=max(df_new.loc[df_new.index[int(training_window*len(df_new))-int(plottingWindow*training_window*len(df_new)):],'Actual'].max(),max_value_anom[c-1],max_value_anom_training[c-1])  

    else:

        if count_of_anomalies_training>0:
            for c in range(1,numberOfColumns):        

                min_value_anom_training[c-1]=df_anom_training.loc[:,'KPI_{}'.format(c)].min()
                max_value_anom_training[c-1]=df_anom_training.loc[:,'KPI_{}'.format(c)].max()    

        min_plotWindow=[1E12 for c in range(1,numberOfColumns)]
        max_plotWindow=[-1E12 for c in range(1,numberOfColumns)]

        if perform_point_anomaly and model_type not in ['sr','sr_batch']:

            for c in range(1,numberOfColumns):
                min_plotWindow[c-1]=min(df_new.loc[:,'forecast_test_LL'].min(),\
                                   df_new.loc[:,'Actual'].min(),\
                                   min_value_anom[c-1],min_value_anom_training[c-1])

                max_plotWindow[c-1]=max(df_new.loc[:,'forecast_test_UL'].max(),\
                                   df_new.loc[:,'Actual'].max(),max_value_anom[c-1],max_value_anom_training[c-1])

        else:

            for c in range(1,numberOfColumns):
                min_plotWindow[c-1]=min(df_new.loc[:,'Actual'].min(),min_value_anom[c-1],min_value_anom_training[c-1])

                max_plotWindow[c-1]=max(df_new.loc[:,'Actual'].max(),max_value_anom[c-1],max_value_anom_training[c-1])                 
    ###########################
    # PLOTTING TREND ANOMALIES
    ###########################
    
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

    if perform_trend_anomaly:    # Display trend anomalies if required

        if display_trend_anomaly:      

            for c in range(1,numberOfColumns):

                ### Plotting boundaries of different trends
                legend_display=True
                for n in reversed(range(len(trend_boundaries))):


                    if not display_all_trend_anomalies:  # if only expected to return the last and penultimate anomalies

                        # only display anomalies in the trend anomaly window
                        if np.isnan(trend_boundaries.iloc[n,c-1])\
                           and trend_boundaries.index[n]>=(df_new.index[-1]-(trend_anomaly_window)*offset)\
                           and trend_boundaries.index[n]<(df_new.index[-1]):

                            df_training_zone=df_new.loc[[trend_boundaries.index[n],trend_boundaries.index[n]],\
                                                        df_new.columns[c-1]]
                            df_training_zone[0]=min_plotWindow[c-1]-plottingYlim*(max_plotWindow[c-1]-min_plotWindow[c-1])
                            df_training_zone[1]=max_plotWindow[c-1]+plottingYlim*(max_plotWindow[c-1]-min_plotWindow[c-1])        

                            if legend_display:
                                ax[c-1].plot_date(df_training_zone.index.to_pydatetime(),df_training_zone,\
                                              '--',linewidth=3,color='magenta',label='Trend anomaly')
                                legend_display=False
                            else:
                                ax[c-1].plot_date(df_training_zone.index.to_pydatetime(),df_training_zone,\
                                              '--',linewidth=3,color='magenta',label='')

                            # Date annotation for trend anomaly

                            ax[c-1].text(df_training_zone.index[0]+offset,\
                                    df_training_zone[1]-0.5*plottingYlim*(max_plotWindow[c-1]-min_plotWindow[c-1]),\
                                    df_training_zone.index[0].strftime('%d-%b-%y'),fontsize=12)


                            # Also display penultimate trend anomaly
                            if len(list_of_trend_anomalies)>1:
                                df_training_zone=df_new.loc[[list_of_trend_anomalies.index[-2],\
                                                             list_of_trend_anomalies.index[-2]]\
                                                            ,df_new.columns[c-1]]

                                df_training_zone[0]=min_plotWindow[c-1]-plottingYlim*(max_plotWindow[c-1]-min_plotWindow[c-1])
                                df_training_zone[1]=max_plotWindow[c-1]+plottingYlim*(max_plotWindow[c-1]-min_plotWindow[c-1])


                                ax[c-1].plot_date(df_training_zone.index.to_pydatetime(),df_training_zone,\
                                                  '--',linewidth=3,color='magenta',label='')

                                # Date annotation for trend anomaly

                                ax[c-1].text(df_training_zone.index[0]+offset,\
                                        df_training_zone[1]-0.5*plottingYlim*(max_plotWindow[c-1]-min_plotWindow[c-1]),\
                                        df_training_zone.index[0].strftime('%d-%b-%y'),fontsize=12)


                            #plotting prediction if only performing trend anomaly detection

                            if decision_point_anomaly==False:

                                # Extract the non null datetimes
                                first_timestamp=changepoint_desc.index[0]                        
                                changepoint_desc=changepoint_desc[changepoint_desc.KPI_1.isnull()]

                                legend_display=True
                                for l in reversed(range(len(changepoint_desc))):

                                    if len(changepoint_desc)-l>2:
                                        break

                                    if l==0:
                                        s=pd.Series(changepoint_desc.iloc[l]['prediction'][:,0],\
                                        index=df_new.loc[:(changepoint_desc.index[l])].index)

                                        u=pd.Series(changepoint_desc.iloc[l]['prediction'][:,0]+\
                                                    3*changepoint_desc.iloc[l]['std'][:,0],\
                                        index=df_new.loc[:(changepoint_desc.index[l])].index)

                                        l=pd.Series(changepoint_desc.iloc[l]['prediction'][:,0]-\
                                                    3*changepoint_desc.iloc[l]['std'][:,0],\
                                        index=df_new.loc[:(changepoint_desc.index[l])].index)



                                    else:
                                        s=pd.Series(changepoint_desc.iloc[l]['prediction'][:,0],\
                                        index=df_new.loc[changepoint_desc.index[l-1]+offset:(changepoint_desc.index[l])].index)              

                                        u=pd.Series(changepoint_desc.iloc[l]['prediction'][:,0]+\
                                            3*changepoint_desc.iloc[l]['std'][:,0],\
                                index=df_new.loc[changepoint_desc.index[l-1]+offset:(changepoint_desc.index[l])].index) 

                                        l=pd.Series(changepoint_desc.iloc[l]['prediction'][:,0]-\
                                            3*changepoint_desc.iloc[l]['std'][:,0],\
                                index=df_new.loc[changepoint_desc.index[l-1]+offset:(changepoint_desc.index[l])].index) 


                                    if legend_display:
                                        ax[c-1].plot_date(s.index.to_pydatetime(),s,'--',color='green',
                                                          linewidth=2,label='Model-based prediction')

                                        ax[c-1].fill_between(s.index,l,u,
                                   color='g',alpha=0.2,label='{} band'.format(confidence_band))

                                        legend_display=False
                                    else:
                                        ax[c-1].plot_date(s.index.to_pydatetime(),s,'--',color='green',
                                                          linewidth=2,label='')                                                
                                        ax[c-1].fill_between(s.index,l,u,
                                   color='g',alpha=0.2,label='')


                            break

                    else:   # if expected to return all anomalies

                        if np.isnan(trend_boundaries.iloc[n,c-1]):

                            df_training_zone=df_new.loc[[trend_boundaries.index[n],trend_boundaries.index[n]],\
                                                       df_new.columns[c-1]]
                            df_training_zone[0]=min_plotWindow[c-1]-plottingYlim*(max_plotWindow[c-1]-min_plotWindow[c-1])
                            df_training_zone[1]=max_plotWindow[c-1]+plottingYlim*(max_plotWindow[c-1]-min_plotWindow[c-1])        

                            if legend_display:
                                ax[c-1].plot_date(df_training_zone.index.to_pydatetime(),df_training_zone,\
                                              '--',linewidth=3,color='magenta',label='Trend anomaly')
                                legend_display=False
                            else:
                                ax[c-1].plot_date(df_training_zone.index.to_pydatetime(),df_training_zone,\
                                              '--',linewidth=3,color='magenta',label='')                            

                # if display all trend anomalies, one time plotting of model predictions                        
                if decision_point_anomaly==False:   # If no point anomaly identified

                    if display_all_trend_anomalies: 

                        first_timestamp=changepoint_desc.index[0]                        
                        changepoint_desc=changepoint_desc[changepoint_desc.KPI_1.isnull()]

                        legend_display=True
                        for l in reversed(range(len(changepoint_desc))):

                            if l==0:

                                s=pd.Series(changepoint_desc.iloc[l]['prediction'][:,0],\
                                index=df_new.loc[:(changepoint_desc.index[l])].index)

                                u=pd.Series(changepoint_desc.iloc[l]['prediction'][:,0]+\
                                            3*changepoint_desc.iloc[l]['std'][:,0],\
                                index=df_new.loc[:(changepoint_desc.index[l])].index)

                                l=pd.Series(changepoint_desc.iloc[l]['prediction'][:,0]-\
                                            3*changepoint_desc.iloc[l]['std'][:,0],\
                                index=df_new.loc[:(changepoint_desc.index[l])].index)

                            else:

                                s=pd.Series(changepoint_desc.iloc[l]['prediction'][:,0],\
                                index=df_new.loc[changepoint_desc.index[l-1]+offset:(changepoint_desc.index[l])].index)      

                                u=pd.Series(changepoint_desc.iloc[l]['prediction'][:,0]+\
                                            3*changepoint_desc.iloc[l]['std'][:,0],\
                                index=df_new.loc[changepoint_desc.index[l-1]+offset:(changepoint_desc.index[l])].index) 

                                l=pd.Series(changepoint_desc.iloc[l]['prediction'][:,0]-\
                                            3*changepoint_desc.iloc[l]['std'][:,0],\
                                index=df_new.loc[changepoint_desc.index[l-1]+offset:(changepoint_desc.index[l])].index) 

                            if legend_display:
                                ax[c-1].plot_date(s.index.to_pydatetime(),s,'--',color='green',
                                                  linewidth=2,label='Model-based prediction')

                                ax[c-1].fill_between(s.index,l,u,
                                   color='g',alpha=0.2,label='{} band'.format(confidence_band))


                                legend_display=False
                            else:
                                ax[c-1].plot_date(s.index.to_pydatetime(),s,'--',color='green',
                                                  linewidth=2,label='')

                                ax[c-1].fill_between(s.index,l,u,color='g',alpha=0.2,label='')

    ###########################
    # PLOTTING POINT ANOMALIES
    ###########################
                                
    if perform_point_anomaly:     # Display point anomalies if required

        if not onlyData:
            if display_all_anomalies:

                if count_of_anomalies>0:
                    for c in range(1,numberOfColumns):        
                        ax[c-1].plot_date(df_anom[['KPI_{}'.format(c)]].index.to_pydatetime(),df_anom[['KPI_{}'.format(c)]],\
                                          color='r',marker='o',label='Point anomaly') 


                if count_of_anomalies_training>0:
                    for c in range(1,numberOfColumns):
                        if count_of_anomalies>0:
                            ax[c-1].plot_date(df_anom_training[['KPI_{}'.format(c)]].\
                                              index.to_pydatetime(),df_anom_training[['KPI_{}'.format(c)]],\
                                              color='r',marker='o')
                        else:
                            ax[c-1].plot_date(df_anom_training[['KPI_{}'.format(c)]].\
                                              index.to_pydatetime(),df_anom_training[['KPI_{}'.format(c)]],\
                                              color='r',marker='o',label='Point anomaly')            
            else:
                # only show last anomaly if its at the last time point
                if count_of_anomalies>0 and decision_point_anomaly:
                    for c in range(1,numberOfColumns):
                        if df_anom.index[-1]==df_new.index[-1]:
                            ax[c-1].plot_date(df_anom.loc[df_anom.index[-1:],'KPI_{}'.format(c)].index.to_pydatetime(),\
                                              df_anom.loc[df_anom.index[-1],'KPI_{}'.format(c)],color='r',\
                                     marker='o',label='Point anomaly') 

    #########################                        
    # UPDATE PLOT ANNOTATION
    #########################
    plot_settings.update_plot_annotation(operation_mode,df_new,df_training,df_test,time_series_file_name,\
                                         plt,fig,ax,plottingWindow,plottingYlim,min_plotWindow,max_plotWindow,\
                                         threshold,threshold_probability,confidence_band,
                                         data_frequency,numberOfColumns,training_window,test_window,\
                                         data_scale,rule_based,trend_boundaries,decision_point_anomaly,perform_point_anomaly)    

    return plt,min_plotWindow,max_plotWindow

