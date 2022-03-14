import matplotlib
matplotlib.use('agg')
from pandas.tseries.offsets import DateOffset,Day,MonthBegin,Hour,Minute
import matplotlib.dates as dates
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
            
def update_plot_annotation(operation_mode,df_new,df_training,df_test,time_series_file_name,plt,fig,ax,\
                           plottingWindow,plottingYlim,min_plotWindow,max_plotWindow,threshold,\
                           threshold_probability,confidence_band,data_frequency,numberOfColumns,training_window,test_window,\
                           data_scale,rule_based,trend_boundaries,decision_point_anomaly,perform_point_anomaly):
                           
    for c in range(1,numberOfColumns):

                       
        ######
        ### Plot settings
        ######

        # Y labels
        ax[c-1].set_ylabel(r'Count ($\times 10^{{{}}}$)'.format(int(data_scale[c-1])),fontsize=16)

        ### Plotting boundaries of the training and test sets

        if isinstance(training_window,float):

            df_training_zone=df_new.iloc[[int(training_window*len(df_new))-1,\
                                           int(training_window*len(df_new))-1],0]

        else:

            df_training_zone=df_new.iloc[[len(df_new.loc[training_window[0]:training_window[1]])-1,\
                                      len(df_new.loc[training_window[0]:training_window[1]])-1],0]
            
            df_test_zone=df_new.iloc[[len(df_new.loc[training_window[0]:test_window])-1,\
                                      len(df_new.loc[training_window[0]:test_window])-1],0] 
            
            df_test_zone[0]=min_plotWindow[c-1]-plottingYlim*(max_plotWindow[c-1]-min_plotWindow[c-1])
            df_test_zone[1]=max_plotWindow[c-1]+plottingYlim*(max_plotWindow[c-1]-min_plotWindow[c-1])
            
        df_training_zone[0]=min_plotWindow[c-1]-plottingYlim*(max_plotWindow[c-1]-min_plotWindow[c-1])
        df_training_zone[1]=max_plotWindow[c-1]+plottingYlim*(max_plotWindow[c-1]-min_plotWindow[c-1]) 
            
                        
        # Mute the training window boundary line
        if operation_mode=='validation':
            
            if isinstance(training_window,float):

                if perform_point_anomaly:

                    if training_window<1.0:    
                        ax[c-1].plot_date(df_training_zone.index.to_pydatetime(),df_training_zone,\
                                          '-',linewidth=2,color='black',label='',alpha=0.4)

            else:

                ax[c-1].plot_date(df_training_zone.index.to_pydatetime(),df_training_zone,\
                                      '-',linewidth=2,color='black',label='',alpha=0.4)

                ax[c-1].plot_date(df_test_zone.index.to_pydatetime(),df_test_zone,\
                                      '-',linewidth=2,color='black',label='',alpha=0.4)         
        
        else:
            
            if decision_point_anomaly:

                if isinstance(training_window,float):

                    if perform_point_anomaly:

                        if training_window<1.0:    
                            ax[c-1].plot_date(df_training_zone.index.to_pydatetime(),df_training_zone,\
                                              '-',linewidth=2,color='black',label='',alpha=0.4)

                else:

                    ax[c-1].plot_date(df_training_zone.index.to_pydatetime(),df_training_zone,\
                                          '-',linewidth=2,color='black',label='',alpha=0.4)

                    ax[c-1].plot_date(df_test_zone.index.to_pydatetime(),df_test_zone,\
                                          '-',linewidth=2,color='black',label='',alpha=0.4)                    

        ax[c-1].set_xlabel('Time',fontsize=16)    
  

        if not rule_based:

            if isinstance(training_window,float):
                
                if plottingWindow<1.0:
                    ax[c-1].annotate('',\
                                xytext=(df_training.index[-int(0.9*plottingWindow*training_window*len(df_new))],\
                                       min_plotWindow[c-1]-plottingYlim*(max_plotWindow[c-1]-min_plotWindow[c-1])/2),\
                            xy=(df_training.index[-int(plottingWindow*training_window*len(df_new))],\
                               min_plotWindow[c-1]-plottingYlim*(max_plotWindow[c-1]-min_plotWindow[c-1])/2),\
                                arrowprops=dict(arrowstyle="->"))
                    
            else:
                
                    ax[c-1].annotate('Training',\
                                xy=(df_training.index[-int(0.9*plottingWindow*len(df_training))],\
                                       min_plotWindow[c-1]-plottingYlim*(max_plotWindow[c-1]-min_plotWindow[c-1])/2))                
                    #ax[c-1].annotate('Test',\
                    #            xy=(df_test.index[-int(0.9*plottingWindow*len(df_test))],\
                    #                   min_plotWindow[c-1]-plottingYlim*(max_plotWindow[c-1]-min_plotWindow[c-1])/2))

                        
                        
        # Set Y-lim        
        ax[c-1].set_ylim([min_plotWindow[c-1]-plottingYlim*(max_plotWindow[c-1]-min_plotWindow[c-1]),\
                     max_plotWindow[c-1]+plottingYlim*(max_plotWindow[c-1]-min_plotWindow[c-1])])        
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

        if isinstance(training_window,float):    
            ax[c-1].set_xlim([df_new.index[int(training_window*len(df_new))-int(plottingWindow*training_window*len(df_new))]-offset,\
                             df_new.index[-1]+offset])
        else:
            ax[c-1].set_xlim([training_window[0]-offset,\
                             df_new.index[-1]+offset])            


        # Combined legend
        ax[c-1].legend(loc='upper left',ncol=3,fontsize=12)             
            
                 
    # This part applies to all anomaly types
    for c in range(1,numberOfColumns):
                
                          
        # Formatting the axes tick locations for both types of anomalies
        if data_frequency=='daily':      

            if len(df_new)<200:
                ax[c-1].xaxis.set_minor_locator(dates.DayLocator())
                ax[c-1].xaxis.set_major_locator(dates.WeekdayLocator(byweekday=SU,
                                            interval=2))
            elif len(df_new)<300:
                ax[c-1].xaxis.set_minor_locator(dates.DayLocator())               
                ax[c-1].xaxis.set_major_locator(dates.MonthLocator())            
            else:
                ax[c-1].xaxis.set_minor_locator(dates.DayLocator())
                ax[c-1].xaxis.set_major_locator(dates.MonthLocator(interval=2))            
                

            if c==(numberOfColumns-1):
                if len(df_new)<200:
                    ax[c-1].xaxis.set_minor_formatter(dates.DateFormatter('%d'))
                    ax[c-1].xaxis.set_major_formatter(dates.DateFormatter('%d\n%b-%y'))
                else:
                    ax[c-1].xaxis.set_major_formatter(dates.DateFormatter('%b\n%y'))

                plt.setp(ax[c-1].get_xticklabels(which='major'), rotation=0)
                plt.setp(ax[c-1].get_xticklabels(which='minor'), visible=False)

            else:
                plt.setp(ax[c-1].get_xticklabels(which='both'), visible=False)
                ax[c-1].xaxis.label.set_visible(False)


            ax[c-1].grid(b=1,axis='x',which='major') 
            ax[c-1].grid(axis='y')


        elif data_frequency=='weekly':    

            if len(df_new)<40:
                
                if df_new.index[0].weekday()==0:
                    ax[c-1].xaxis.set_major_locator(dates.WeekdayLocator(byweekday=MO,interval=1))        
                elif df_new.index[0].weekday()==1:
                    ax[c-1].xaxis.set_major_locator(dates.WeekdayLocator(byweekday=TU,interval=1))        
                elif df_new.index[0].weekday()==2:
                    ax[c-1].xaxis.set_major_locator(dates.WeekdayLocator(byweekday=WE,interval=1))        
                elif df_new.index[0].weekday()==3:
                    ax[c-1].xaxis.set_major_locator(dates.WeekdayLocator(byweekday=TH,interval=1))        
                elif df_new.index[0].weekday()==4:
                    ax[c-1].xaxis.set_major_locator(dates.WeekdayLocator(byweekday=FR,interval=1))                        
                elif df_new.index[0].weekday()==5:
                    ax[c-1].xaxis.set_major_locator(dates.WeekdayLocator(byweekday=SA,interval=1))                        
                elif df_new.index[0].weekday()==6:
                    ax[c-1].xaxis.set_major_locator(dates.WeekdayLocator(byweekday=SU,interval=1))        


                if c==(numberOfColumns-1):                
                    ax[c-1].xaxis.set_major_formatter(dates.DateFormatter('%d\n%b-%y'))

                    plt.setp(ax[c-1].get_xticklabels(which='major'), rotation=0)
                    plt.setp(ax[c-1].get_xticklabels(which='minor'), visible=False)

                else:
                    plt.setp(ax[c-1].get_xticklabels(which='both'), visible=False)
                    ax[c-1].xaxis.label.set_visible(False)

            else:

                if df_new.index[0].weekday()==0:
                    ax[c-1].xaxis.set_minor_locator(dates.WeekdayLocator(byweekday=MO,interval=1))        
                    ax[c-1].xaxis.set_major_locator(dates.WeekdayLocator(byweekday=MO,interval=8))        
                elif df_new.index[0].weekday()==1:
                    ax[c-1].xaxis.set_minor_locator(dates.WeekdayLocator(byweekday=TU,interval=1))        
                    ax[c-1].xaxis.set_major_locator(dates.WeekdayLocator(byweekday=TU,interval=8))        
                elif df_new.index[0].weekday()==2:
                    ax[c-1].xaxis.set_minor_locator(dates.WeekdayLocator(byweekday=WE,interval=1))        
                    ax[c-1].xaxis.set_major_locator(dates.WeekdayLocator(byweekday=WE,interval=8))        
                elif df_new.index[0].weekday()==3:
                    ax[c-1].xaxis.set_minor_locator(dates.WeekdayLocator(byweekday=TH,interval=1))        
                    ax[c-1].xaxis.set_major_locator(dates.WeekdayLocator(byweekday=TH,interval=8))        
                elif df_new.index[0].weekday()==4:
                    ax[c-1].xaxis.set_minor_locator(dates.WeekdayLocator(byweekday=FR,interval=1))                        
                    ax[c-1].xaxis.set_major_locator(dates.WeekdayLocator(byweekday=FR,interval=8))                        
                elif df_new.index[0].weekday()==5:
                    ax[c-1].xaxis.set_minor_locator(dates.WeekdayLocator(byweekday=SA,interval=1))                        
                    ax[c-1].xaxis.set_major_locator(dates.WeekdayLocator(byweekday=SA,interval=8))                        
                elif df_new.index[0].weekday()==6:
                    ax[c-1].xaxis.set_minor_locator(dates.WeekdayLocator(byweekday=SU,interval=1))        
                    ax[c-1].xaxis.set_major_locator(dates.WeekdayLocator(byweekday=SU,interval=8))        


                if c==(numberOfColumns-1):                
                    ax[c-1].xaxis.set_major_formatter(dates.DateFormatter('%d\n%b-%y'))

                    plt.setp(ax[c-1].get_xticklabels(which='major'), rotation=0)
                    plt.setp(ax[c-1].get_xticklabels(which='minor'), visible=False)

                else:
                    plt.setp(ax[c-1].get_xticklabels(which='both'), visible=False)
                    ax[c-1].xaxis.label.set_visible(False)

                    
            ax[c-1].grid(b=1,axis='x',which='major') 
            ax[c-1].grid(axis='y')


        elif data_frequency=='monthly':    

            ax[c-1].xaxis.set_minor_locator(dates.MonthLocator())                
            ax[c-1].xaxis.set_major_locator(dates.YearLocator())
            ax[c-1].yaxis.grid()
            
            if c==(numberOfColumns-1):
                
                ax[c-1].xaxis.set_minor_formatter(dates.DateFormatter('%m'))
                ax[c-1].xaxis.set_major_formatter(dates.DateFormatter('%b-%y'))
                plt.setp(ax[c-1].get_xticklabels(which='major'), rotation=0)
                plt.setp(ax[c-1].get_xticklabels(which='minor'), visible=False)
                
            else:
                plt.setp(ax[c-1].get_xticklabels(which='both'), visible=False)
                ax[c-1].xaxis.label.set_visible(False)


            ax[c-1].grid(b=1,axis='x',which='major') 
            ax[c-1].grid(axis='y')


        elif data_frequency=='hourly':

            ax[c-1].xaxis.set_minor_locator(dates.DayLocator(interval=1))                
            ax[c-1].xaxis.set_major_locator(dates.DayLocator(interval=7))
            ax[c-1].yaxis.grid()

            if c==(numberOfColumns-1):
                #ax[c-1].xaxis.set_minor_formatter(dates.DateFormatter('%H'))
                ax[c-1].xaxis.set_major_formatter(dates.DateFormatter('%D'))
                #plt.setp(ax[c-1].get_xticklabels(which='major'), rotation=0)
                #plt.setp(ax[c-1].get_xticklabels(which='minor'), visible=False)

            else:
                plt.setp(ax[c-1].get_xticklabels(which='both'), visible=False)
                ax[c-1].xaxis.label.set_visible(False)


            ax[c-1].grid(b=1,axis='x',which='both') 
            ax[c-1].grid(axis='y')
            
        elif data_frequency=='minutely' or data_frequency=='5minutely':

            #if data_frequency=='minutely':
            #    ax[c-1].xaxis.set_minor_locator(dates.MinuteLocator())
            #else:
            #    ax[c-1].xaxis.set_minor_locator(dates.MinuteLocator(interval=5))
            
            ax[c-1].xaxis.set_major_locator(dates.HourLocator(interval=24))
            ax[c-1].xaxis.set_minor_locator(dates.HourLocator(interval=6))
            ax[c-1].yaxis.grid()                

            if c==(numberOfColumns-1):
                #ax[c-1].xaxis.set_minor_formatter(dates.DateFormatter('%M'))
                ax[c-1].xaxis.set_major_formatter(dates.DateFormatter('%H'))
                #plt.setp(ax[c-1].get_xticklabels(which='major'), rotation=0)
                #plt.setp(ax[c-1].get_xticklabels(which='minor'), visible=False)

            else:
                plt.setp(ax[c-1].get_xticklabels(which='both'), visible=False)
                ax[c-1].xaxis.label.set_visible(False)


            ax[c-1].grid(b=1,axis='x',which='both') 
            ax[c-1].grid(axis='y')    


