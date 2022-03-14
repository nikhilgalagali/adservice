# -*- coding: utf-8 -*-
from collections import namedtuple
from tools.extract_seasonal_plus_trend import extract_seasonal,extract_trend
from pandas import DataFrame, Timestamp
import pandas as pd

Direction = namedtuple('Direction', ['one_tail', 'upper_tail'])

def long_term_seasonal_plus_trend(df, cycle_period,longterm=True,
                                  piecewise_median_period_weeks=2):

    if piecewise_median_period_weeks < 2:
        raise ValueError(
            "piecewise_median_period_weeks must be greater than equal to 2 periods")
                
    # Preparing data for trend extraction    
    if longterm:

        # The plus 1 here is because STL requires more than 2 period of data
        num_obs_in_each_window = cycle_period * piecewise_median_period_weeks+1
        
        all_data = []

        for j in range(0, len(df.Date), num_obs_in_each_window):
            
            start_index = j
            end_index = min(start_index+num_obs_in_each_window,len(df.Date))
            
            if (start_index+2*num_obs_in_each_window>len(df.Date)):
                sub_df = df[start_index:]
                all_data.append(sub_df)
                break
            else:
                sub_df = df.iloc[start_index:end_index]
            all_data.append(sub_df)
    else:
        all_data = [df]    
                
    # Trend extraction: uses piecewise median            
    trend_dataframe = DataFrame(columns=['trend'])
    
    for i in range(len(all_data)):

        data_decomp = extract_trend(all_data[i],num_obs_per_period=cycle_period)
        trend_dataframe = trend_dataframe.append(data_decomp)

    # Seasonal component extraction
    seasonal_dataframe = extract_seasonal(df,num_obs_per_period=cycle_period)
        
    seasonal_plus_trend=pd.concat([trend_dataframe,seasonal_dataframe],axis=1)    

    return {
        'plot': None,
        'seasonal_plus_trend':seasonal_plus_trend
    }
