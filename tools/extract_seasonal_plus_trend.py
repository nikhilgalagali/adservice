from rstl import STL
import numpy as np
import pandas as ps

def extract_seasonal(data, num_obs_per_period=None):    

    if num_obs_per_period is None:
        raise ValueError("must supply period length for time series decomposition")

    if list(data.columns.values) != ["timestamp", "value"]:
        data.columns = ["timestamp", "value"]
        
    data = data.set_index('timestamp')    
    decomp = STL(data.value, num_obs_per_period, "periodic", robust=True)

    p = {
        'timestamp': data.index,
        'seasonal': ps.to_numeric(ps.Series(decomp.seasonal))
    }
        
    data_seasonal_component = ps.DataFrame(p)
    data_seasonal_component.set_index('timestamp',inplace=True)    
                
    return data_seasonal_component


def extract_trend(data, num_obs_per_period=None):    

    if num_obs_per_period is None:
        raise ValueError("must supply period length for time series decomposition")

    if list(data.columns.values) != ["timestamp", "value"]:
        data.columns = ["timestamp", "value"]
        
    data = data.set_index('timestamp')

    p = {
        'timestamp': data.index,
        'trend': ps.to_numeric(ps.Series(np.repeat(data.value.median(),len(data)))),
    }
        
    data_trend_component = ps.DataFrame(p)
    data_trend_component.set_index('timestamp',inplace=True)    
        
    return data_trend_component

