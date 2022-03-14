import matplotlib
matplotlib.use('agg')
import numpy as np

def normalize_data(df_complete_data):

    numberOfColumns=len(df_complete_data.columns)
    data_scale=[]    
    for c in range(1,numberOfColumns):
        data_scale.append(np.floor(np.log10(np.abs(df_complete_data['KPI_{}'.format(c)]).max())))   
        df_complete_data['KPI_{}'.format(c)]=df_complete_data['KPI_{}'.format(c)]/10**data_scale[c-1]    
        
    return data_scale

# Function to normalize data before prediction
def pred_normalize_data(df_complete_data,data_scale):

    numberOfColumns=len(df_complete_data.columns)
    for c in range(1,numberOfColumns):
        df_complete_data['KPI_{}'.format(c)]=df_complete_data['KPI_{}'.format(c)]/10**data_scale[c-1]      
        