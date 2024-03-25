# imports 
import datetime as dt 
import pandas as pd
import numpy as np
import torch
import os
from sklearn.preprocessing import MinMaxScaler 


# data prosessing 

# load data
def load_data_new(path_vars, path_qobs, path_attrs, ID, variables, start_date, stop_date, dataset='lamah', delimiter=';', qobs_mm='False'):
    """
    This function loads the data from the chosen dataset. 
    
    LamaH-CE dataset: 
    The function loads both the meteorological variables as well as the discharge observations for the 
    selected catchment combines them into one dataframe. The dataframe is clipped according to the time span for which 
    both meteorological data as well as discharge observations are available.
    
    path_vars:      Path to meteorological variables 
    path_qobs:      Path to discharge observations
    ID:             Catchment ID 
    dataset:        Dataset
    environment:    Set to "colab" or "local"
    delimiter:      Set according to delimiter of CSV file

    """
    df_combi = pd.DataFrame()
    if dataset=='lamah':
        file_name = f'ID_{ID}.csv'

        file_vars = os.path.join(path_vars, file_name)
        file_qobs = os.path.join(path_qobs, file_name)

        # universal
        df_vars = pd.read_csv(file_vars, delimiter=delimiter)
        df_qobs = pd.read_csv(file_qobs, delimiter=delimiter)
        
        # create datetime column
        df_vars['date_str'] = df_vars['YYYY'].astype(str) + '-' + df_vars['MM'].astype(str) + '-' + df_vars['DD'].astype(str)
        df_vars['date'] = pd.to_datetime(df_vars['date_str'])
        
        # drop unused columns 
        df_vars.drop(columns=['YYYY', 'MM', 'DD', 'date_str'], inplace=True)
        
        # create datetime column
        df_qobs['date_str'] = df_qobs['YYYY'].astype(str) + '-' + df_qobs['MM'].astype(str) + '-' + df_qobs['DD'].astype(str)
        df_qobs['date'] = pd.to_datetime(df_qobs['date_str'])

        # drop unused columns 
        df_qobs.drop(columns=['YYYY', 'MM', 'DD', 'date_str'], inplace=True)
        
        # clip dataframe to time span for which both meteorological variables as well as discharge observations are available
        start_date_dt = pd.to_datetime(start_date)
        ### start_date = max(df_vars.date.iloc[0], df_qobs.date.iloc[0]) # this is replaced with the line above so that the start_date can be set manually, also this is not needed anyways as the dfs are already sorted out so that they all start 1981-01-01
        stop_date_dt = pd.to_datetime(stop_date)
        ### end_date = min(df_vars.date.iloc[-1], df_qobs.date.iloc[-1])
        
        
        df_vars = df_vars[(df_vars['date'] >= start_date_dt) & (df_vars['date'] <= stop_date_dt)]
        df_qobs = df_qobs[(df_qobs['date'] >= start_date_dt) & (df_qobs['date'] <= stop_date_dt)]
        
        # new column names
        new_names = {'2m_temp_max': 'tmax',  '2m_temp_mean':'tmean', '2m_temp_min': 'tmin', '2m_dp_temp_max':'tdpmax', '2m_dp_temp_mean':'tdpmean','2m_dp_temp_min':'tdpmin', '10m_wind_u':'windu', '10m_wind_v':'windv', 'fcst_alb':'albedo', 'swe':'swe', 'surf_net_solar_rad_mean': 'sradmean', 'surf_net_therm_rad_mean':'tradmean', 'surf_press':'spress', 'total_et': 'et',  'prec': 'prcp'}
        df_vars.rename(columns=new_names, inplace=True)
        

        df_combi = pd.concat([df_vars[variables], df_qobs[['date', 'qobs']]], axis=1)
        
        # change the units of qobs to mm if qobs_mm = True
        if qobs_mm == True:
            # first load the catchment area from the attributes file 
            catchment_attrs = pd.read_csv(path_attrs + "Catchment_attributes.csv", delimiter=";")
            # area_calc is in km2
            area_km2 = catchment_attrs.loc[catchment_attrs['ID']==ID, 'area_calc'].item()
            # from m3/s to m3/d
            df_combi['qobs_s'] = df_combi['qobs'] * 60 * 60 * 24
            # convert area_calc form km2 to m2
            area_m2 = area_km2 * 10**6
            # convert qobs from m3/d to mm/d
            df_combi['qobs_mm'] = (df_combi['qobs_s'] * 1000) / area_m2
            # drop unused stuff 
            df_combi.drop(columns=['qobs_s'], inplace=True)
        
        
        # adjust dtypes if it is part of the dataframe
        if 'sradmean' in df_combi.columns:
            df_combi['sradmean'] = df_combi['sradmean'].astype('float64')
        
        if 'tradmean' in df_combi.columns:
            df_combi['tradmean'] = df_combi['tradmean'].astype('float64')
        
        df_combi.set_index('date', inplace=True)
    
    # add EOBS part here
    if dataset=='eobs':
        
        file_name = f'ID_{ID}.csv'

        file_vars = os.path.join(path_vars, file_name)
        file_qobs = os.path.join(path_qobs, file_name)

        # universal
        df_vars = pd.read_csv(file_vars, delimiter=",") # set manually to comma! 
        df_qobs = pd.read_csv(file_qobs, delimiter=delimiter)
        
        df_vars.head()
        
        # create datetime column
        #### df_vars['date_str'] = df_vars['YYYY'].astype(str) + '-' + df_vars['MM'].astype(str) + '-' + df_vars['DD'].astype(str)
        df_vars['date'] = pd.to_datetime(df_vars['time'].astype(str))
        
        # drop unused columns 
        ### df_vars.drop(columns=['YYYY', 'MM', 'DD', 'date_str'], inplace=True)
        
        # create datetime column
        df_qobs['date_str'] = df_qobs['YYYY'].astype(str) + '-' + df_qobs['MM'].astype(str) + '-' + df_qobs['DD'].astype(str)
        df_qobs['date'] = pd.to_datetime(df_qobs['date_str'])

        # drop unused columns 
        df_qobs.drop(columns=['YYYY', 'MM', 'DD', 'date_str'], inplace=True)
        
        # clip dataframe to time span for which both meteorological variables as well as discharge observations are available
        start_date_dt = pd.to_datetime(start_date)
        ### start_date = max(df_vars.date.iloc[0], df_qobs.date.iloc[0]) # this is replaced with the line above so that the start_date can be set manually, also this is not needed anyways as the dfs are already sorted out so that they all start 1981-01-01
        stop_date_dt = pd.to_datetime(stop_date)
        ### end_date = min(df_vars.date.iloc[-1], df_qobs.date.iloc[-1])
        
        
        df_vars = df_vars[(df_vars['date'] >= start_date_dt) & (df_vars['date'] <= stop_date_dt)]
        df_qobs = df_qobs[(df_qobs['date'] >= start_date_dt) & (df_qobs['date'] <= stop_date_dt)]
        
        # names 
        # 'rr', 'tg', 'tn', 'tx', 'pp', 'hu', 'fg', 'qq', 'fal', 'ET0', 'daylength', 'pev'
        new_names = {'rr': 'prcp', 'tg': 'tmean', 'tn': 'tmin', 'tx': 'tmax', 'pp': 'seapress', 'hu': 'humidity', 'fg': 'windspeed', 'qq': 'srad', 'fal': 'albedo', 'ET0': 'pet'}
        df_vars.rename(columns=new_names, inplace=True)


        df_combi = pd.concat([df_vars[variables], df_qobs[['date', 'qobs']]], axis=1)
        
        # change the units of qobs to mm if qobs_mm = True
        if qobs_mm == True:
            # first load the catchment area from the attributes file 
            catchment_attrs = pd.read_csv(path_attrs + "Catchment_attributes.csv", delimiter=";")
            # area_calc is in km2
            area_km2 = catchment_attrs.loc[catchment_attrs['ID']==ID, 'area_calc'].item()
            # from m3/s to m3/d
            df_combi['qobs_s'] = df_combi['qobs'] * 60 * 60 * 24
            # convert area_calc form km2 to m2
            area_m2 = area_km2 * 10**6
            # convert qobs from m3/d to mm/d
            df_combi['qobs_mm'] = (df_combi['qobs_s'] * 1000) / area_m2
            # drop unused stuff 
            df_combi.drop(columns=['qobs_s'], inplace=True)
        

        df_combi.set_index('date', inplace=True)

    return df_combi

# data preparation incl. train, val test split
def prep_data_new(df, start_train: str, stop_train: str, start_val: str, stop_val: str, start_test: str, stop_test: str, variables, tensors=False, return_only_df_combi=False):
    """
    This function does the data preparation on the dataframe that has been loaded with the function before. This means the columns are already 
    renamed. 
    Enter dates in this format: "YYYY-MM-DD" 
    """
    
    # convert string to datetime
    start_train = dt.datetime.strptime(start_train, "%Y-%m-%d")
    stop_train = dt.datetime.strptime(stop_train, "%Y-%m-%d")
    start_val = dt.datetime.strptime(start_val, "%Y-%m-%d")
    stop_val = dt.datetime.strptime(stop_val, "%Y-%m-%d")
    start_test = dt.datetime.strptime(start_test, "%Y-%m-%d")
    stop_test = dt.datetime.strptime(stop_test, "%Y-%m-%d")

    # create intervals 
    ### < or <= ? 
    df_train = df.loc[(df.index >= start_train) & (df.index <= stop_train)]
    df_val = df.loc[(df.index >= start_val) & (df.index <= stop_val)]
    df_test = df.loc[(df.index >= start_test) & (df.index <= stop_test)]

    # normalize using scikit learn MinMaxScaler
    scaler = MinMaxScaler()
    
    ### added ".values"
    train_data = scaler.fit_transform(df_train.values)
    val_data = scaler.transform(df_val.values)
    test_data = scaler.transform(df_test.values)
    
    
    # should this be turned into dataframes again? 
    # variables is causing an issue here as it has no elemen 'qobs'
    variables = [*variables, 'qobs', 'qobs_mm'] 
    
    df_train = pd.DataFrame(train_data, index = df_train.index, columns = variables)
    df_val = pd.DataFrame(val_data, index = df_val.index, columns = variables) 
    df_test = pd.DataFrame(test_data, index = df_test.index, columns = variables)


    #### added to return the full dataframe with still appropriate scaling approach 
    df_combined = pd.concat([df_train, df_val, df_test])
    
    # create tensors here
    if tensors==True:
        X_train = torch.tensor(df_train.values, dtype=torch.float32, requires_grad = False)
        X_val = torch.tensor(df_val.values, dtype=torch.float32, requires_grad = False)
        X_test = torch.tensor(df_test.values, dtype=torch.float32, requires_grad = False)
        y_train = torch.tensor(df_train['qobs_mm'].values, dtype=torch.float32, requires_grad = False)
        y_val = torch.tensor(df_val['qobs_mm'].values, dtype=torch.float32, requires_grad = False)
        y_test = torch.tensor(df_test['qobs_mm'].values, dtype=torch.float32, requires_grad = False)
            
    if tensors==False:
        X_train = df_train
        X_val = df_val
        X_test = df_test
        y_train = df_train['qobs_mm']
        y_val = df_val['qobs_mm']
        y_test = df_test['qobs_mm']
    
    if return_only_df_combi==True:
        return df_combined, scaler
    else:
    # return also scaler for inverse transform after training
        return X_train, X_val, X_test, y_train, y_val, y_test, scaler, df_combined
    
    
### these are not needed anymore remove
# data preparation incl. train, val test split
def prep_data_hybrid_new(df, start_train: str, stop_train: str, start_val: str, stop_val: str, start_test: str, stop_test: str, variables):
    """
    This function does the data preparation on the dataframe that has been loaded with the function before. This means the columns are already 
    renamed. 
    Enter dates in this format: "YYYY-MM-DD" 
    """
    
    # convert string to datetime
    start_train = dt.datetime.strptime(start_train, "%Y-%m-%d")
    stop_train = dt.datetime.strptime(stop_train, "%Y-%m-%d")
    start_val = dt.datetime.strptime(start_val, "%Y-%m-%d")
    stop_val = dt.datetime.strptime(stop_val, "%Y-%m-%d")
    start_test = dt.datetime.strptime(start_test, "%Y-%m-%d")
    stop_test = dt.datetime.strptime(stop_test, "%Y-%m-%d")

    # create intervals 
    ### < or <= ? #####################################################################
    df_train = df.loc[(df.index >= start_train) & (df.index <= stop_train)]
    df_val = df.loc[(df.index >= start_val) & (df.index <= stop_val)]
    df_test = df.loc[(df.index >= start_test) & (df.index <= stop_test)]

    # normalize using scikit learn MinMaxScaler
    scaler = MinMaxScaler()
    
    ### added ".values"
    train_data = scaler.fit_transform(df_train.values)
    val_data = scaler.transform(df_val.values)
    test_data = scaler.transform(df_test.values)
    
    
    # should this be turned into dataframes again? 
    # variables is causing an issue here as it has no element 'qobs'
    """ Note: before calling this function, qobs is already transformed to qobs_mm in the df""" # this could be added to this function 
    variables = [*variables, 'qobs_mm'] 
    
    df_train = pd.DataFrame(train_data, index = df_train.index, columns = variables)
    df_val = pd.DataFrame(val_data, index = df_val.index, columns = variables) 
    df_test = pd.DataFrame(test_data, index = df_test.index, columns = variables)
    
    return df_train, df_val, df_test, scaler

### not used remove 
# data preparation incl. train, val test split --> Note: this also includes a calibration period df and a short train df which is the training period that starts after the calibration period
def prep_data_hybrid_long(df, start_cal: str, stop_cal: str, start_train_short: str, start_train: str, stop_train: str, start_val: str, stop_val: str, start_test: str, stop_test: str, variables, scaling=False):
    """
    This function does the data preparation on the dataframe that has been loaded with the function before. This means the columns are already 
    renamed. 
    Enter dates in this format: "YYYY-MM-DD" 
    """
    
    # convert string to datetime
    start_cal = dt.datetime.strptime(start_cal, "%Y-%m-%d")
    stop_cal = dt.datetime.strptime(stop_cal, "%Y-%m-%d")
    start_train_short = dt.datetime.strptime(start_train_short, "%Y-%m-%d") 
    start_train = dt.datetime.strptime(start_train, "%Y-%m-%d")
    stop_train = dt.datetime.strptime(stop_train, "%Y-%m-%d")
    start_val = dt.datetime.strptime(start_val, "%Y-%m-%d")
    stop_val = dt.datetime.strptime(stop_val, "%Y-%m-%d")
    start_test = dt.datetime.strptime(start_test, "%Y-%m-%d")
    stop_test = dt.datetime.strptime(stop_test, "%Y-%m-%d")

    # create intervals 
    # < or <=? <= was chosen
    df_cal = df.loc[(df.index >= start_cal) & (df.index <= stop_cal)] # <
    df_train_short = df.loc[(df.index >= start_train_short) & (df.index <= stop_train)] # <
    df_train = df.loc[(df.index >= start_train) & (df.index <= stop_train)] # <
    df_val = df.loc[(df.index >= start_val) & (df.index <= stop_val)] # <
    df_test = df.loc[(df.index >= start_test) & (df.index <= stop_test)] # <
    
    if scaling==False:
        return df_cal, df_train_short, df_train, df_val, df_test

    if scaling==True:
        # normalize using scikit learn MinMaxScaler
        scaler = MinMaxScaler()
        
        ### added ".values"
        train_data = scaler.fit_transform(df_train.values)
        
        ### now we create cal_data and train_short_data based on train_data so that the "fit_transform" is applied to the whole training period
        
        
        """ Figure out how to index this! """
        
        #cal_data = train_data[]
        #train_data_short = train_data[]
        
        val_data = scaler.transform(df_val.values)
        test_data = scaler.transform(df_test.values)
        
        
        # should this be turned into dataframes again? 
        # variables is causing an issue here as it has no element 'qobs'
        """ Note: before calling this function, qobs is already transformed to qobs_mm in the df""" # this could be added to this function 
        variables = [*variables, 'qobs_mm'] 
        
        df_train = pd.DataFrame(train_data, index = df_train.index, columns = variables)
        df_val = pd.DataFrame(val_data, index = df_val.index, columns = variables) 
        df_test = pd.DataFrame(test_data, index = df_test.index, columns = variables)
        
        return df_train, df_val, df_test, scaler


# data preparation for LSTM incl. train, val test split
def prep_data_lstm_new(df, start_train: str, stop_train: str, start_val: str, stop_val: str, start_test: str, stop_test: str, window_size=365, step_size=1, tensors=False):
    """
    Enter dates in this format:     "YYYY-MM-DD" 
    window_size:                    Sequence length
    step_size:                      Shift
    """
    
    # number of columns wihtout index / date
    n_cols = df.shape[1]
    
    # convert string to datetime
    start_train = dt.datetime.strptime(start_train, "%Y-%m-%d")
    stop_train = dt.datetime.strptime(stop_train, "%Y-%m-%d")
    start_val = dt.datetime.strptime(start_val, "%Y-%m-%d")
    stop_val = dt.datetime.strptime(stop_val, "%Y-%m-%d")
    start_test = dt.datetime.strptime(start_test, "%Y-%m-%d")
    stop_test = dt.datetime.strptime(stop_test, "%Y-%m-%d")

    # create intervals 
    ### < or <= ? 
    df_train = df.loc[(df.index >= start_train) & (df.index <= stop_train)]
    df_val = df.loc[(df.index >= start_val) & (df.index <= stop_val)]
    df_test = df.loc[(df.index >= start_test) & (df.index <= stop_test)]

    # normalize using scikit learn MinMaxScaler
    scaler = MinMaxScaler()
    
    
    ### added ".values"
    train_data = scaler.fit_transform(df_train.values)
    val_data = scaler.transform(df_val.values)
    test_data = scaler.transform(df_test.values)
    
    # create overlapping windows of training data
    X_train, y_train = [], []
    for i in range(window_size, len(train_data), step_size):
        X_train.append(train_data[i-window_size:i, :n_cols-1]) ###
        y_train.append(train_data[i, n_cols-1]) ###
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # create overlapping windows of validation data
    X_val, y_val = [], []
    for i in range(window_size, len(val_data), step_size):
        X_val.append(val_data[i-window_size:i, :n_cols-1]) ###
        y_val.append(val_data[i, n_cols-1]) ###
    X_val, y_val = np.array(X_val), np.array(y_val)
    
    # create overlapping windows of test data
    X_test, y_test = [], []
    for i in range(window_size, len(test_data), step_size):
        X_test.append(test_data[i-window_size:i, :n_cols-1]) ###
        y_test.append(test_data[i, n_cols-1]) ###
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    # reshape the data for LSTM
    y_train, y_val, y_test = np.reshape(y_train, (len(y_train), 1)), np.reshape(y_val, (len(y_val), 1)), np.reshape(y_test, (len(y_test), 1))
    
    # create tensors here
    if tensors==True:
        X_train = torch.tensor(X_train, dtype=torch.float32, requires_grad = False)
        X_val = torch.tensor(X_val, dtype=torch.float32, requires_grad = False)
        X_test = torch.tensor(X_test, dtype=torch.float32, requires_grad = False)
        y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad = False)
        y_val = torch.tensor(y_val, dtype=torch.float32, requires_grad = False)
        y_test = torch.tensor(y_test, dtype=torch.float32, requires_grad = False)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler
    