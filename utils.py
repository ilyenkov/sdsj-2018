import datetime
import pandas as pd

def parse_dt(x):
    if not isinstance(x, str):
        return None
    elif len(x) == len('2010-01-01'):
        return datetime.datetime.strptime(x, '%Y-%m-%d')
    elif len(x) == len('2010-01-01 10:10:10'):
        return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    else:
        return None


def transform_datetime_features(df):
    datetime_columns = [
        col_name
        for col_name in df.columns
        if col_name.startswith('datetime')
    ]
    for col_name in datetime_columns:
        df[col_name] = df[col_name].apply(lambda x: parse_dt(x))
        df['number_weekday_{}'.format(col_name)] = df[col_name].apply(lambda x: x.weekday())
        df['number_month_{}'.format(col_name)] = df[col_name].apply(lambda x: x.month)
        df['number_day_{}'.format(col_name)] = df[col_name].apply(lambda x: x.day)
        df['number_hour_{}'.format(col_name)] = df[col_name].apply(lambda x: x.hour)
        df['number_hour_of_week_{}'.format(col_name)] = df[col_name].apply(lambda x: x.hour + x.weekday() * 24)
        df['number_minute_of_day_{}'.format(col_name)] = df[col_name].apply(lambda x: x.minute + x.hour * 60)
    return df

def make_dtype(filename):
    df = pd.read_csv(filename, nrows=0)

    cols = df.columns.tolist()
    num = [i for i in cols if i[:6]=='number']
    date = [i for i in cols if i[:3]=='dat']
    st = [i for i in cols if i[:3]=='str']
    
    dtype = {}
    
    for i in st:
        dtype[i]='str'
        
    for i in date:
        dtype[i] = 'str'
        
    for i in num:
        dtype[i] = 'float'
    
    return dtype