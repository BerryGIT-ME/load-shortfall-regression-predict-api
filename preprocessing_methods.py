import pandas as pd


def split_time(df):
    '''
    split_time takes a dataframe with a column named "time" and splits it into four new columns; year, months, days, hours.
    '''
    years = []
    months = []
    days = []
    hours = []
    
    df = df.copy()
    
    def split(composite_time):
        year_month_day, time = composite_time.split(" ")
        year, month, day = year_month_day.split("-")
        hour = time.split(":")[0]

        years.append(float(year))
        months.append(float(month))
        days.append(float(day))
        hours.append(float(hour))
        return
    
    df['time'].apply(split)
    
    df['years'] = years
    df['months'] = months
    df['days'] = days
    df['hours'] = hours
    
    return df


def replace_valencia_pressure(df):
    '''
    Takes a pandas dataframe with three columns 'years', 'month', and 'Valencia_pressure'
    Replaces missing values in the 'Valencia_pressure' column with the monthly average
    '''
    sub_df = df[['years', 'months', 'Valencia_pressure']]
    aggregate = sub_df.groupby(['years', 'months']).mean()
    
    for index, column in df[['Valencia_pressure']].iterrows():
        if pd.isnull(df.loc[index, 'Valencia_pressure']):
            year = df.loc[index, 'years']
            month = df.loc[index, 'months']
            df.loc[index, 'Valencia_pressure'] = aggregate.loc[year].loc[month, 'Valencia_pressure']
            
    return df

def handle_categorical_column(input_df, column_name):
    return pd.get_dummies(input_df, columns=[column_name], drop_first=True)

def handle_categorical_column_v2(input_df):
    input_df = input_df.copy()
    #input_df["Valencia_wind_deg"] = input_df["Valencia_wind_deg"].apply(lambda level: float(level.split("_")[1]))
    input_df["Seville_pressure"] = input_df["Seville_pressure"].apply(lambda level: float(level[2:]))
    return input_df


def handle_colinear_temp_cols(df):
    df['Valencia_temp_min_max'] = df['Valencia_temp'] + df['Valencia_temp_min'] + df['Valencia_temp_max']
    df['Seville_temp_min_max'] = df['Seville_temp'] + df['Seville_temp_min'] + df['Seville_temp_max']
    df['Madrid_temp_min_max'] = df['Madrid_temp'] + df['Madrid_temp_min'] + df['Madrid_temp_max']
    df['Bilbao_temp_min_max'] = df['Bilbao_temp'] + df['Bilbao_temp_min'] + df['Bilbao_temp_max']
    df['Barcelona_temp_min_max'] = df['Barcelona_temp'] + df['Barcelona_temp_min'] + df['Barcelona_temp_max']
    return df

def drop_columns(df):   
    # columns to drop because of colinearity
    colinear_columns = [
        'Valencia_temp_max',
        'Valencia_temp_min',
        'Valencia_temp',
        'Seville_temp_max',
        'Seville_temp_min',
        'Seville_temp',
        'Madrid_temp_max', 
        'Madrid_temp_min',
        'Madrid_temp',
        'Bilbao_temp_max',
        'Bilbao_temp_min',
        'Bilbao_temp',
        'Barcelona_temp_max',
        'Barcelona_temp_min',
        'Barcelona_temp'
    ]
    
    
    
    drop_total = colinear_columns
    df = df.drop(drop_total, axis=1)
    return df
