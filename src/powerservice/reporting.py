import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from powerservice import trading

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def compute_file_datetime(system_datetime):
    """
    Compute date_time representing system datetime when report will be generated.
    :param system_datetime: Machine date time.
    :return: date_time_format
    """

    date_time = system_datetime.strftime("%Y%m%d_%H%M")
    return date_time


def count_daily_records(df):
    """
    Count the total number of records
    :param df: original input dataframe
    :return: count
    """
    return len(df)


def validate_time_format(str_time, format):
    """
    Validate time format for each rows
    :param str_time: input time
    :param format: input format
    :return: boolean
    """
    res = True
    try:
        res = bool(datetime.strptime(str(str_time), format))
    except ValueError:
        res = False
    return res


def correct_time_format(df):
    """
    Count the number of rows with the correct time format.
    :param df: input dataframe containing original datasets.
    :return: count of rows with valid time format.
    """
    df["validate_time"] = df['time'].apply(lambda x: validate_time_format(x, '%H:%M'))
    df_out = df.query("(validate_time==True)")
    return len(df_out)


def compute_start_time(df):
    """
    Fetch the value of the start time.
    :param df: input dataframe containing original datasets.
    :return: start time value
    """
    return df['CalcTime'].iat[0]


def compute_end_time(df):
    """
    Fetch the value of the end time.
    :param df: input dataframe containing original datasets.
    :return: end time value
    """
    return df['CalcTime'].iat[len(df) - 1]


def compute_data_quality(df):
    """
    Calculating data quality results to be written into CSV file.
    :param df: input dataframe containing original datasets.
    :return: output dataframe containing data quality results
    """
    data = {
        "quality_check": ['Start Time', 'End time', 'Missing Volume', 'Missing Time', 'Correct Time Format',
                          'Interval Check'],
        "value": [compute_start_time(df), compute_end_time(df), compute_missing_volume(df), compute_missing_time(df),
                  correct_time_format(df), compute_time_interval(df)]
    }

    # load data into a DataFrame object:
    df_dq = pd.DataFrame(data)
    return df_dq


def compute_missing_volume(df):
    """
    Count the number of missing volumes.
    :param df: input dataframe containing original datasets.
    :return: count of volume values that are null
    """
    return df['volume'].isnull().sum()


def compute_missing_time(df):
    """
    Count the number of missing time
    :param df: input dataframe containing original datasets.
    :return: count of time values that are null
    """
    return df['time'].isnull().sum()


def compute_time_data_profiling(df):
    """
    Method used for calculating the percentage of null time
    :param df: input dataframe containing original datasets.
    :return: percent_null_time
    """
    percent_null_time = round((compute_missing_time(df) / count_daily_records(df)) * 100, 2)
    return percent_null_time


def compute_volume_data_profiling(df):
    """
    Method used for calculating the percentage of null volumes
    :param df: input dataframe containing original datasets.
    :return: percent_null_volume
    """
    percent_null_volume = round((compute_missing_volume(df) / count_daily_records(df)) * 100, 2)
    return percent_null_volume


def compute_data_profiling(df):
    """
    Method for computing profiling data from dictionary object.
    :param df: input dataframe containing original datasets.
    :return: output dataframe to store data profile information.
    """
    data = {
        "profile": ['Total', 'Volume Null %', 'Time Null %'],
        "value": [count_daily_records(df), compute_volume_data_profiling(df), compute_time_data_profiling(df)]
    }

    # load data into a DataFrame object:
    dff = pd.DataFrame(data)
    return dff


def replace_missing_values(df):
    """
    Used for replacing missing values with something sensible
    :param df: input dataframe containing all the records.
    :return: output a dataframe with missing values replaced with mean value.
    """
    df['volume'] = df['volume'].fillna((df['volume'].mean()))
    return df


def generate_csv_report_from_df(df, file_pattern, location=None):
    """
    Generate Main CSV file to store output data after transformation has been applied.
    :param df: dataframe containing data to be written into csv file.
    :param file_pattern: file naming convention
    :param location: The directory where csv files generated are to be stored.
    :return:
    """

    df['hour'] = df['CalcTime'].str.slice(0, 2)
    df['date'] = df['CalcDate']

    df1 = df[['date', 'hour', 'volume']]

    df3 = df1.groupby(by=["date", "hour"]).sum().reset_index()
    """
    drop the date column as this is not required in the output report.
    """
    del df3['date']
    # Rename fields according to requirements.
    df3['Local Time'] = df3['hour'] + ":00"
    df3['Volume'] = df['volume']
    df3 = df3[['Local Time', 'Volume']]
    full_path = location+'/' + file_pattern

    # Write the csv file generated to CSV file
    df3.to_csv(full_path, index=False)

    logging.info('Generated CSV file ({}) - '.format(full_path))


def generate_csv_report_data_qual_or_prof(df_qual_or_prof, file_pattern, location):
    """
    Generate CSV file to store data profiling or data quality information.
    :param df_qual_or_prof: dataframe containing data to be stored in CSV
    :param file_pattern: file naming convention.
    :param location: The directory where csv files generated are to be stored.
    :return:
    """
    full_path = location + '/' + file_pattern
    logging.info('Writing data profiling report to CSV file ({}) - '.format(full_path))

    # Write the csv file generated to CSV file
    df_qual_or_prof.to_csv(full_path, index=False)


def compute_time_interval(df):
    """
    Compute time interval between each consecutive rows.
    :param df: dataframe
    :return: number of consecutive rows with 5 minutes as the interval
    """
    df['lag_datetime'] = df['datetime'].shift(1)
    df['diff_minutes'] = df['datetime'] - df['lag_datetime']
    df['diff_minutes'] = df['diff_minutes'] / np.timedelta64(1, 'm')

    # Query the number of rows where time interval between 2 consecutive rows is 5 minutes
    df_tm_int = df.query("(diff_minutes==5)")

    return len(df_tm_int)


def compute_localtime(df):
    """
    This is for computing the local time column tp be written into CSV file.
    :param df: input dataframe to be modified.
    :return: dataframe output to show local time starting from 23:00 the previous day.
    """
    t11 = pd.to_datetime(df['time'])
    df['datetime'] = t11 - pd.Timedelta(hours=1)
    df['CalcDate'] = df['datetime'].dt.strftime('%d-%m-%y')
    df['CalcTime'] = df['datetime'].dt.strftime('%H:%M')
    return df


def generate_reports(location):
    """
    Wrapper method for generating all CSV reports
    :param location: directory or folder containing csv files.
    :return:
    """
    logging.info('Invocation of report generating function.')
    trades = trading.get_trades(date='12/09/2022')

    df = pd.DataFrame(trades[0])
    df = compute_localtime(df)

    df_prof = compute_data_profiling(df)
    today = datetime.now()
    file_part = compute_file_datetime(today)

    dp_file_pattern = 'PowerPosition_{}_data_profiling.csv'.format(file_part)
    generate_csv_report_data_qual_or_prof(df_prof, dp_file_pattern, location)

    df_qual = compute_data_quality(df)
    dq_file_pattern = 'PowerPosition_{}_data_quality.csv'.format(file_part)
    generate_csv_report_data_qual_or_prof(df_qual, dq_file_pattern, location)

    file_pattern = 'PowerPosition_{}.csv'.format(file_part)
    generate_csv_report_from_df(df, file_pattern, location)


if __name__ == '__main__':

    # Get the number of arguments
    numOfArgs = len(sys.argv)

    if numOfArgs == 2:
        dirLocation = sys.argv[1]
        if os.path.exists(dirLocation):
            logging.info("File exist")
            generate_reports(dirLocation)
        else:
            logging.error('Output directory has not been created. Do so before proceeding')
    else:
        logging.error('ERROR. This programme should be executed using the form below:')
        logging.error('python reporting.py <CSV Output Path Directory>')
