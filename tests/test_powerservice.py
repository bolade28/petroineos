""" Tests for the powerservice"""
import os
import tempfile
from datetime import datetime

import pandas as pd
import pytest

from powerservice.reporting import compute_file_datetime
from powerservice.reporting import compute_localtime
from powerservice.reporting import compute_start_time
from powerservice.reporting import compute_time_interval
from powerservice.reporting import correct_time_format
from powerservice.reporting import count_daily_records
from powerservice.reporting import generate_reports
from powerservice.reporting import replace_missing_values
from powerservice.trading import check_if_valid_date
from powerservice.trading import generate_new_random_trade_position

cases = [("", False),
         ("01/04/2015", True),
         ("30/04/2015", True),
         (1, False),
         ("01/30/2015", False),
         ("30/04/bla", False)
         ]

test_file_datetime_format = datetime(2022, 9, 12, 10, 35, 12)

data = \
        [('12/09/2022', '00:00', 186.0, '3da7dc1f4f714c07bc7a020ac2860826'),
         ('12/09/2022', '00:05', 145.0, '3da7dc1f4f714c07bc7a020ac2860826'),
         ('12/09/2022', '00:10', 234.0, '3da7dc1f4f714c07bc7a020ac2860826'),
         ('12/09/2022', '00:15', None, '3da7dc1f4f714c07bc7a020ac2860826'),
         ('12/09/2022', '00:20', 309.0, '3da7dc1f4f714c07bc7a020ac2860826')]

columns = ['date', 'time', 'volume', 'id']

df = pd.DataFrame.from_records(data, columns=columns)


@pytest.mark.parametrize("date,expected", cases)
def test_date_checker(date, expected):
    """ Check that only d/m/y formatted date is accepted"""
    assert check_if_valid_date(date) == expected


def test_generate_new_random_trade_position_time_series_len():
    """Check that the period and volume series are of the same length"""
    new_trade = generate_new_random_trade_position(date="01/04/2015")
    period_list = new_trade["time"]
    volume_list = new_trade["volume"]

    assert len(period_list) == len(volume_list)


def test_compute_time_interval():
    """ Check that 5 minutes interval exist between consecutive rows. """
    df_out = compute_localtime(df)
    assert (compute_time_interval(df_out) == 4)


def test_correct_time_format():
    """Validate that the time conform to HH:MM"""
    assert (correct_time_format(df) == 5)


def test_replace_missing_values():
    """ Replacing missing volume field with something sensible. """
    df_new = replace_missing_values(df)
    assert (df_new['volume'].isnull().sum() == 0)


def test_compute_start_time():
    """ Check the start time is 23:00. """
    df_out = compute_localtime(df)
    assert (compute_start_time(df_out) == '23:00')


def test_compute_file_datetime():
    """ Build datetime pattern used as part of filename. """
    assert (compute_file_datetime(test_file_datetime_format) == '20220912_1035')


def test_count_daily_records():
    """ Count number of records for a given trade. """
    assert (count_daily_records(df) == 5)


def test_generate_reports():
    """ Test case for capturing reports generated successfully. """
    generate_reports(tempfile.gettempdir())
    assert (len(os.listdir(tempfile.gettempdir())) > 0)
