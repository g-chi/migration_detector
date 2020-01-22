import graphlab as gl
import pandas as pd
import numpy as np
import os
from .core import TrajRecord


def read_csv(file_path):
    user_daily_loc_count = gl.SFrame.read_csv(file_path, verbose=False)
    user_daily_loc_count['user_id'] = user_daily_loc_count['user_id'].astype(str)
    # Prepare migration record
    # Assign day index to each date
    start_date_ori = str(user_daily_loc_count['date'].min())
    end_date_ori = str(user_daily_loc_count['date'].max())
    # MM/DD/YYYY
    start_date = '/'.join([start_date_ori[4:6], start_date_ori[6:],
                           start_date_ori[:4]])
    end_date = '/'.join([end_date_ori[4:6], end_date_ori[6:], end_date_ori[:4]])
    all_date = pd.date_range(start=start_date, end=end_date)
    all_date_new = [int(str(x)[:4] + str(x)[5:7] + str(x)[8:10])
                    for x in all_date]
    date2index = dict(zip(all_date_new, range(len(all_date_new))))
    index2date = dict(zip(range(len(all_date_new)), all_date_new))

    end_date_long_ori = str(pd.Timestamp(end_date)+pd.Timedelta('200 day'))
    all_date_long = pd.date_range(start=start_date, end=end_date_long_ori)
    all_date_long_new = [int(str(x)[:4] + str(x)[5:7] + str(x)[8:10])
                         for x in all_date_long]
    date_num_long = gl.SFrame({'date': all_date_long_new,
                               'date_num': range(len(all_date_long_new))})

    migration_df = user_daily_loc_count
    migration_df['date_num'] = migration_df.apply(
        lambda x: date2index[x['date']]
    )
    # Aggregate user daily records
    user_loc_date_agg = migration_df.groupby(
        ['user_id', 'location'],
        {'all_date': gl.aggregate.CONCAT('date_num')}
    )
    user_loc_agg = user_loc_date_agg.groupby(
        ['user_id'],
        {'all_record': gl.aggregate.CONCAT('location', 'all_date')}
    )
    traj = TrajRecord(user_loc_agg, migration_df, index2date, date_num_long)
    return traj


def to_csv(result, result_path='result', file_name='migration_event.csv'):
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    save_file = os.path.join(result_path, file_name)
    result.select_columns(
        ['user_id', 'home', 'destination', 'migration_date',
         'uncertainty', 'num_error_day',
         'home_start', 'home_end',
         'destination_start', 'destination_end',
         'home_start_date', 'home_end_date',
         'destination_start_date', 'destination_end_date']
    ).export_csv(save_file)
