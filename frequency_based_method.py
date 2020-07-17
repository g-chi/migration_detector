
# coding: utf-8

from __future__ import division
import graphlab as gl
import pandas as pd
import numpy as np
import os
import copy
from array import array
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta
from calendar import monthrange

def month_index(x):
    date_str = str(x)
    year = int(date_str[:4])
    month = int(date_str[4:6])
    month_idx = month+12*(year - start_year)
    return month_idx


# The input file1 [user_hourly_tower_dist] should contain the following columns: 
# user_id, date(yyyymmdd, int), hour(int), cell_tower_id, district_id, nearby_tower_list(list)
# If input file1 [user_hourly_tower_dist] does not contain the 'district_id' or 'nearby_tower' column, 
# could use the codes at the end to match the district,
# or find the nearby 1km tower and add the list to the sframe as a new column.
user_hourly_tower_dist = gl.SFrame(pd.read_pickle('./sample_user_hourly_tower_dist.pkl'))
start_year = int(str(min(user_hourly_tower_dist['date']))[:4])
# add the column of month index and select data at night 
user_hourly_tower_dist['month_idx'] = user_hourly_tower_dist['date'].apply(lambda x: month_index(x))
user_hourly_tower_dist_night = user_hourly_tower_dist.filter_by(range(0,10)+range(19,25),'hour')

# The input file2 [tower_district] should contain cell_tower_id and district_id 
# to provide a reference for (cell_tower_id -> district_id)
tower_district = gl.SFrame.read_csv('./sample_tower_district.csv', verbose=False)

# Create date table 
start_date_ori = str(min(user_hourly_tower_dist['date']))
end_date_ori = str(max(user_hourly_tower_dist['date']))
start_date = start_date_ori[4:6]+'/'+start_date_ori[6:]+'/'+start_date_ori[:4] # M/D/YYYY
end_date = end_date_ori[4:6]+'/'+end_date_ori[6:]+'/'+end_date_ori[:4] # M/D/YYYY
all_date = pd.date_range(start=start_date, end=end_date)
all_date_new = [int(str(x)[:4] + str(x)[5:7] + str(x)[8:10]) for x in all_date]
date_num = gl.SFrame({'date': all_date_new, 'date_num': range(len(all_date_new))})
date_num['month_idx'] = date_num['date'].apply(lambda x: month_index(x))

def find_migration(x):
    """
    x is the dictionary for monthly home location for each user, as {month_idx: home_loc}。
    Return a dictionary of migration for this user, {mig_month: [home, destination]}
    """
    month_list = np.array(x.keys())
    dist_list = np.array(x.values())
    dist_list_next = np.append(dist_list[1:],dist_list[-1])
    result_dict = {}
    
    # find if the district for this month and the next month are different
    dist_diff = dist_list - dist_list_next
    check_period_idx = np.where(dist_diff!=0)[0]
    if len(check_period_idx)>0:
        for idx in check_period_idx:
            if (idx >= 2) and (idx+3 <= len(dist_list) -1):
                if ((dist_list[idx-2] == dist_list[idx-1] == dist_list[idx]) and 
                (dist_list[idx+1] == dist_list[idx+2] == dist_list[idx+3]) and
                (dist_list[idx] != dist_list[idx+1]) and
                 month_list[idx-2]+5 == month_list[idx+3]):
                    mig_month = month_list[idx] #the third month
                    home = dist_list[idx]
                    destination = dist_list[idx+1]
                    result_dict[mig_month] = [home, destination]
    return result_dict


## Method 1: 
# The majority of both outgoing and incoming calls and texts were made (amount of activities criterion)

def find_top1_loc_by_count(x):
    """
    x should be a dictionary listing the counts for each location appeared each month, 
    as {location_ID: location_monthly_count}
    """
    loc_dict = x
    top1_count = max(loc_dict.values())
    top1_loc = [k for k,v in loc_dict.iteritems() if v == top1_count]
    top1_loc = top1_loc[0]
    return top1_loc


def method1_monthly_loc(x):
    """
    Infer monthly location as where the majority of both outgoing and incoming calls and texts were made 
    (amount of activities criterion)
    
    x should be a SFrame showing users' hourly district location, 
    with a month_idx column indicating the month index for each date.
    """
    # using the raw network data(from hourly to daily)
    sel_user_monthly_m1_2 = (x.groupby(['user_id', 'month_idx', 'Dist_ID'], 
                           {'monthly_dist_count': gl.aggregate.COUNT('user_id')}))
    sel_user_monthly_m1_3 = (sel_user_monthly_m1_2.groupby(['user_id', 'month_idx'], 
                            {'dist_count_night': gl.aggregate.CONCAT('Dist_ID', 'monthly_dist_count')}))
    sel_user_monthly_m1_3['home_loc'] = (sel_user_monthly_m1_3['dist_count_night']
                                         .apply(lambda x: find_top1_loc_by_count(x)))

    method1_data2 = (sel_user_monthly_m1_3.groupby(['user_id'],
                                             {'monthly_home': gl.aggregate.CONCAT('month_idx', 'home_loc')}))
    method1_data2['freq_mig_result'] = method1_data2['monthly_home'].apply(lambda x: find_migration(x))
    freq_stack = (method1_data2.stack('freq_mig_result', new_column_name=['month_idx', 'home_des'])
              .dropna().unpack('home_des'))
    freq_stack.rename({'home_des.0':'home', 'home_des.1':'destination'})
    freq_stack['home'] = freq_stack['home'].apply(lambda x: int(x))
    freq_stack['destination'] = freq_stack['destination'].apply(lambda x: int(x))
    freq_stack = freq_stack.select_columns(['user_id','month_idx','home','destination'])
    
    return(freq_stack)


## Method 2: 
# The maximum number of distinct days with phone activities – both outgoing and incoming calls and texts – was observed (amount of distinct days criterion)

def method2_monthly_loc(x):
    """
    Infer monthly location as where the maximum number of distinct days with phone activities 
    – both outgoing and incoming calls and texts – was observed. (amount of distinct days criterion)
    
    x should be a SFrame showing users' hourly district location, 
    with a month_idx column indicating the month index for each date.
    """
    sel_user_monthly_m22_1 = (x.groupby(['user_id', 'month_idx', 'Dist_ID'], 
                       {'distinct_date_count': gl.aggregate.COUNT_DISTINCT('date')}))
    sel_user_monthly_m22_2 = (sel_user_monthly_m22_1.groupby(['user_id', 'month_idx'], 
                            {'distinct_date_count_list': gl.aggregate.CONCAT('Dist_ID', 'distinct_date_count')}))
    sel_user_monthly_m22_2['home_loc'] = (sel_user_monthly_m22_2['distinct_date_count_list']
                                         .apply(lambda x: find_top1_loc_by_count(x)))

    method2_data2 = (sel_user_monthly_m22_2.groupby(['user_id'],
                                             {'monthly_home': gl.aggregate.CONCAT('month_idx', 'home_loc')}))
    method2_data2['freq_mig_result'] = method2_data2['monthly_home'].apply(lambda x: find_migration(x))
    freq_stack = (method2_data2.stack('freq_mig_result', new_column_name=['month_idx', 'home_des'])
              .dropna().unpack('home_des'))
    freq_stack.rename({'home_des.0':'home', 'home_des.1':'destination'})
    freq_stack['home'] = freq_stack['home'].apply(lambda x: int(x))
    freq_stack['destination'] = freq_stack['destination'].apply(lambda x: int(x))
    freq_stack = freq_stack.select_columns(['user_id','month_idx','home','destination'])
    
    return(freq_stack)


## Method 2 + propDays

date_num['year'] = date_num['date'].apply(lambda x: int(str(x)[:4]))
date_num['month'] = date_num['date'].apply(lambda x: int(str(x)[4:6]))
month_len_sf = date_num.select_columns(['year','month','month_idx']).unique()
month_len_sf['month_len'] = month_len_sf.apply(lambda x: monthrange(x['year'], x['month'])[1])
month_len_sf_dict = (month_len_sf.select_columns(['month_idx','month_len']).unique()
                    .to_dataframe().set_index('month_idx').to_dict(orient='dict')['month_len'])

def find_top1_dist_over_prop_by_count(x, prop):
    """
    x is each row of user_monthly_district_list.
    column ['monthly_dist_count_list']: {dist_ID: Dist_monthly_count}
    Similar to the segment-based method that, only keep those segments that appear >= prop*len(segment),
    here we only keep the district that appear >= prop*len(month) as the top1 district. 
    If none of the districts satisfy this rule, then no top1 district will be returned for this month.
    """
    dist_dict = x['distinct_date_count_list']
    month = x['month_idx']
    month_len = month_len_sf_dict[month]
    top1_count = max(dist_dict.values())
    if top1_count >= prop*month_len:
        top1_dist = [k for k,v in dist_dict.iteritems() if v == top1_count]
        top1_dist = top1_dist[0]
    else:
        top1_dist = None
    return top1_dist


def method2_monthly_loc_over_prop(x, prop):
    """
    Based on method2, add a rule that the user must apppear at the home location over (prop * num of days in that month).
    """
    sel_user_monthly_m22_1 = (x.groupby(['user_id', 'month_idx', 'Dist_ID'], 
                       {'distinct_date_count': gl.aggregate.COUNT_DISTINCT('date')}))
    sel_user_monthly_m22_2 = (sel_user_monthly_m22_1.groupby(['user_id', 'month_idx'], 
                        {'distinct_date_count_list': gl.aggregate.CONCAT('Dist_ID', 'distinct_date_count')}))
    sel_user_monthly_m22_2['home_loc_prop'] = (sel_user_monthly_m22_2
                                     .apply(lambda x: find_top1_dist_over_prop_by_count(x, prop)))
    # drop those monthly records without monthly district(over prop)
    sel_user_monthly_m22_3 = sel_user_monthly_m22_2.dropna('home_loc_prop')
    method2_data3 = (sel_user_monthly_m22_3.groupby(['user_id'],
                                         {'monthly_home': gl.aggregate.CONCAT('month_idx', 'home_loc_prop')}))
    method2_data3['freq_mig_result'] = method2_data3['monthly_home'].apply(lambda x: find_migration(x))
    freq_stack = (method2_data3.stack('freq_mig_result', new_column_name=['month_idx', 'home_des'])
              .dropna().unpack('home_des'))
    freq_stack.rename({'home_des.0':'home', 'home_des.1':'destination'})
    freq_stack['home'] = freq_stack['home'].apply(lambda x: int(x))
    freq_stack['destination'] = freq_stack['destination'].apply(lambda x: int(x))
    freq_stack = freq_stack.select_columns(['user_id','month_idx','home','destination'])
    
    return(freq_stack)


## Method 3:
# Most phone activities were recorded during 7 p.m. and 9 a.m. (time constraints criterion)

def method3_monthly_loc(x):
    """
    Infer monthly location as where most phone activities were recorded during 7 p.m. and 9 a.m. 
    (time constraints criterion)
    
    x should be a SFrame showing users' hourly district location at night(7pm to 9am), 
    with a month_idx column indicating the month index for each date.
    """
    sel_user_monthly_m3 = (x.groupby(['user_id', 'month_idx', 'Dist_ID'], 
                       {'monthly_dist_count_night': gl.aggregate.COUNT('user_id')}))
    sel_user_monthly_m3_2 = (sel_user_monthly_m3.groupby(['user_id', 'month_idx'], 
                        {'dist_count_night_list': gl.aggregate.CONCAT('Dist_ID', 'monthly_dist_count_night')}))
    
    sel_user_monthly_m3_2['home_loc'] = (sel_user_monthly_m3_2['dist_count_night_list']
                                     .apply(lambda x: find_top1_loc_by_count(x)))
    method3_data = (sel_user_monthly_m3_2.groupby(['user_id'],
                                         {'monthly_home': gl.aggregate.CONCAT('month_idx', 'home_loc')}))
    method3_data['freq_mig_result'] = method3_data['monthly_home'].apply(lambda x: find_migration(x))
    freq_stack = (method3_data.stack('freq_mig_result', new_column_name=['month_idx', 'home_des'])
              .dropna().unpack('home_des'))
    freq_stack.rename({'home_des.0':'home', 'home_des.1':'destination'})
    freq_stack['home'] = freq_stack['home'].apply(lambda x: int(x))
    freq_stack['destination'] = freq_stack['destination'].apply(lambda x: int(x))
    freq_stack = freq_stack.select_columns(['user_id','month_idx','home','destination'])
    
    return(freq_stack)


## Method 4:
# Most phone activities were recorded, implementing a spatial perimeter of 1,000 meters around a cell tower that aggregates all activities within (space constraints criterion) 

def method4_monthly_loc(x):
    """
    Infer monthly location as where most phone activities were recorded, 
    implementing a spatial perimeter of 1,000 meters around a cell tower that aggregates all activities within. 
    (space constraints criterion)
    
    x should be a SFrame showing users' hourly tower location, 
    with a month_idx column indicating the month index for each date.
    """
    sel_user_hourly_dist_stack = x.stack('nearby_tower', new_column_name='cell_tower2')
    sel_user_hourly_dist_tower2 = (sel_user_hourly_dist_stack
                               .select_columns(['user_id','date','hour', 'cell_tower2','month_idx','Dist_ID']))
    sel_user_hourly_dist_tower2.rename({'cell_tower2':'cell_tower'})
    sel_user_hourly_dist_tower2 = sel_user_hourly_dist_tower2.dropna('cell_tower')
    sel_user_hourly_dist_tower2['cell_tower'] = sel_user_hourly_dist_tower2['cell_tower'].apply(lambda x: int(x))
    sel_user_hourly_dist_added_ntower = (x
                               .select_columns(['user_id','date','hour', 'cell_tower','month_idx','Dist_ID'])
                               .append(sel_user_hourly_dist_tower2))
    sel_user_monthly_m4 = (sel_user_hourly_dist_added_ntower.groupby(['user_id', 'month_idx', 'cell_tower'], 
                       {'monthly_tower_count': gl.aggregate.COUNT('user_id')}))
    sel_user_monthly_m4_2 = (sel_user_monthly_m4.groupby(['user_id', 'month_idx'], 
                        {'monthly_tower_count_list': gl.aggregate.CONCAT('cell_tower', 'monthly_tower_count')}))
    sel_user_monthly_m4_2['home_tower'] = (sel_user_monthly_m4_2['monthly_tower_count_list']
                                     .apply(lambda x: find_top1_loc_by_count(x)))
    
    sel_user_monthly_m4_3 = (sel_user_monthly_m4_2.join(tower_district.select_columns(['Dist_ID','SITEID']), 
                                                    on = {'home_tower':'SITEID'}, how = 'inner'))
    sel_user_monthly_m4_3.rename({'Dist_ID':'home_loc'})
    method4_data = (sel_user_monthly_m4_3.groupby(['user_id'],
                                             {'monthly_home': gl.aggregate.CONCAT('month_idx', 'home_loc')}))
    method4_data['freq_mig_result'] = method4_data['monthly_home'].apply(lambda x: find_migration(x))
    freq_stack = (method4_data.stack('freq_mig_result', new_column_name=['month_idx', 'home_des'])
              .dropna().unpack('home_des'))
    freq_stack.rename({'home_des.0':'home', 'home_des.1':'destination'})
    freq_stack['home'] = freq_stack['home'].apply(lambda x: int(x))
    freq_stack['destination'] = freq_stack['destination'].apply(lambda x: int(x))
    freq_stack = freq_stack.select_columns(['user_id','month_idx','home','destination'])
    
    return(freq_stack)


## Method 5:
# The combination of 3) and 4), thus most phone activities recorded during 7 p.m. and 9 a.m. and implementing a spatial perimeter of 1,000 meter (time constraints and space constraint criterion)

def method5_monthly_loc(x):
    """
    Infer monthly location as where most phone activities were recorded, 
    implementing a spatial perimeter of 1,000 meters around a cell tower that aggregates all activities within. 
    (space constraints criterion)
    
    x should be a SFrame showing users' hourly tower location at night(7pm to 9am), 
    with a month_idx column indicating the month index for each date.
    """
    sel_user_hourly_dist_stack_night = x.stack('nearby_tower', new_column_name='cell_tower2')
    sel_user_hourly_dist_tower2_night = (sel_user_hourly_dist_stack_night
                                   .select_columns(['user_id','date','hour', 'cell_tower2','month_idx','Dist_ID']))
    sel_user_hourly_dist_tower2_night.rename({'cell_tower2':'cell_tower'})
    sel_user_hourly_dist_tower2_night = sel_user_hourly_dist_tower2_night.dropna('cell_tower')
    sel_user_hourly_dist_tower2_night['cell_tower'] = sel_user_hourly_dist_tower2_night['cell_tower'].apply(lambda x: int(x))
    
    sel_user_hourly_dist_added_ntower_night = (x
                               .select_columns(['user_id','date','hour', 'cell_tower','month_idx','Dist_ID'])
                               .append(sel_user_hourly_dist_tower2_night))
    
    sel_user_monthly_m5 = (sel_user_hourly_dist_added_ntower_night.groupby(['user_id', 'month_idx', 'cell_tower'], 
                       {'monthly_tower_count_night': gl.aggregate.COUNT('user_id')}))
    sel_user_monthly_m5_2 = (sel_user_monthly_m5.groupby(['user_id', 'month_idx'], 
                            {'monthly_tower_count_night_list': gl.aggregate.CONCAT('cell_tower', 'monthly_tower_count_night')}))
    sel_user_monthly_m5_2['home_tower'] = (sel_user_monthly_m5_2['monthly_tower_count_night_list']
                                         .apply(lambda x: find_top1_loc_by_count(x)))

    sel_user_monthly_m5_3 = (sel_user_monthly_m5_2.join(tower_district.select_columns(['Dist_ID','SITEID']), 
                                                    on = {'home_tower':'SITEID'}, how = 'inner'))

    sel_user_monthly_m5_3.rename({'Dist_ID':'home_loc'})
    method5_data = (sel_user_monthly_m5_3.groupby(['user_id'],
                                         {'monthly_home': gl.aggregate.CONCAT('month_idx', 'home_loc')}))
    method5_data['freq_mig_result'] = method5_data['monthly_home'].apply(lambda x: find_migration(x))
    freq_stack = (method5_data.stack('freq_mig_result', new_column_name=['month_idx', 'home_des'])
              .dropna().unpack('home_des'))
    freq_stack.rename({'home_des.0':'home', 'home_des.1':'destination'})
    freq_stack['home'] = freq_stack['home'].apply(lambda x: int(x))
    freq_stack['destination'] = freq_stack['destination'].apply(lambda x: int(x))
    freq_stack = freq_stack.select_columns(['user_id','month_idx','home','destination'])
    
    return(freq_stack)

## Method 6:
# Identifies locations where the individual spends the most time. First identifies the hourly modal location by computing the most frequently visited district in every hour of the entire dataset, then aggregates hourly modal locations to find the daily modal location, and finally identifies the monthly modal location by taking the mode over the daily modal locations.

def assign_midnight_to_previous_day(x):
    hour = x['hour']
    new_date = str(x['date'])
    if hour <= 8:
        new_date = datetime.strptime(new_date, '%Y%m%d')- timedelta(days = 1)
        new_date = new_date.strftime("%Y%m%d")
    result_date = new_date
    return(result_date)

def find_top1_dist_by_count_with_tie(x):
    dist_dict = x
    top1_count = max(dist_dict.values())
    top1_dist_list = [int(k) for k,v in dist_dict.iteritems() if v == top1_count]
    # top1_dist = top1_dist[0]
    return top1_dist_list

def get_one_dist_hour_and_day(x, daily_dist_dict, monthly_dist_dict):
    user_id = x['user_id']
    date = x['new_date']
    month = x['new_month_idx']
    top1_dist_list = x['top1_dist_list']
    if len(top1_dist_list) == 1:
        top1_dist = int(top1_dist_list[0])
    else:
        # first use daily all modal
        daily_dist_dict_ = daily_dist_dict[user_id][date]
        hourly_top1_dist_daily_count_dict = {k:v for k,v in daily_dist_dict_.iteritems() if k in top1_dist_list}
        h_daily_top1_dist_list = find_top1_dist_by_count_with_tie(hourly_top1_dist_daily_count_dict)
        if len(h_daily_top1_dist_list) == 1:
            top1_dist = h_daily_top1_dist_list[0]
        else:
            # if daily all modal cannot solve, then use month all modal
            monthly_dist_dict_ = monthly_dist_dict[user_id][month]
            # only look for the top1 tied districts when use daily all modal
            hourly_top1_dist_monthly_count_dict = {k:v for k,v in monthly_dist_dict_.iteritems() if k in h_daily_top1_dist_list}
            h_d_monthly_top1_dist_list = find_top1_dist_by_count_with_tie(hourly_top1_dist_monthly_count_dict)
            if len(h_d_monthly_top1_dist_list) == 1:
                top1_dist = h_d_monthly_top1_dist_list[0]
            else:
                top1_dist = None        
                # may output None value of top1 dist if monthly modal could not resolve
    return(top1_dist)

def get_one_dist_month(x, monthly_dist_dict):
    user_id = x['user_id']
    month = x['new_month_idx']
    top1_dist_list = x['top1_dist_list']
    if len(top1_dist_list) == 1:
        top1_dist = int(top1_dist_list[0])
    else:
        # use month all modal
        monthly_dist_dict_ = monthly_dist_dict[user_id][month]
        # only look for the top1 tied districts when use daily all modal
        monthly_top1_dist_monthly_count_dict = {k:v for k,v in monthly_dist_dict_.iteritems() if k in top1_dist_list}
        h_d_monthly_top1_dist_list = find_top1_dist_by_count_with_tie(monthly_top1_dist_monthly_count_dict)
        if len(h_d_monthly_top1_dist_list) == 1:
            top1_dist = h_d_monthly_top1_dist_list[0]
        else:
            top1_dist = None        
            # may output None value of top1 dist if monthly modal could not resolve
    return(top1_dist)

def method6_monthly_loc(x):
    """
    Infer monthly location where the individual spends the most time. 
    First identifies the hourly modal location by computing the most frequently visited district in every hour of the entire dataset, 
    then aggregates hourly modal locations to find the daily modal location, 
    and finally identifies the monthly modal location by taking the mode over the daily modal locations.
    
    x should be a SFrame showing users' hourly tower/district location,
    with a month_idx column indicating the month index for each date.
    """
    # method 6: h >= 18 or h <=7:
    sel_user_hourly_dist_m6 = x.filter_by(range(0,8)+range(18,25),'hour')
    sel_user_hourly_dist_m6['new_date'] = sel_user_hourly_dist_m6.apply(lambda x: assign_midnight_to_previous_day(x))
    sel_user_hourly_dist_m6['new_month_idx'] = sel_user_hourly_dist_m6['new_date'].apply(lambda x: month_index(x))
    
    # get all daily district not based on hour. this is used to choose hour tie
    daily_dist_sf = sel_user_hourly_dist_m6.groupby(['user_id','new_date','Dist_ID'],
                                 {'daily_dist_count': gl.aggregate.COUNT('Dist_ID')})
    daily_dist_sf2 = (daily_dist_sf.groupby(['user_id','new_date'], 
                                            {'daily_dist_dict': gl.aggregate.CONCAT('Dist_ID','daily_dist_count')}))
    daily_dist_sf3 = (daily_dist_sf2.groupby(['user_id'], 
                                             {'daily_dist_count_dict': gl.aggregate.CONCAT('new_date','daily_dist_dict')}))
    daily_dist_dict = daily_dist_sf3.to_dataframe().set_index('user_id').to_dict(orient='dict')['daily_dist_count_dict']
    
    
    # get all monthly district not based on day.
    monthly_dist_sf = sel_user_hourly_dist_m6.groupby(['user_id','new_month_idx','Dist_ID'],
                                 {'monthly_dist_count': gl.aggregate.COUNT('Dist_ID')})
    monthly_dist_sf2 = (monthly_dist_sf.groupby(['user_id','new_month_idx'], 
                                            {'monthly_dist_dict': gl.aggregate.CONCAT('Dist_ID','monthly_dist_count')}))

    monthly_dist_sf3 = (monthly_dist_sf2.groupby(['user_id'], 
                                {'monthly_dist_count_dict': gl.aggregate.CONCAT('new_month_idx','monthly_dist_dict')}))
    monthly_dist_dict = monthly_dist_sf3.to_dataframe().set_index('user_id').to_dict(orient='dict')['monthly_dist_count_dict']

    # step 2: prepare hourly data to calculate hourly modal district
    hourly_dist_sf = sel_user_hourly_dist_m6.groupby(['user_id','new_month_idx','new_date','hour','Dist_ID'],
                                 {'hourly_dist_count': gl.aggregate.COUNT('Dist_ID')})
    hourly_dist_sf2 = (hourly_dist_sf.groupby(['user_id','new_month_idx','new_date','hour'],
                                 {'hourly_dist_dict': gl.aggregate.CONCAT('Dist_ID','hourly_dist_count')}))
    hourly_dist_sf2['top1_dist_list'] = (hourly_dist_sf2['hourly_dist_dict']
                                         .apply(lambda x: find_top1_dist_by_count_with_tie(x)))
    # find one dist hour
    hourly_dist_sf2['hourly_top1_dist'] = (hourly_dist_sf2.apply(lambda x: 
                                            get_one_dist_hour_and_day(x, daily_dist_dict, monthly_dist_dict)))
    
    # step 3: prepare daily data to calculate daily modal district
    daily_htop1_dist_sf = (hourly_dist_sf2.dropna().groupby(['user_id', 'new_month_idx','new_date','hourly_top1_dist'],
                       {'daily_htop1_dist_count': gl.aggregate.COUNT('hour')}))
    daily_htop1_dist_sf2 = (daily_htop1_dist_sf.groupby(['user_id', 'new_month_idx','new_date'],
                            {'daily_htop1_dist_dict':gl.aggregate.CONCAT('hourly_top1_dist','daily_htop1_dist_count')}))
    
    daily_htop1_dist_sf2['top1_dist_list'] = (daily_htop1_dist_sf2['daily_htop1_dist_dict']
                                              .apply(lambda x: find_top1_dist_by_count_with_tie(x)))
    
    # find one dist daily
    daily_htop1_dist_sf2['daily_top1_dist'] = (daily_htop1_dist_sf2.apply(lambda x: 
                                            get_one_dist_hour_and_day(x, daily_dist_dict, monthly_dist_dict)))
    
    # step 4: prepare daily data to calculate monthly modal district
    # drop na hourly_top1_dist before groupby
    monthly_dtop1_dist_sf = (daily_htop1_dist_sf2.dropna().groupby(['user_id', 'new_month_idx','daily_top1_dist'],
                           {'monthly_dtop1_dist_count': gl.aggregate.COUNT('new_date')}))
    monthly_dtop1_dist_sf2 = (monthly_dtop1_dist_sf.groupby(['user_id', 'new_month_idx'],
                            {'monthly_dtop1_dist_dict':gl.aggregate.CONCAT('daily_top1_dist','monthly_dtop1_dist_count')}))
    monthly_dtop1_dist_sf2['top1_dist_list'] = (monthly_dtop1_dist_sf2['monthly_dtop1_dist_dict']
                                             .apply(lambda x: find_top1_dist_by_count_with_tie(x)))
    # find one dist monthly
    monthly_dtop1_dist_sf2['monthly_top1_dist'] = (monthly_dtop1_dist_sf2.apply(lambda x: 
                                                    get_one_dist_month(x, monthly_dist_dict)))
    monthly_dtop1_dist_sf3 = monthly_dtop1_dist_sf2.dropna()
    
    # find migration by monthly modal district
    method6_data = (monthly_dtop1_dist_sf3.groupby(['user_id'],
                                         {'monthly_home': gl.aggregate.CONCAT('new_month_idx', 'monthly_top1_dist')}))
    method6_data['freq_mig_result'] = method6_data['monthly_home'].apply(lambda x: find_migration(x))
    freq_stack = (method6_data.stack('freq_mig_result', new_column_name=['month_idx', 'home_des'])
              .dropna().unpack('home_des'))
    freq_stack.rename({'home_des.0':'home', 'home_des.1':'destination'})
    freq_stack['home'] = freq_stack['home'].apply(lambda x: int(x))
    freq_stack['destination'] = freq_stack['destination'].apply(lambda x: int(x))
    freq_stack = freq_stack.select_columns(['user_id','month_idx','home','destination'])
    
    return(freq_stack)



method1_data = method1_monthly_loc(user_hourly_tower_dist)
method2_data = method2_monthly_loc(user_hourly_tower_dist)
method2_data_over_prop = method2_monthly_loc_over_prop(user_hourly_tower_dist, 0.3)
method3_data = method3_monthly_loc(user_hourly_tower_dist_night)
method4_data = method4_monthly_loc(user_hourly_tower_dist)
method5_data = method5_monthly_loc(user_hourly_tower_dist_night)
method6_data = method6_monthly_loc(user_hourly_tower_dist)

# If input data1 user_hourly_tower_dist does not contain the 'nearby_tower' column, 
# could use the following codes to find the nearby 1km tower and add the list to the sframe as a new column.
# (The radius to find towers nearby could be adjusted.) 

tower_district = gl.SFrame.read_csv('./sample_tower_district.csv', verbose=False)
tower_coord = tower_district.select_columns(['Dist_ID','SITEID','LONG','LAT'])
tower_coord['coordinate'] = tower_coord.apply(lambda x: (x['LONG'],x['LAT']))
tower_coord_dict = (tower_coord.select_columns(['SITEID','coordinate']).unique()
                    .to_dataframe().set_index('SITEID').to_dict(orient='dict')['coordinate'])

all_tower = np.array(tower_coord_dict.keys())
all_coord = np.array(tower_coord_dict.values())
lon = np.radians(np.array(zip(*all_coord)[0]))
lat = np.radians(np.array(zip(*all_coord)[1]))
R = 6373.0

def find_tower_nearby(x, raduis):
    """
    x: each row in tower_district
    raduis: km
    R = 6373.0
    """
    
    dlon = lon - np.radians(x['LONG'])
    dlat = lat - np.radians(x['LAT'])
        
    a = np.sin(dlat / 2)**2 + np.cos(lat) * np.cos(np.radians(x['LAT'])) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    #distance2 = np.array([geopy.distance.vincenty((coords_1[1],coords_1[0]), (x['LAT'],x['LONG'])).km 
    #                      for coords_1 in all_coord])
    
    nearby_tower = all_tower[np.where(distance<=raduis)]
    nearby_tower = list(set(nearby_tower) - set([x['SITEID']]))
    
    return nearby_tower

# find nearby tower
tower_coord['nearby_tower'] = tower_coord.apply(lambda x: find_tower_nearby(x, 1))
tower_dist_w_nearby = tower_coord.select_columns(['Dist_ID','SITEID','nearby_tower'])

# from tower to district, add a new column of nearby_tower
user_hourly_tower_dist2 = user_hourly_tower_dist.join(tower_dist_w_nearby, on={'cell_tower':'SITEID'}) 

