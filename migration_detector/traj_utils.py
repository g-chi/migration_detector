import pandas as pd
import numpy as np
from array import array


def fill_missing_day(all_loc_rec, k):
    """
    For any location (L) in any day, see the next k days.
    If in these k days, there is >= 1 day when this person appears in L,
    then fill the gaps with L.
    """
    result_dict = {}
    for loc in all_loc_rec.keys():
        loc_date = all_loc_rec[loc]
        loc_date.sort()
        new_loc_date = list(loc_date)
        if len(loc_date) > 1:
            for i, date in enumerate(loc_date[:-1]):
                close_date = loc_date[i+1]
                date_diff = close_date - date
                if 1 < date_diff <= k:
                    new_loc_date += range(date+1, close_date)
        new_loc_date.sort()
        result_dict[loc] = new_loc_date
    return result_dict


def find_segment(all_fill_rec, k):
    """
    Group consecutive days in the same location together into segments.
    A segment need to have consecutive days >= k days. (default k = 30)
    """
    result_dict = {}
    for loc in all_fill_rec.keys():
        loc_date = all_fill_rec[loc]
        if len(loc_date) >= k:
            loc_date_next = loc_date[1:]
            loc_date_next.append(0)
            diff_next = np.array(loc_date_next) - np.array(loc_date)

            # the index of date that is not consecutive
            # i.e., the next element is not the next date for today
            # but today could be the next date for the former element
            split_idx = np.where(diff_next != 1)[0]
            segment_idx = []

            if len(split_idx) == 1:
                # the last element in diff_next must <0
                # so all the date in loc_date are consective
                segment_idx.append([loc_date[0], loc_date[-1]])
            else:
                # the days before split_idx[0]
                if split_idx[0] >= k-1:
                    segment_idx.append([loc_date[0], loc_date[split_idx[0]]])

                for i, index in enumerate(split_idx[:-1]):
                    next_idx = split_idx[i+1]
                    # Find the start and end index of each period(consecutive days)
                    # For each period, the start and end index are defined by two
                    # nearby elements in split_idx(a list),
                    # and the two elements, which are also indexes in loc_date
                    # (the date list of loc L),
                    # should have a difference >=k,
                    # because we want the period have at least k day
                    # so in loc_date, a period must statisfy:
                    # [split_idx1, start_idx,...(k-2 days), end_idx(split_idx2)]
                    if next_idx - index >= k:
                        seg_start_idx = index + 1
                        seg_end_idx = next_idx
                        seg_start_date = loc_date[seg_start_idx]
                        seg_end_date = loc_date[seg_end_idx]
                        segment_idx.append([seg_start_date, seg_end_date])

            if len(segment_idx) > 0:
                result_dict[loc] = segment_idx

    return result_dict


def filter_seg_appear_prop(x, filter_column, prop):
    """
    Only keep those segments that appear >= prop*len(segment).
    Note: appeared days are original days before filling.
    Do it for the filled record.
    """
    all_record = x['all_record']
    to_filter_record = x[filter_column]

    result_dict = {}
    for location in to_filter_record.keys():
        loc_daily_record = all_record[location]
        loc_segment = to_filter_record[location]
        new_loc_record = []
        for segment in loc_segment:
            segment_len = segment[1] - segment[0] + 1
            appear_len = len([day for day in loc_daily_record
                              if day <= segment[1] and day >= segment[0]])
            if appear_len >= prop * segment_len:
                new_loc_record.append(segment)
        if len(new_loc_record) > 0:
            result_dict[location] = new_loc_record

    return result_dict


def join_segment_if_no_gap(old_segment):
    """
    In the same location, join continuous segments together,
    if there are no other segments within in gap period in other location.
    If there is only one location for this user,
    it means that he/she is not a migrant, return {}.
    """
    loc_list = old_segment.keys()
    new_segment = {}

    if len(loc_list) > 1:
        for location in loc_list:
            loc_record = old_segment[location]
            new_loc_record = []
            if len(loc_record) == 1:
                new_loc_record.append(loc_record[0])
            elif len(loc_record) > 1:
                current_segment = loc_record[0]
                other_loc_record = [value
                                    for key, value in old_segment.iteritems()
                                    if key != location]
                other_loc_record = [j for l in other_loc_record for j in l]
                other_loc_exist_days = [range(int(x[0]), int(x[1]) + 1)
                                        for x in other_loc_record]
                # merge sublist and unique(), set() will also sort the list automatically
                other_loc_exist_days = set([j for l in other_loc_exist_days
                                            for j in l])

                for i in range(1, len(loc_record)):
                    next_segment = loc_record[i]
                    # Find gap
                    gap = [current_segment[1] + 1, next_segment[0] - 1]
                    gap_days_list = range(int(gap[0]), int(gap[1]) + 1)
                    # Check if there are any other segments cover the gap
                    gap_other_loc_interset = set(gap_days_list) & other_loc_exist_days
                    if len(gap_other_loc_interset) == 0:
                        current_segment = [current_segment[0], next_segment[1]]
                    else:
                        new_loc_record.append(current_segment)
                        current_segment = next_segment
                new_loc_record.append(current_segment)
            new_segment[location] = new_loc_record

    return new_segment


def change_overlap_segment(x, filter_segment_col, k, d):
    """
    Check if there is any overlap period longer than k days between segments.
    If there are, remove the overlapped part rather than remove the whole segment.
    After removing the overlapped part, only keep the segments that are longer than d days.
    Return the changed segments that satisfy the rule.
    """
    segments = x[filter_segment_col]
    loc_list = segments.keys()
    remove_overlap_date_dict = {}
    change_result_dict = {}
    # Filter out any segment by k
    for location in loc_list:
        other_loc = list(set(loc_list) - set([location]))
        current_loc_changed_date = []
        for current_segment in segments[location]:
            other_segment_list = ([value for key, value in segments.iteritems()
                                   if key in other_loc])
            other_segment_list = [j for l in other_segment_list for j in l]
            current_segment_date = set(range(int(current_segment[0]), int(current_segment[1]) + 1))
            for other_segment in other_segment_list:
                other_segment_date = range(int(other_segment[0]), int(other_segment[1]) + 1)
                seg_intersect = set(current_segment_date) & set(other_segment_date)
                if len(seg_intersect) > k:
                    current_segment_date = current_segment_date - seg_intersect
            current_loc_changed_date += list(current_segment_date)
        if len(current_loc_changed_date) > 0:
            remove_overlap_date_dict[location] = current_loc_changed_date

    change_result_dict = find_segment(remove_overlap_date_dict, d)

    return change_result_dict


def output_segments(user_loc_agg, segment_file, min_overlap_part_len, index2date):
    user_loc_agg_tmp = user_loc_agg.copy()
    user_loc_agg_tmp['long_seg'] = user_loc_agg_tmp.apply(
        lambda x: change_overlap_segment(
            x,
            'medium_segment',
            min_overlap_part_len,
            30
        )
    )
    user_loc_agg_tmp['long_seg_num'] = user_loc_agg_tmp['long_seg'].apply(lambda x: len(x))
    user_seg = user_loc_agg_tmp[user_loc_agg_tmp['long_seg_num'] > 0]
    user_seg_migr = user_seg.stack(
        'long_seg',
        new_column_name=['location', 'migration_list']
    )
    user_seg_migr = user_seg_migr.stack(
        'migration_list',
        new_column_name='segment'
    )
    user_seg_migr['segment_start'] = user_seg_migr['segment'].apply(
        lambda x: x[0]
    )
    user_seg_migr['segment_end'] = user_seg_migr['segment'].apply(
        lambda x: x[1]
    )
    user_seg_migr['segment_start_date'] = user_seg_migr.apply(
        lambda x: index2date[x['segment_start']]
    )
    user_seg_migr['segment_end_date'] = user_seg_migr.apply(
        lambda x: index2date[x['segment_end']]
    )
    user_seg_migr['segment_length'] = (user_seg_migr['segment_end'] -
                                       user_seg_migr['segment_start'])
    user_seg_migr.select_columns(
        ['user_id', 'location',
         'segment_start', 'segment_end', 'segment_length',
         'segment_start_date', 'segment_end_date']
    ).export_csv(segment_file)


def find_migration_by_segment(segments, k):
    """
    Filter out migration that the time period of home covers more than k days
    of the time period of destination.
    Return home_segment, destination_segment.
    """
    all_segments_date = [j for l in segments.values() for j in l]
    all_segments_date.sort()
    segment_loc_sort = []
    migration_result = []

    for current_segment in all_segments_date:
        # transform the current_segment into array('d',current_segment)
        # to match with the dict
        # depends on the format of data
        current_segment_transform = array('d', current_segment)
        current_segment_loc_list = [key for key, value in segments.iteritems()
                                    if current_segment_transform in value]
        segment_loc_sort.append(current_segment_loc_list)
    segment_loc_sort = np.array([j for l in segment_loc_sort for j in l])

    for index, current_segment in enumerate(all_segments_date[:-1]):
        # Find first different location segment
        current_loc = segment_loc_sort[index]
        other_loc_segment_idx = np.where(segment_loc_sort != current_loc)[0]

        next_segment_idx_list = filter(lambda x: x > index, other_loc_segment_idx)
        next_segment_idx_list.sort()
        if len(next_segment_idx_list) > 0:
            for i in range(len(next_segment_idx_list)):
                next_segment_idx = next_segment_idx_list[i]
                next_segment = all_segments_date[next_segment_idx]
                if next_segment[0] - current_segment[1] >= -k+1:
                    home_segment = current_segment
                    des_segment = next_segment
                    home_id = current_loc
                    des_id = segment_loc_sort[next_segment_idx]
                    migration_result.append([home_segment, des_segment,
                                             home_id, des_id])
                    break

    return migration_result


def create_migration_dict(x):
    """
    Transform migration information into a dictionary for plotting.
    """
    home_id = x[2]
    des_id = x[3]
    home_segment = x[0]
    des_segment = x[1]
    return {home_id: [home_segment], des_id: [des_segment]}


# Functions to infer migration day
def find_migration_day_segment(x):
    """
    Return migration day and minimum number of error days.
    """
    home = x['home']
    des = x['destination']
    home_seg = x['migration_segment'][home][0]
    des_seg = x['migration_segment'][des][0]
    home_end = home_seg[1]
    des_start = des_seg[0]

    # Find the day of minimun error day
    home_record_bwn = [d for d in x['all_record'][home]
                       if home_end <= d <= des_start]
    des_record_bwn = [d for d in x['all_record'][des]
                      if home_end <= d <= des_start]
    poss_m_day = range(int(home_end), int(des_start)+1)
    wrong_day_before = [[des_day
                         for des_day in des_record_bwn
                         if des_day < current_day]
                         for current_day in poss_m_day]
    num_wrong_day_before = [len(des_day_list)
                            for des_day_list in wrong_day_before]
    wrong_day_after = [[home_day
                        for home_day in home_record_bwn
                        if home_day > current_day]
                        for current_day in poss_m_day]
    num_wrong_day_after = [len(home_day_list)
                           for home_day_list in wrong_day_after]
    num_error_day = np.array(num_wrong_day_before) + np.array(num_wrong_day_after)
    min_error_idx = np.where(num_error_day == num_error_day.min())[0][-1]

    # If there are several days that the num_error_day is the minimun,
    # take the last one.
    migration_day = poss_m_day[min_error_idx]
    min_error_day = min(num_error_day)

    return [int(migration_day), int(min_error_day)]


# Functions for short term migration
def filter_migration_segment_len(x, hmin=0, hmax=float("inf"), dmin=0,
                                 dmax=float("inf")):
    """
    Filter migration segment by home segment length and destination length.
    x: ['migration_list']
    hmin: min_home_segment_len
    hmax: max_home_segment_len
    dmin: min_des_segment_len
    dmax: max_des_segment_len
    """
    home_segment = x[0]
    des_segment = x[1]
    home_len = home_segment[1]-home_segment[0]+1
    des_len = des_segment[1]-des_segment[0]+1
    if (hmin <= home_len <= hmax) and (dmin <= des_len <= dmax):
        return 1
    else:
        return 0
