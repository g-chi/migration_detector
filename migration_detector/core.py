from __future__ import division
import pandas as pd
import numpy as np
import graphlab as gl
import os
import copy
from array import array
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
from traj_utils import *


class TrajRecord():
    # Input data: user_id, date(int: YYYYMMDD), location(int)
    # Output data:
    #     user_id, migration_date, home, destination, home_start, home_end,
    #     destiantion_start, destination_end,
    #     uncertainty, num_error_day
    def __init__(self, user_traj, raw_traj, index2date, date_num_long):
        """
        Attributes
        ----------
        user_traj : gl.dataframe
            Trajector of users after aggregation
        raw_traj : gl.dataframe
            Raw dataset of users' trajectory
        index2date: dict
            Convert from date index to real date
        date_num_long : gl.SFrame
            Date and num: 'date', 'date_num'
        """
        self.user_traj = user_traj
        self.raw_traj = raw_traj
        self.index2date = index2date
        self.date_num_long = date_num_long

    def plot_trajectory(self, user_id, start_date=None, end_date=None, if_save=True, fig_path='figure'):
        """
        Plot an individual's trajectory.

        Attributes
        ----------
        user_id : string
            user id
        start_date : str
            start date of the figure in the format of 'YYYYMMDD'
        end_date : str
            end date of the figure in the format of 'YYYYMMDD'
        if_save : boolean
            if save the figure
        fig_path : str
            the path to save figures
        """
        date_min = self.raw_traj.filter_by(user_id, 'user_id')['date'].min()
        date_max = self.raw_traj.filter_by(user_id, 'user_id')['date'].max()
        if start_date:
            assert int(start_date) >= date_min, "start date must be later than the first day of this user's records, which is " + str(date_min)
            start_day = self.date_num_long.filter_by(int(start_date), 'date')['date_num'][0]
        else:
            start_day = self.date_num_long.filter_by(date_min, 'date')['date_num'][0]
            start_date = str(self.date_num_long.filter_by(start_day, 'date_num')['date'][0])
        if end_date:
            assert int(end_date) <= date_max, "end date must be earlier than the last day of this user's records, which is " + str(date_max)
            end_day = self.date_num_long.filter_by(int(end_date), 'date')['date_num'][0]
        else:
            end_day = self.date_num_long.filter_by(date_max, 'date')['date_num'][0]
            end_date = str(self.date_num_long.filter_by(end_day, 'date_num')['date'][0])
        fig, ax, _, _ = plot_traj_common(self.raw_traj, user_id, start_day, end_day, self.date_num_long)
        if not os.path.isdir(fig_path):
            os.makedirs(fig_path)
        save_path = os.path.join(fig_path, user_id  + '_' + start_date + '-' + end_date + '_trajectory')
        if if_save:
            fig.savefig(save_path, bbox_inches="tight")

    def find_migrants(self, num_stayed_days_migrant=90, num_days_missing_gap=7,
                      small_seg_len=30, seg_prop=0.6, min_overlap_part_len=0,
                      max_gap_home_des=30):
        """
        Find migrants step by step

        - step 1: Fill the small missing gaps
                  fill_missing_day('all_record', num_days_missing_gap) -> 'filled_record'

        - step 2: Group consecutive days in the same location together into segments
                  and find segments over certain length.
                  find_segment('filled_record',small_seg_len) -> 'segment_dict'

        - step 3: Find segments in which the user appeared more than prop*len(segment)
                  number of days for that segment.
                  filter_seg_appear_prop(x, 'segment_dict',seg_prop)
                  -> 'segment_over_prop'

        - step 4: Merge neighboring segments together if there are no segments
                  in other districts between the neighboring segments.
                  join_segment_if_no_gap('segment_over_prop') -> 'medium_segment'

        - step 5: Remove overlap parts between any segments who have overlapping
                  and keep segments >= num_stayed_days_migrant days.
                  change_overlap_segment(x, 'medium_segment',
                  min_overlap_part_len, num_stayed_days_migrant) -> 'long_seg'

        - step 6: Find migration: home, destination
                  user_loc_agg['long_seg_num'] = user_loc_agg['long_seg'].apply(lambda x: len(x))
                  user_long_seg = user_loc_agg.filter_by([0,1],'long_seg_num',exclude=True)
                  find_migration_by_segment('long_seg',min_overlap_part_len) -> 'migration_result'

        - step 7: Find migration day
                  find_migration_day_segment(x)

        - step 8: Filter migration segment
                  a) The gap between home segment and destination segment <= 31 days.
                  'seg_diff' <= 31  -> seg_migr_filter
                  b) For short-term migration: Restriction on the length of home segment
                  and destination segment.
                  filter_migration_segment_len('migration_list', hmin, hmax, dmin, dmax)
                  -> 'flag_home_des_len' (0 or 1)

        Attributes
        ----------
        num_stayed_days_migrant : int
            Number of stayed days in home/destination to consider as a migrant
        num_days_missing_gap : int
            Fill the small missing gaps (radius/epsilon)
        small_seg_len : int
            First threshold to filter small segment (minPts)
        seg_prop : float
            Only keep segments that appear >= prop*len(segment).
        min_overlap_part_len : int
            Overlap: 0 days
        max_gap_home_des : int
            Gaps beteen home segment and destination segment
        """
        self.user_traj['filled_record'] = self.user_traj['all_record'].apply(
            lambda x: fill_missing_day(x, num_days_missing_gap)
        )
        self.user_traj['segment_dict'] = self.user_traj['filled_record'].apply(
            lambda x: find_segment(x, small_seg_len)
        )
        self.user_traj['segment_over_prop'] = self.user_traj.apply(
            lambda x: filter_seg_appear_prop(x, 'segment_dict', seg_prop)
        )
        self.user_traj['medium_segment'] = self.user_traj['segment_over_prop'].apply(
            lambda x: join_segment_if_no_gap(x)
        )
        print('Start: Detecting migration')
        self.user_traj['long_seg'] = self.user_traj.apply(
            lambda x: change_overlap_segment(
                x,
                'medium_segment',
                min_overlap_part_len,
                num_stayed_days_migrant
            )
        )
        self.user_traj['long_seg_num'] = self.user_traj['long_seg'].apply(lambda x: len(x))
        self.user_traj['medium_segment_num'] = self.user_traj['medium_segment'].apply(lambda x: len(x))
        self.user_traj['segment_over_prop_num'] = self.user_traj['segment_over_prop'].apply(lambda x: len(x))

        # filter out those users with no record or only one location in ['long_seg]
        user_long_seg = self.user_traj.filter_by([0, 1], 'long_seg_num', exclude=True)
        user_long_seg['migration_result'] = user_long_seg['long_seg'].apply(
            lambda x: find_migration_by_segment(x, min_overlap_part_len)
        )
        migrant_size = [len(m) for m in user_long_seg['migration_result']]
        if len(migrant_size) == 0:
            print('No migrants are found.')
            return None
        user_seg_migr = user_long_seg.stack(
            'migration_result',
            new_column_name='migration_list'
        )
        user_seg_migr = user_seg_migr.dropna('migration_list')

        user_seg_migr['migration_segment'] = user_seg_migr['migration_list'].apply(
            lambda x: create_migration_dict(x)
        )
        user_seg_migr['home'] = user_seg_migr['migration_list'].apply(lambda x: x[2])
        user_seg_migr['destination'] = user_seg_migr['migration_list'].apply(
            lambda x: x[3]
        )

        user_seg_migr['migration_day_result'] = user_seg_migr.apply(
            lambda x: find_migration_day_segment(x)
        )

        user_seg_migrs = user_seg_migr.unpack('migration_day_result')
        user_seg_migrs.rename({'migration_day_result.0': 'migration_day',
                               'migration_day_result.1': 'num_error_day'})

        user_seg_migrs['migration_day'] = user_seg_migrs['migration_day'].apply(
            lambda x: int(x)
        )
        user_seg_migrs['migration_date'] = user_seg_migrs.apply(
            lambda x: self.index2date[x['migration_day']]
        )
        user_seg_migrs['home_start'] = user_seg_migrs['migration_list'].apply(
            lambda x: x[0][0]
        )
        user_seg_migrs['destination_end'] = user_seg_migrs['migration_list'].apply(
            lambda x: x[1][1]
        )
        user_seg_migrs['home_end'] = user_seg_migrs['migration_list'].apply(
            lambda x: x[0][1]
        )
        user_seg_migrs['destination_start'] = user_seg_migrs['migration_list'].apply(
            lambda x: x[1][0]
        )
        user_seg_migrs['home_start_date'] = user_seg_migrs.apply(
            lambda x: self.index2date[x['home_start']]
        )
        user_seg_migrs['home_end_date'] = user_seg_migrs.apply(
            lambda x: self.index2date[x['home_end']]
        )
        user_seg_migrs['destination_start_date'] = user_seg_migrs.apply(
            lambda x: self.index2date[x['destination_start']]
        )
        user_seg_migrs['destination_end_date'] = user_seg_migrs.apply(
            lambda x: self.index2date[x['destination_end']]
        )
        user_seg_migrs['seg_diff'] = (user_seg_migrs['destination_start'] -
                                      user_seg_migrs['home_end'])
        seg_migr_filter = user_seg_migrs[user_seg_migrs['seg_diff'] <= max_gap_home_des]
        seg_migr_filter['uncertainty'] = seg_migr_filter['seg_diff'] - 1
        print('Done')
        return seg_migr_filter

    def output_segments(self, result_path='result', segment_file='segments.csv', which_step=3):
        """
        Output segments after step 1, 2, or 3
        step 1: Identify contiguous segments
        step 2: Merge segments
        step 3: Remove overlap

        Attributes
        ----------
        segment_file : string
            File name of the outputed segment
        which_step : int
            Output segments in which step
        """
        if which_step == 3:
            user_seg = self.user_traj[self.user_traj['long_seg_num'] > 0]
            user_seg['seg_selected'] = user_seg['long_seg']
        elif which_step == 2:
            user_seg = self.user_traj[self.user_traj['medium_segment_num'] > 0]
            user_seg['seg_selected'] = user_seg['medium_segment']
        elif which_step == 1:
            user_seg = self.user_traj[self.user_traj['segment_over_prop_num'] > 0]
            user_seg['seg_selected'] = user_seg['segment_over_prop']
        user_seg_migr = user_seg.stack(
            'seg_selected',
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
            lambda x: self.index2date[x['segment_start']]
        )
        user_seg_migr['segment_end_date'] = user_seg_migr.apply(
            lambda x: self.index2date[x['segment_end']]
        )
        user_seg_migr['segment_length'] = (user_seg_migr['segment_end'] -
                                           user_seg_migr['segment_start'])
        user_seg_migr = user_seg_migr.sort(['user_id', 'segment_start_date'], ascending=True)
        if not os.path.isdir(result_path):
            os.makedirs(result_path)
        save_file = os.path.join(result_path, segment_file)
        user_seg_migr.select_columns(
            ['user_id', 'location',
             'segment_start_date', 'segment_end_date', 'segment_length']
        ).export_csv(save_file)

    def plot_segment(self, user_result, if_migration=False,
                     start_date=None, end_date=None,
                     segment_which_step=3,
                     if_save=True, fig_path='figure'):
        """
        Plot migrant daily records in a year and highlight the segments.

        Attributes
        ----------
        user_result : dict
            a user with all the attributes
        if_migration : boolean
            if this record contains a migration event.
            if so, a migration date line will be added
        start_date : str
            start date of the figure in the format of 'YYYYMMDD'
        end_date : str
            end date of the figure in the format of 'YYYYMMDD'
        segment_which_step : int
            1: 'segment_over_prop'
            2: 'medium_segment'
            3: 'long_seg'
        if_save : boolean
            if save the figure
        fig_path : str
            the path to save figures
        """
        segment_which_step_dict = {
            1: 'segment_over_prop',
            2: 'medium_segment',
            3: 'long_seg'
        }
        user_id = user_result['user_id']
        plot_segment = user_result[segment_which_step_dict[segment_which_step]]
        if if_migration:
            migration_day = user_result['migration_day']
            home_start = int(user_result['home_start'])
            des_end = int(user_result['destination_end'])
            start_day = home_start
            end_day = des_end
            # only plot one year's trajectory if the des_end - home_start is longer than one year
            if end_day - start_day > 365 - 1:
                if migration_day <= 180:
                    start_day = home_start
                    end_day = home_start + 365 - 1
                else:
                    start_day = migration_day - 180
                    end_day = migration_day + 184
            start_date = str(self.date_num_long.filter_by(start_day, 'date_num')['date'][0])
            end_date = str(self.date_num_long.filter_by(end_day, 'date_num')['date'][0])
        else:
            date_min = self.raw_traj.filter_by(user_id, 'user_id')['date'].min()
            date_max = self.raw_traj.filter_by(user_id, 'user_id')['date'].max()
            if start_date:
                assert int(start_date) >= date_min, "start date must be later than the first day of this user's records, which is " + str(date_min)
                start_day = self.date_num_long.filter_by(int(start_date), 'date')['date_num'][0]
            else:
                start_day = self.date_num_long.filter_by(date_min, 'date')['date_num'][0]
                start_date = str(self.date_num_long.filter_by(start_day, 'date_num')['date'][0])
            if end_date:
                assert int(end_date) <= date_max, "end date must be earlier than the last day of this user's records, which is " + str(date_max)
                end_day = self.date_num_long.filter_by(int(end_date), 'date')['date_num'][0]
            else:
                end_day = self.date_num_long.filter_by(date_max, 'date')['date_num'][0]
                end_date = str(self.date_num_long.filter_by(end_day, 'date_num')['date'][0])

        duration = end_day - start_day + 1
        fig, ax, location_y_order_loc_appear, appear_loc = plot_traj_common(self.raw_traj, user_id, start_day, end_day, self.date_num_long)
        plot_appear_segment = {k: v for k, v in plot_segment.items() if k in appear_loc}
        for location, value in plot_appear_segment.items():
            y_min = location_y_order_loc_appear[location]
            for segment in value:
                seg_start = segment[0]
                seg_end = segment[1]
                ax.add_patch(
                    patches.Rectangle((seg_start - start_day, y_min),
                                      seg_end - seg_start + 1, 1,
                                      linewidth=4,
                                      edgecolor='red',
                                      facecolor='none')
                )
        if if_migration:
            ax.axvline(migration_day + 0.5 - start_day, color='orange', linewidth=4)
        if not os.path.isdir(fig_path):
            os.makedirs(fig_path)
        save_file = os.path.join(fig_path, user_id + '_' + start_date + '-' + end_date + '_segment')
        if if_save:
            fig.savefig(save_file, bbox_inches="tight")
