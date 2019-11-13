from __future__ import division
import pandas as pd
import numpy as np
import graphlab as gl
import os
import copy
from array import array
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from traj_utils import *


class TrajRecord():
    """
    # Find Segment
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
    """
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

    def find_migrants(self, num_stayed_days_migrant=90, num_days_missing_gap=7,
                      small_seg_len=30, seg_prop=0.6, min_overlap_part_len=0,
                      max_gap_home_des=30, min_home_segment_len=7,
                      min_des_segment_len=7, max_des_segment_len=14,
                      if_output_segment=True):
        """
        Find migrants step by step

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
        min_home_segment_len : int
            short-term displacement: Home segment length
        min_des_segment_len : int
            short-term displacement: Destination segment length
        max_des_segment_len : int
            short-term displacement: Destination segment length
        if_output_segment : boolean
            whether output detected migrants
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
        # output users with segments of longer than D gap_days_list
        # if if_output_segment:
        #     output_segments(self.user_traj, segment_file, self.min_overlap_part_len, self.index2date)
        # filter out those users with no record or only one location in ['long_seg]
        self.user_traj['long_seg_num'] = self.user_traj['long_seg'].apply(lambda x: len(x))
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

    def plot_migration_segment(self, migrant, plot_column='long_seg',
                               if_save=False, fig_path='figure'):
        """
        Plot migrant daily records in a year and highlight the segments.

        Attributes
        ----------
        migrant : gl.sArray
            one row of a migrant with all the attributes
        plot_column : str
            long_seg
        if_save : boolean
            if save the figure
        fig_path : str
            the path to save figures
        """
        user_id = migrant['user_id']
        plot_segment = migrant[plot_column]
        migration_day = migrant['migration_day']
        home_start = int(migrant['home_start'])
        des_end = int(migrant['destination_end'])
        if des_end - home_start <= 365 - 1:
            start_day = home_start
            end_day = home_start + 365 - 1
        elif des_end - home_start > 365 - 1:
            if migration_day <= 180:
                start_day = home_start
                end_day = home_start + 365 - 1
            else:
                start_day = migration_day - 180
                end_day = migration_day + 184

        daily_record = self.raw_traj.filter_by(user_id, 'user_id').filter_by(
            range(start_day, end_day + 1), 'date_num'
        )
        daily_record['date_count'] = [1] * len(daily_record)
        appear_loc = list(set(daily_record['location']))
        appear_loc.sort()
        # plot_template
        date_plot = range(start_day, end_day + 1)
        date_plot_sort = date_plot * len(appear_loc)
        date_plot_sort.sort()
        template_df_plot = gl.SFrame({'location': appear_loc * len(date_plot),
                                      'date_num': date_plot_sort})

        heatmap_df_join = template_df_plot.join(
            daily_record.select_columns(['location', 'date_count', 'date_num']),
            on=['date_num', 'location'],
            how='left'
        )
        heatmap_df_join = heatmap_df_join.fillna('date_count', 0)
        heatmap_pivot = heatmap_df_join.to_dataframe().pivot("location", "date_num", "date_count")

        height = len(appear_loc)
        fig, ax = plt.subplots(dpi=300, figsize=(28, height))
        cmap = sns.cubehelix_palette(dark=0, light=1, as_cmap=True)

        sns.heatmap(heatmap_pivot, cmap=cmap, cbar=False, linewidths=1)

        for xline in np.arange(365):
            plt.axvline(xline, color='lightgray', alpha=0.5)
        for yline in range(len(appear_loc) + 1):
            plt.axhline(yline, color='lightgray', alpha=0.5)

        location_appear_df = gl.SFrame({'location': appear_loc})
        location_appear_df = location_appear_df.sort('location')
        location_appear_df['y_order'] = range(len(appear_loc))
        location_y_order_loc_appear = (location_appear_df
                                       .select_columns(['location', 'y_order'])
                                       .to_dataframe()
                                       .set_index('location')
                                       .to_dict(orient='dict')['y_order'])

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

        plt.axvline(migration_day + 0.5 - start_day, color='orange', linewidth=4)
        plt.ylabel('Location', fontsize=22)
        plt.xlabel('Date', fontsize=22)
        plt.xlim(0, 365)

        start_date = str(self.date_num_long.filter_by(start_day, 'date_num')['date'][0])
        end_date = str(self.date_num_long.filter_by(end_day, 'date_num')['date'][0])
        month_start = pd.date_range(start=start_date, end=end_date, freq='MS')

        month_start_2 = [str(d)[:4] + str(d)[5:7] + str(d)[8:10] for d in month_start]
        month_mid = [str(int(d) + 14) for d in month_start_2]

        month_all_axis = month_start_2 + month_mid
        month_all_axis.sort()
        month_all_axis_trans = [int(d) for d in month_all_axis]

        ori_xaxis_idx = self.date_num_long.filter_by(month_all_axis_trans, 'date')['date_num']
        ori_xaxis_idx.sort()
        xaxis_idx = np.array(ori_xaxis_idx) + 0.5 - start_day

        plt.xticks(xaxis_idx, month_all_axis, fontsize=22, rotation=30)
        plt.yticks(fontsize=25, rotation='horizontal')
        plt.tick_params(axis='both', which='both', bottom='on', top='off',
                        labelbottom='on', right='off', left='off',
                        labelleft='on')
        plt.subplots_adjust(left=0.05, bottom=0.2, right=0.97, top=0.95)
        if not os.path.isdir(fig_path):
            os.makedirs(fig_path)
        save_path = os.path.join(fig_path, user_id + '_' + str(migration_day))
        plt.show()
        if if_save:
            plt.savefig(save_path, bbox_inches="tight")
        plt.close()
