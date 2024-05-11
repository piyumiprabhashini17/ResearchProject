import pandas as pd
import os

base_dir = os.getcwd()
input_file_path = os.path.join(base_dir, 'data\\IPL Ball-by-Ball 2008-2022.csv')
datapoints_path = os.path.join(base_dir, 'data\\datapoints.csv')
final_output_path = os.path.join(base_dir, 'data\\datapointwithMatchMarks.csv')

df = pd.read_csv(input_file_path) # read and import csv file to a dataframe(df)

df['bowler_category'] = df['bowler_hand'] + df['bowling_type']  #In the imported dataframe,bowler category column was created by concatinating bowler_hand and bowling_type columns

grouped = df.groupby(['id','inning','batsman']) # Group the dataframe by id,inning and batsman

bowler_category_wise_performance_df = pd.DataFrame() # the dataframe bowler_category_wise_performance_df
bowler_category_wise_overall_performances =	{}        # the dictionary wler_category_wise_overall_performances
overall_performances =	{}  # the dictionary overall_performances
exsistingCategories = {}  #  the dictionary exsistingCategories

categories = [
                'leftfast','leftfastmedium','leftmediumfast', 'leftmedium','leftorthodox',
                'leftlegbreak', 'leftmystery', 'rightfast','rightfastmedium','rightmediumfast',                 #The bowling categories have been defined in an array
                'rightmedium','rightlegbreak','rightorthodox', 'rightmystery'
            ]

for (id,inning,batsman), batsman_group in grouped:
    exsistingCategories = {}
    overallruns = 0
    overallballs = 0
    overalldismissals = 0
    overallkey = str(inning) + str(batsman) 

    if overallkey in overall_performances: 
        overall_performances[overallkey][0] += batsman_group.batsman_runs.sum() 
        overall_performances[overallkey][1] += batsman_group.ball.count()
        overall_performances[overallkey][2] += batsman_group.is_wicket.sum()
    else:
        overall_performances[overallkey] = [batsman_group.batsman_runs.sum(), batsman_group.ball.count(), batsman_group.is_wicket.sum()]

    for (bowler_category), batsman_bowler_category_group in batsman_group.groupby("bowler_category"):
        key = str(inning)+str(batsman)+str(bowler_category)
        cumSumRuns = 0
        cumSumBalls = 0
        cumDismissed = 0

        if key in bowler_category_wise_overall_performances:
            bowler_category_wise_overall_performances[key][0] += batsman_bowler_category_group.batsman_runs.sum()
            bowler_category_wise_overall_performances[key][1] += batsman_bowler_category_group.ball.count()
            bowler_category_wise_overall_performances[key][2] += batsman_bowler_category_group.is_wicket.sum()
            cumSumRuns = bowler_category_wise_overall_performances[key][0]
            cumSumBalls = bowler_category_wise_overall_performances[key][1]
            cumDismissed = bowler_category_wise_overall_performances[key][2]
 
        else:
            bowler_category_wise_overall_performances[key] = [batsman_bowler_category_group.batsman_runs.sum(),
                                                               batsman_bowler_category_group.ball.count(), 
                                                               batsman_bowler_category_group.is_wicket.sum()]
            cumSumRuns = batsman_bowler_category_group.batsman_runs.sum()
            cumSumBalls = batsman_bowler_category_group.ball.count()
            cumDismissed = batsman_bowler_category_group.is_wicket.sum()

        pr = [
                id, inning, batsman, 
                batsman_bowler_category_group['batsman_hand'].iloc[0], bowler_category, 
                batsman_bowler_category_group.bowler.unique().size, batsman_bowler_category_group.batsman_runs.sum(), batsman_bowler_category_group.ball.count(), 
                batsman_bowler_category_group.is_wicket.sum(), cumSumRuns, cumSumBalls, cumDismissed, 
                overall_performances[overallkey][0], overall_performances[overallkey][1], 
                overall_performances[overallkey][2]
            ]
        
        intermediate_df = pd.DataFrame(
                                            [pr], 
                                            columns = [
                                                'id', 'inning', 'batsman', 'batsman_hand', 
                                                'bowler_category', 'unique_bowler_count', 
                                                'runs', 'balls', 'dismissed', 'cum_runs', 
                                                'cum_balls', 'cum_dismissed', 'overall_runs',
                                                 'overall_balls', 'overall_dismissed'
                                            ]
        )

        bowler_category_wise_performance_df = bowler_category_wise_performance_df._append(intermediate_df, ignore_index=True)

        if bowler_category in exsistingCategories:
            continue
        else:
            exsistingCategories[bowler_category] = batsman_bowler_category_group.bowler.unique().size
    
    for key in categories:
        if key in exsistingCategories:
            continue
        else:
            pr = [
                    id, inning, batsman, 
                    batsman_group['batsman_hand'].iloc[0], key, 0, 0, 0, 0, 0, 0, 0, overall_performances[overallkey][0], overall_performances[overallkey][1], 
                    overall_performances[overallkey][2]
                ]
            intermediate_missing_categories_df = pd.DataFrame(
                                                [pr], 
                                                columns = [
                                                            'id', 'inning', 'batsman', 
                                                            'batsman_hand', 'bowler_category',
                                                            'unique_bowler_count', 'runs',
                                                            'balls', 'dismissed', 'cum_runs',
                                                            'cum_balls', 'cum_dismissed', 
                                                            'overall_runs', 'overall_balls', 
                                                            'overall_dismissed'
                                                ]
                                            )
            bowler_category_wise_performance_df=bowler_category_wise_performance_df._append(intermediate_missing_categories_df, ignore_index=True)


outgrouped = bowler_category_wise_performance_df.groupby(["id", "inning", "batsman"])
datapoint_df = pd.DataFrame()
for (id,inning,batsman), batsman_group in outgrouped:
    batsman_group = batsman_group.sort_values(by=['bowler_category'])
    arr = [id, inning, batsman, batsman_group['batsman_hand'].iloc[0]]

    for index, row in batsman_group.iterrows():
        category_wise_arr = [row['unique_bowler_count'], row['cum_runs'], row['cum_balls'], row['cum_dismissed']]
        arr = arr + category_wise_arr

    overall_stats = [batsman_group.overall_runs.max(), batsman_group.overall_balls.max(), batsman_group.overall_dismissed.max()]
    arr = arr + overall_stats
    intermediate_datapoint_df = pd.DataFrame([arr],columns = [
                                                'id', 'inning', 'batsman', 'batsman_hand',
                                                'leftfast_bowlers', 'leftfast_cumruns', 'leftfast_cumballs', 'leftfast_cumdismissals',
                                                'leftfastmedium_bowlers', 'leftfastmedium_cumruns', 'leftfastmedium_cumballs', 'leftfastmedium_cumdismissals', 
                                                'leftlegbreak_bowlers', 'leftlegbreak_cumruns', 'leftlegbreak_cumballs', 'leftlegbreak_cumdismissals', 
                                                'leftmedium_bowlers', 'leftmedium_cumruns', 'leftmedium_cumballs', 'leftmedium_cumdismissals', 
                                                'leftmediumfast_bowlers', 'leftmediumfast_cumruns', 'leftmediumfast_cumballs', 'leftmediumfast_cumdismissals', 
                                                'leftmystery_bowlers', 'leftmystery_cumruns', 'leftmystery_cumballs', 'leftmystery_cumdismissals',
                                                'leftorthodox_bowlers', 'leftorthodox_cumruns', 'leftorthodox_cumballs', 'leftorthodox_cumdismissals', 
                                                'rightfast_bowlers', 'rightfast_cumruns', 'rightfast_cumballs', 'rightfast_cumdismissals',
                                                'rightfastmedium_bowlers', 'rightfastmedium_cumruns', 'rightfastmedium_cumballs', 'rightfastmedium_cumdismissals', 
                                                'rightlegbreak_bowlers', 'rightlegbreak_cumruns', 'rightlegbreak_cumballs', 'rightlegbreak_cumdismissals', 
                                                'rightmedium_bowlers', 'rightmedium_cumruns', 'rightmedium_cumballs', 'rightmedium_cumdismissals', 
                                                'rightmediumfast_bowlers', 'rightmediumfast_cumruns', 'rightmediumfast_cumballs', 'rightmediumfast_cumdismissals', 
                                                'rightmystery_bowlers', 'rightmystery_cumruns', 'rightmystery_cumballs', 'rightmystery_cumdismissals',
                                                'rightorthodox_bowlers', 'rightorthodox_cumruns', 'rightorthodox_cumballs', 'rightorthodox_cumdismissals', 
                                                'overall_runs', 'overall_balls', 'overall_dismissed'
                                            ]
                                        )
    datapoint_df=datapoint_df._append(intermediate_datapoint_df, ignore_index=True)
   
datapoint_df.to_csv(datapoints_path,index=False)

data = pd.read_csv(datapoints_path)
datapointdf = pd.DataFrame()
for group, frame in datapoint_df.groupby(['batsman', 'inning']):
    frame.sort_values(by=['id'])
    prev_value = frame['overall_runs'].iloc[0]
    
    if frame.overall_balls.sum() < 50:
        continue
    for index, row in frame.iterrows():
        row_list = row.tolist()
        if frame['id'].iloc[0] == row['id']:
            row_list = row_list + [prev_value]
        else:
            row_list = row_list + [row['overall_runs'] - prev_value]
        prev_value = row['overall_runs']
        intermediatedatapointdf = pd.DataFrame([row_list],columns = [
                                                'id', 'inning', 'batsman', 'batsman_hand',
                                                'leftfast_bowlers', 'leftfast_cumruns', 'leftfast_cumballs', 'leftfast_cumdismissals',
                                                'leftfastmedium_bowlers', 'leftfastmedium_cumruns', 'leftfastmedium_cumballs', 'leftfastmedium_cumdismissals', 
                                                'leftlegbreak_bowlers', 'leftlegbreak_cumruns', 'leftlegbreak_cumballs', 'leftlegbreak_cumdismissals', 
                                                'leftmedium_bowlers', 'leftmedium_cumruns', 'leftmedium_cumballs', 'leftmedium_cumdismissals', 
                                                'leftmediumfast_bowlers', 'leftmediumfast_cumruns', 'leftmediumfast_cumballs', 'leftmediumfast_cumdismissals', 
                                                'leftmystery_bowlers', 'leftmystery_cumruns', 'leftmystery_cumballs', 'leftmystery_cumdismissals',
                                                'leftorthodox_bowlers', 'leftorthodox_cumruns', 'leftorthodox_cumballs', 'leftorthodox_cumdismissals', 
                                                'rightfast_bowlers', 'rightfast_cumruns', 'rightfast_cumballs', 'rightfast_cumdismissals',
                                                'rightfastmedium_bowlers', 'rightfastmedium_cumruns', 'rightfastmedium_cumballs', 'rightfastmedium_cumdismissals', 
                                                'rightlegbreak_bowlers', 'rightlegbreak_cumruns', 'rightlegbreak_cumballs', 'rightlegbreak_cumdismissals', 
                                                'rightmedium_bowlers', 'rightmedium_cumruns', 'rightmedium_cumballs', 'rightmedium_cumdismissals', 
                                                'rightmediumfast_bowlers', 'rightmediumfast_cumruns', 'rightmediumfast_cumballs', 'rightmediumfast_cumdismissals', 
                                                'rightmystery_bowlers', 'rightmystery_cumruns', 'rightmystery_cumballs', 'rightmystery_cumdismissals',
                                                'rightorthodox_bowlers', 'rightorthodox_cumruns', 'rightorthodox_cumballs', 'rightorthodox_cumdismissals', 
                                                'overall_runs', 'overall_balls', 'overall_dismissed', 'match_runs'
                                            ]
                                        )
        datapointdf=datapointdf._append(intermediatedatapointdf, ignore_index=True)
datapointdf = datapointdf[['batsman', 'id', 'inning', 
                           'leftfast_bowlers', 'leftfastmedium_bowlers', 'leftmediumfast_bowlers', 'leftmedium_bowlers', 'leftorthodox_bowlers', 'leftlegbreak_bowlers', 'leftmystery_bowlers',
                           'rightfast_bowlers', 'rightfastmedium_bowlers', 'rightmediumfast_bowlers', 'rightmedium_bowlers', 'rightorthodox_bowlers', 'rightlegbreak_bowlers', 'rightmystery_bowlers'] 
                           + 
                           [col for col in datapointdf.columns if col not in ['batsman', 'id', 'inning', 
                           'leftfast_bowlers', 'leftfastmedium_bowlers', 'leftmediumfast_bowlers', 'leftmedium_bowlers', 'leftorthodox_bowlers', 'leftlegbreak_bowlers', 'leftmystery_bowlers',
                           'rightfast_bowlers', 'rightfastmedium_bowlers', 'rightmediumfast_bowlers', 'rightmedium_bowlers', 'rightorthodox_bowlers', 'rightlegbreak_bowlers', 'rightmystery_bowlers']]]
datapointdf['batsman_hand'] = datapointdf['batsman_hand'].replace({'right': 1, 'left': 0})
datapointdf.to_csv(final_output_path, index=False)


    