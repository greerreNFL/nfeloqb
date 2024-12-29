## functions for feature optimization ##
## import modules ##
import pathlib
import pandas as pd
import datetime
## import resources ##
from .Resources import *
from .DataModels import ModelConfig
from .Optimizer import ConfigOptimizer

def optimize_config_subsets(save_result:bool=True, update_config:bool=False):
    '''
    Optimizes the config as defined in the model_config.json file, using subesets
    for better optimization results
    '''
    ## load config ##
    package_folder = pathlib.Path(__file__).parent.parent.resolve()
    config = ModelConfig.from_file('{0}/model_config.json'.format(package_folder))
    ## get the games data ##
    data = DataLoader() 
    ## define subsets ##
    subsets = [
        {
            'subset_name' : 'weekly_player_adjustments',
            'subset' : [
                'player_sf',
                'player_career_sf_base',
                'player_career_sf_height',
                'player_career_sf_mp',
                'player_prog_disc_alpha'
            ]
        },
        {
            'subset_name' : 'rookie_values',
            'subset' : [
                'rookie_draft_intercept',
                'rookie_draft_slope',
                'rookie_league_reg',
                'rookie_league_cap'
            ]
        },
        {
            'subset_name' : 'offseason_player_regression',
            'subset' : [
                'player_regression_league_mp',
                'player_regression_league_height',
                'player_regression_career_mp',
                'player_regression_career_height',
            ]
        },
        {
            'subset_name' : 'player_adjustment_core',
            'subset' : [
                'player_sf',
                'player_career_sf_base',
                'player_career_sf_height',
                'player_career_sf_mp',
                'player_regression_league_mp',
                'player_regression_league_height',
                'player_regression_career_mp',
                'player_regression_career_height',
                'player_team_adj_allotment_disc',
                'player_prog_disc_alpha',
                'team_def_sf',
                'team_def_reversion',
            ]
        }
    ]
    ## optimize each subset ##
    for subset in subsets:
        optimizer = ConfigOptimizer(
            data=data.model_df,
            config=config,
            subset=subset['subset'],
            subset_name=subset['subset_name']
        )
        optimizer.optimize(save_result, update_config)


def optimize_config(save_result:bool=True, update_config:bool=False):
    '''
    Optimizes the config as defined in the model_config.json file.

    Parameters
    * save_result : bool - Whether to save the result to a csv file.
    * update_config : bool - Whether to update the model_config.json file with the optimized values.
    '''
    ## load config ##
    package_folder = pathlib.Path(__file__).parent.parent.resolve()
    config = ModelConfig.from_file('{0}/model_config.json'.format(package_folder))
    ## get the games data ##
    data = DataLoader() 
    optimizer = ConfigOptimizer(data.model_df, config)
    optimizer.optimize(save_result, update_config)

def optimize_config_subsets_with_rand(rounds:int=25):
    '''
    Optimizes the config as defined in the model_config.json file, using subesets
    for better optimization results AND using rounds of randomized best guesses to 
    explore global optimization and validate whether the optimizer is getting
    stuck due to local minima.
    '''
    ## load config ##
    package_folder = pathlib.Path(__file__).parent.parent.resolve()
    config = ModelConfig.from_file('{0}/model_config.json'.format(package_folder))
    ## get the games data ##
    data = DataLoader() 
    ## define subsets ##
    subsets = [
        {
            'subset_name' : 'weekly_player_adjustments',
            'subset' : [
                'player_sf',
                'player_career_sf_base',
                'player_career_sf_height',
                'player_career_sf_mp',
                'player_prog_disc_alpha'
            ]
        },
        {
            'subset_name' : 'rookie_values',
            'subset' : [
                'rookie_draft_intercept',
                'rookie_draft_slope',
                'rookie_league_reg',
                'rookie_league_cap'
            ]
        },
        {
            'subset_name' : 'offseason_player_regression',
            'subset' : [
                'player_regression_league_mp',
                'player_regression_league_height',
                'player_regression_career_mp',
                'player_regression_career_height'
            ]
        },
        {
            'subset_name' : 'player_adjustment_core',
            'subset' : [
                'player_sf',
                'player_career_sf_base',
                'player_career_sf_height',
                'player_career_sf_mp',
                'player_regression_league_mp',
                'player_regression_league_height',
                'player_regression_career_mp',
                'player_regression_career_height',
                'player_prog_disc_alpha',
                'player_team_adj_allotment_disc',
                'team_def_sf',
                'team_def_reversion',
            ]
        }
    ]
    ## optimize each subset ##
    for subset in subsets:
        print('Optimizing subset: {0}'.format(subset['subset_name']))
        best_recs = []
        for i in range(rounds):
            print('Round {0} of {1}'.format(i+1, rounds))
            optimizer = ConfigOptimizer(
                data=data.model_df,
                config=config,
                subset=subset['subset'],
                subset_name=subset['subset_name'],
                randomize_bgs=True
            )
            optimizer.optimize(False, False)
            best_recs.append(optimizer.get_best_record())
            ## save the best records ##
            df = pd.DataFrame(best_recs)
            df.to_csv('{0}/Optimizer/Results/{1}{2}.csv'.format( 
                pathlib.Path(__file__).parent.resolve(),
                datetime.datetime.now().strftime('%Y%m%d'),
                '_{0}_randomized_bgs'.format(subset['subset_name'])
            ))


def optimize_config_with_rand(rounds:int=100, subset_names:list[str]=[]):
    '''
    Optimizes the config as defined in the model_config.json file, using rounds of
    randomized best guesses to explore global optimization and validate whether
    the optimizer is getting stuck due to local minima.
    '''
    ## load config ##
    package_folder = pathlib.Path(__file__).parent.parent.resolve()
    config = ModelConfig.from_file('{0}/model_config.json'.format(package_folder))
    ## get the games data ##
    data = DataLoader() 
    ## define subsets ##
    subsets = [
        {
            'subset_name' : 'player_adjustment',
            'subset' : [
                'player_sf',
                'player_career_sf_base',
                'player_career_sf_height',
                'player_career_sf_mp',
                'player_regression_league_mp',
                'player_regression_league_height',
                'player_regression_career_mp',
                'player_regression_career_height',
                'player_prog_disc_alpha',
                'team_def_sf',
                'team_def_reversion',
            ],
            'objective' : 'mae'
        },
        {
            'subset_name' : 'weather',
            'subset' : [
                'wind_disc_height',
                'wind_disc_mp',
                'temp_disc_height',
                'temp_disc_mp'
            ],
            'objective' : 'mae'
        },
        {
            'subset_name' : 'backup_values',
            'subset' : [
                'player_team_adj_allotment_disc',
            ],
            'objective' : 'mae_backup'
        },
        {
            'subset_name' : 'rookie_values',
            'subset' : [
                'rookie_draft_intercept',
                'rookie_draft_slope',
                'rookie_league_reg',
                'rookie_league_cap'
            ],
            'objective' : 'mae_first_16'
        }
    ]
    ## optimize each subset ##
    for subset in subsets:
        if subset['subset_name'] not in subset_names and len(subset_names) > 0:
            continue
        print('Optimizing subset: {0}'.format(subset['subset_name']))
        best_recs = []
        for i in range(rounds):
            print('Round {0} of {1}'.format(i+1, rounds))
            optimizer = ConfigOptimizer(
                data=data.model_df,
                config=config,
                subset=subset['subset'],
                subset_name=subset['subset_name'],
                objective_name=subset['objective'] if 'objective' in subset else 'mae',
                randomize_bgs=True
            )
            optimizer.optimize(False, False)
            best_recs.append(optimizer.get_best_record())
            ## save the best records ##
            df = pd.DataFrame(best_recs)
            df.to_csv('{0}/Optimizer/Results/{1}{2}.csv'.format( 
                pathlib.Path(__file__).parent.resolve(),
                datetime.datetime.now().strftime('%Y%m%d'),
                '_{0}_randomized_bgs'.format(subset['subset_name'])
            ))