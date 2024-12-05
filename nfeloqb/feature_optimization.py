## functions for feature optimization ##
## import modules ##
import pathlib
import pandas as pd

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
            'subset_name' : 'rookie_values',
            'subset' : [
                'rookie_draft_intercept',
                'rookie_draft_slope',
                'rookie_league_reg',
                'rookie_league_cap'
            ]
        },
        {
            'subset_name' : 'weekly_player_adjustments',
            'subset' : [
                'player_sf',
                'player_career_sf_base',
                'player_career_sf_height',
                'player_career_sf_mp'
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
                'player_regression_career_height'
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