## wrapper for running the pacakge ##
## import modules ##
import pathlib
import json
import datetime
import pandas as pd

## get games for last played ##
import nfelodcm as dcm

## import resources ##
from .Resources import *
from .DataModels import ModelConfig

## import env ##
import os
from dotenv import load_dotenv

try:
    env_path = '{0}/.env'.format(
        pathlib.Path(__file__).parent.parent.resolve()
    )
    load_dotenv(env_path)
except Exception as e:
    ## if running as action, these will already be in env ##
    pass

def run(perform_starter_update=False, model_only=False, force_run=False):
    ## load configs and meta ##
    config = None
    meta = None
    package_folder = pathlib.Path(__file__).parent.parent.resolve()
    config = ModelConfig.from_file('{0}/model_config.json'.format(package_folder))
    with open('{0}/package_meta.json'.format(package_folder)) as fp:
        meta = json.load(fp)
    ## init AT ##
    at_wrapper = AirtableWrapper(
        None,
        at_config={
            'base' : os.environ.get('AIRTABLE_BASE'),
            'qb_table' : os.environ.get('AIRTABLE_QB_TABLE'),
            'starter_table' : os.environ.get('AIRTABLE_START_TABLE'),
            'token' : os.environ.get('AIRTABLE_TOKEN'),
            'qb_fields' : [os.environ.get('AIRTABLE_QB_FIELDS')],
            'dropdown_field' : os.environ.get('AIRTABLE_DROPDOWN_ID')
        },
        perform_starter_update=perform_starter_update
    )
    ## get last starter change ##
    last_starter_change = at_wrapper.get_last_update()
    last_package_update = meta['last_updated']
    last_package_week = meta['last_full_week']
    ## get last full week ##
    last_full_season, last_full_week = dcm.get_season_state()
    last_full_week = '{0}_{1}'.format(last_full_season, last_full_week)
    ## see if update is required ##
    if last_package_update is not None and not force_run:
        if last_starter_change < pd.to_datetime(last_package_update, utc=True) and last_full_week == last_package_week:
            return None
    ## load data ##
    data = DataLoader()
    ## run model ##
    print('Running QB model...')
    model = QBModel(data.model_df, config)
    model.run_model()
    if model_only:
        return model
    ## update starters ##
    at_wrapper.model_df = model.games
    at_wrapper.update_qb_table()
    at_wrapper.update_qb_options()
    ## The script will run automatically now, which means the starters will be updated outside
    ## of this process ##
    ##      at_wrapper.update_starters()
    ##      pause and wait for confirmation that manual edits have been made in airtable ##
    ##      decision = input('When starters have been updated in Airtable, type "RUN" and press enter:')
    ##      print(decision)
    ## run elo model ##
    print('Running Elo model...')
    elo = Elo(
        data.games,
        pd.DataFrame(model.data)
    )
    elo.run()
    ## construct elo file ##
    constructor = EloConstructor(
        data.games,
        model,
        at_wrapper,
        elo,
        package_folder
    )
    constructor.construct_elo_file()
    ## save flattened qb and team data ##
    pd.DataFrame(model.data_team).sort_values(
        by=['team', 'season', 'week'],
        ascending=[True, True, True]
    ).reset_index(drop=True).to_csv(
        '{0}/Other Data/defensive_adjustments.csv'.format(package_folder),
        index=False
    )
    ## save flattened qb records ##
    pd.DataFrame(model.qb_records).to_csv(
        '{0}/Other Data/weekly_qb_states.csv'.format(package_folder),
        index=False
    )
    ## save meta data ##
    meta_constructor = MetaConstructor(
        players=data.db['players'],
        elo_file=constructor.new_file
    )
    ## update the last updated timestamp ##
    with open('{0}/package_meta.json'.format(package_folder), 'w') as fp:
        json.dump(
            {
                'last_updated' : datetime.datetime.utcnow().isoformat() + 'Z',
                'last_full_week' : last_full_week
            },
            fp
        )
