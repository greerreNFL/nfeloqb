## Built-ins ##
import time
import math
import pathlib
from typing import Tuple, Any
## packages ##
import pandas as pd
import numpy

## local ##
from ..DataModels import QB, Team, ModelConfig, GameContext

class QBModel():
    ## This class is used to store, retrieve, and update data as we
    ## iterate over the game file ##
    def __init__(self,
        games: pd.DataFrame,
        model_config: ModelConfig
    ):
        self.games: pd.DataFrame = games
        self.config: ModelConfig = model_config
        ## league states ##
        self.season_avgs: dict = {} ## storage for season averages ##
        self.team_avgs: dict = {} ## storage for season team averages ##
        ## QB and Team state objects ##
        self.qbs: dict[str, QB] = {} ## storage for most recent QB Data
        self.teams: dict[str, Team] = {} ## storage for most recent team data
        self.qb_records: list[dict[str, Any]] = [] ## storage for weekly QB records over time##
        ## ouput data ##
        self.data: list[dict] = [] ## storage for all game records ##
        self.data_team: list[dict] = [] ## storage for team defense data ##
        ## tracking ##
        self.current_week: int = 1 ## track the current week to know when it has changed ##
        self.model_run_time: float = 0 ## track the time it takes to run the model ##
        ## import original elo file location ##
        data_folder = pathlib.Path(__file__).parent.parent.resolve()
        self.original_file_loc = '{0}/Manual Data/original_elo_file.csv'.format(data_folder)
        ## initial ##
        self.chrono_sort() ## sort by date, so games can be iter'd ##
        self.add_averages()
    
    ##############################
    ## INITIALIZATION FUNCTIONS ##
    ##############################
    def chrono_sort(self):
        ## sort games by date ##
        self.games = self.games.sort_values(
            by=['season', 'week', 'game_id'],
            ascending=[True, True, True]
        ).reset_index(drop=True)
    
    def add_averages(self):
        ## adds the avg QB values for teams and leagues which are used in reversion ##
        ## calc team averages ##
        team_avgs = self.games.groupby(
            ['season', 'team']
        )['team_VALUE'].mean().reset_index()
        ## calc league average ##
        season_avgs = self.games.groupby(
            ['season']
        )['team_VALUE'].mean().reset_index()
        ## write to stoarge ##
        for index, row in team_avgs.iterrows():
            self.team_avgs['{0}{1}'.format(row['season'], row['team'])] = row['team_VALUE']
        for index, row in season_avgs.iterrows():
            self.season_avgs[row['season']] = row['team_VALUE']
    
    ####################################
    ## RETRIEVAL METHODS FOR AVERAGES ##
    ####################################
    def get_prev_season_team_avg(self, season, team):
        ## get the teams previous season average while controlling for errors ##
        ## the first season will not have a pervous season ##
        return self.team_avgs.get('{0}{1}'.format(
            season - 1, team
        ), self.config.values['init_value'])
    
    def get_prev_season_league_avg(self, season):
        ## get the leagues previous season average while controlling for errors ##
        ## the first season will not have a pervous season ##
        return self.season_avgs.get(season - 1, self.config.values['init_value'])
    
    ###################################
    ## RETRIEVAL METHODS FOR OBJECTS ##
    ###################################
    def get_team(self, team):
        ## retrieve the team object from storage ##
        if team not in self.teams:
            self.teams[team] = Team(team, self.config)
        return self.teams[team]
    
    def get_qb(self, row):
        '''
        Retrives a QB object from storage, or creates it if it doesn't exist
        '''
        if row['player_id'] not in self.qbs:
            ## init the qb if it doesn't exist ##
            self.qbs[row['player_id']] = QB(
                player_id=row['player_id'],
                player_name=row['player_display_name'],
                config=self.config,
                draft_number=row['draft_number'],
                inital_team_avg=self.get_prev_season_team_avg(row['season'], row['team']),
                inital_league_avg=self.get_prev_season_league_avg(row['season']),
                first_game_date=row['gameday'],
                first_game_season=row['season']
            )
        ## retieve the qb from storage ##
        return self.qbs[row['player_id']]
    
    def get_objects(self, row) -> Tuple[QB, Team, Team]:
        '''
        Helper to retrive QB and Team objects while handling object creation, regression, and
        inter-object adjustments (ie team adjustment on the QB). This is used internally by the model
        to init populate a new game with necessary information about the QB and teams, but it is also used
        externally by the Elo file constructor. Since it touches state, external use should not be used
        iteratively.

        Parameters:
        * row: dict - a game df row, or a dict simulating a game df row. Minimum required fields are:
            player_id, team, opponent, season

        Returns:
        * tuple - A tuple containing the QB and Team objects
        '''
        ## get objects and init as necessary using the get_X functions ##
        qb = self.get_qb(row)
        team = self.get_team(row['team'])
        opponent = self.get_team(row['opponent'])
        ## handle regressions ##
        ## QB ##
        if row['season'] > (qb.last_game_season if qb.last_game_season is not None else row['season']):
            qb.regress_value(
                prev_season_league_avg=self.get_prev_season_league_avg(row['season']),
            )
        ## TEAM ##
        if row['season'] > (team.last_game_season_off if team.last_game_season_off is not None else row['season']):
            team.regress_offense(
                qb_val=qb.current_value,
                prev_season_league_avg=self.get_prev_season_league_avg(row['season'])
            )
        ## OPPONENT ##
        if row['season'] > (opponent.last_game_season_def if opponent.last_game_season_def is not None else row['season']):
            opponent.regress_defense()
        ## return the objects ##
        return qb, team, opponent


    #####################
    ## MODEL FUNCTIONS ##
    #####################
    def run_model(self):
        '''
        Iters through the games df, updates states, and saves the output
        '''
        ## set a start epoch time ##
        start_time = time.time()
        ## clear out any existing values ##
        self.qbs = {} ## storage for most recent QB Data
        self.teams = {} ## storage for most recent team data
        self.data = [] ## storage for all game records ##
        self.current_week = 1 ## track the current week to know when it has changed ##
        ## iterate through games df ##
        for index, row in self.games.iterrows():
            ## retrive the objects ##
            qb, team, opponent = self.get_objects(row)
            ## create a game context ##
            game_context = GameContext(
                game_id=row['game_id'],
                config=self.config,
                temp=row['temp'],
                wind=row['wind']
            )
            weather_adj = game_context.weather_adj()
            ## get values ##
            qb_expected_value = qb.get_value(team) ## use get_value to account for the team adjustment
            team_off_value = team.off_value
            team_def_adjustment = opponent.def_value
            ## calc expectation relative to def ##
            qb_expected_value_adj_def = qb_expected_value - team_def_adjustment + weather_adj
            ## calculate the team adjustment ##
            qb_adj = qb_expected_value - team_off_value
            ## calculate performance relative to expectations ##
            ## Defensive rating is positive for above average teams, therefore we add it ##
            def_adjusted_performance = row['player_VALUE'] + team_def_adjustment - weather_adj
            ## qb expected value is relative to an average defense (ie 0). Netting it from the actual
            ## performance provides some notion of how the defense may have impacted the output ##
            performance_vs_expected = qb_expected_value - (row['player_VALUE'] - weather_adj)
            ## update values ##
            qb.update_value(
                ## qbs performance is inclusive of quality of defense ##
                value=def_adjusted_performance,
                proj=qb_expected_value_adj_def,
                gameday=row['gameday'],
                season=row['season'],
                team=row['team']
            )
            team.update_off_value(
                ## team receives the same measurement of performance as qb ##
                value=def_adjusted_performance,
                ## note, qb_expected_value is the qb's current value before the game.
                ## netting it from the updated current value gives the adjustment made to the
                ## qb's value ##
                qb_adj=qb.current_value - qb_expected_value,
                gameday=row['gameday'],
                season=row['season']
            )
            opponent.update_def_value(
                ## defensive values are netted vs expected
                value=performance_vs_expected,
                gameday=row['gameday'],
                season=row['season']
            )
            ## add opponent defense data to storage ##
            self.data_team.append({
                'game_id' : row['game_id'],
                'season' : row['season'],
                'week' : row['week'],
                'team' : row['opponent'],
                'def_value_pre' : round(team_def_adjustment, 3),
                'def_value_post' : round(opponent.def_value, 3)
            })
            ## construct the row records ##
            row['qb_value_pre'] = qb_expected_value
            row['team_value_pre'] = team_off_value
            row['qb_adj'] = qb_adj
            row['opponent_def_value_pre'] = team_def_adjustment
            row['qb_value_pre_def_adj'] = qb_expected_value_adj_def
            row['player_VALUE_adj'] = def_adjusted_performance
            row['qb_value_post'] = qb.current_value
            row['team_value_post'] = team.off_value
            row['opponent_def_value_post'] = opponent.def_value
            ## write row to data ##
            self.data.append(row)
            qb_record = qb.as_record()
            qb_record['value_pre'] = row['qb_value_pre']
            qb_record['opponent_def_value_pre'] = row['opponent_def_value_pre']
            qb_record['value_pre_def_adj'] = row['qb_value_pre_def_adj']
            qb_record['value_performance_def_adj'] = row['player_VALUE_adj']
            qb_record['game_id'] = row['game_id']
            qb_record['season'] = row['season']
            qb_record['week'] = row['week']
            ## round all instances of floats ##
            for key in qb_record:
                if isinstance(qb_record[key], float):
                    qb_record[key] = round(qb_record[key], 3)
            self.qb_records.append(qb_record)
        end_time = time.time()
        self.model_runtime = end_time - start_time
    
    ## scoring ##
    def add_elo(self, df):
        ## add elo values from 538 to df for comparison of accuracy ##
        ## read in elo data ##
        elo = pd.read_csv(
            self.original_file_loc,
            index_col=0,
        )
        ## flatten elo df ##
        elo = pd.concat([
            elo[[
                'date', 'team1', 'team2', 'qb1', 'qb1_value_pre', 'qb1_adj'
            ]].rename(columns={
                'date' : 'gameday',
                'team1' : 'team',
                'team2' : 'opponent',
                'qb1' : 'player_display_name',
                'qb1_value_pre' : 'f38_projected_value',
                'qb1_adj' : 'f38_team_adj'
            }),
            elo[[
                'date', 'team2', 'team1', 'qb2', 'qb2_value_pre', 'qb2_adj'
            ]].rename(columns={
                'date' : 'gameday',
                'team2' : 'team',
                'team1' : 'opponent',
                'qb2' : 'player_display_name',
                'qb2_value_pre' : 'f38_projected_value',
                'qb2_adj' : 'f38_team_adj'
            }),
        ])
        ## dedupe ##
        elo = elo.groupby([
            'gameday', 'team', 'opponent', 'player_display_name'
        ]).head(1)
        ## convert elo to value ##
        elo['f38_projected_value'] = elo['f38_projected_value'] / 3.3
        elo['f38_team_adj'] = elo['f38_team_adj'] / 3.3
        ## merge elo data ##
        df = pd.merge(
            df,
            elo,
            on=['gameday', 'team', 'opponent', 'player_display_name'],
            how='left'
        )
        ## return df ##
        return df
    
    def score_model(self, first_season=2009, add_elo=True):
        ## function for scoring model for testing purposes ##
        ## create df from data ##
        df = pd.DataFrame(self.data)
        ## get mean squared error ##
        df['se'] = (df['qb_value_pre_def_adj'] - df['player_VALUE']) ** 2 ## expectation for game including D, vs actual ##
        df['abs_error'] = numpy.absolute(df['qb_value_pre_def_adj'] - df['player_VALUE'])
        ## create rolling averages as a quasi control ##
        for roll in [8, 16, 24, 32]:
            df['se_r{0}'.format(roll)] = (
                ## prediction based on a simple rolling average ##
                df.groupby(['player_id'])['player_VALUE'].transform(
                    lambda x: x.rolling(roll, min_periods=1).mean().shift()
                ).fillna(df['qb_value_pre_def_adj']) -
                ## actual value ##
                df['player_VALUE']
            ) ** 2
            df['abs_error_r{0}'.format(roll)] = numpy.absolute(
                ## prediction based on a simple rolling average ##
                df.groupby(['player_id'])['player_VALUE'].transform(
                    lambda x: x.rolling(roll, min_periods=1).mean().shift()
                ).fillna(df['qb_value_pre_def_adj']) -
                ## actual value ##
                df['player_VALUE']
            )
        ## only look at data past first season ##
        ## this is to give model time to catch up since we are starting in 1999 ##
        ## and veteran QBs are treated like rookies in that season ##
        df = df[df['season'] >= first_season].copy()
        ## copy config to serve as a record of what was used ##
        record = self.config.values.copy()
        ## add rmse and mae to record ##
        record['rmse'] = df['se'].mean() ** 0.5
        record['mae'] = df['abs_error'].mean()
        ## add specials ##
        record['mae_first_16'] = numpy.nanmean(numpy.where(
            df['start_number'] <= 16,
            df['abs_error'],
            numpy.nan
        ))
        record['mae_backup'] = numpy.nanmean(numpy.where(
            df['qb_value_pre'] - df['team_value_pre'] < -15,
            df['abs_error'],
            numpy.nan
        ))
        ## add rolling averages to record ##
        for roll in [8, 16, 24, 32]:
            record['rmse_r{0}'.format(roll)] = df['se_r{0}'.format(roll)].mean() ** 0.5
            record['mae_r{0}'.format(roll)] = df['abs_error_r{0}'.format(roll)].mean()
        if add_elo:
            ## add elo data ##
            df = self.add_elo(df)
            ## add comparison to 538 ##
            f = df[~pd.isnull(df['f38_projected_value'])].copy()
            f['f38_se'] = (f['f38_projected_value'] - f['player_VALUE_adj']) ** 2
            record['delta_vs_538'] = (f['f38_se'].mean() ** 0.5) - (f['se'].mean() ** 0.5)
            ## add rookie model comp ##
            r = f[f['start_number'] <= 10].copy()
            record['delta_vs_538_rookies'] = (r['f38_se'].mean() ** 0.5) - (r['se'].mean() ** 0.5)
        record['model_runtime'] = self.model_runtime
        ## return record ##
        return record
    
    def score_adj(self, first_season=2009):
        ## Function for scoring the team adjustment ##
        ## While the model should try to predict VALUE as best as it can ##
        ## The team adj should try to get as close to the 538 team adj as this ##
        ## ghe main nfelo model has already been optimized for this value ##
        ## create df from data ##
        df = pd.DataFrame(self.data)
        df = df[df['season'] >= first_season].copy()
        ## add elo ##
        df = self.add_elo(df)
        ## add comparison to 538 ##
        f = df[~pd.isnull(df['f38_team_adj'])].copy()
        f['adj_se'] = (f['f38_team_adj'] - f['qb_adj']) ** 2
        ## add rmse to record ##
        record = self.config.copy()
        record['adjustment_rmse'] = f['adj_se'].mean() ** 0.5
        return record