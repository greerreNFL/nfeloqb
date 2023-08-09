import pandas as pd
import numpy
import time
import math
import pathlib


class QBModel():
    ## This class is used to store, retrieve, and update data as we
    ## iterate over the game file ##
    def __init__(self, games, model_config):
        self.games = games
        self.config = model_config
        self.season_avgs = {} ## storage for season averages ##
        self.team_avgs = {} ## storage for season team averages ##
        self.qbs = {} ## storage for most recent QB Data
        self.teams = {} ## storage for most recent team data
        self.data = [] ## storage for all game records ##
        self.league_avg_def = model_config['init_value'] ## each week, we need to calculate the league avg def for adjs ##
        self.current_week = 1 ## track the current week to know when it has changed ##
        self.model_run_time = 0 ## track the time it takes to run the model ##
        ## import original elo file location ##
        data_folder = pathlib.Path(__file__).parent.parent.resolve()
        self.original_file_loc = '{0}/Manual Data/original_elo_file.csv'.format(data_folder)
        ## initial ##
        self.chrono_sort() ## sort by data, so games can be iter'd ##
        self.add_averages()
    
    ## setup functions ##
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
    
    ## retrieval functions for averages ##
    def get_prev_season_team_avg(self, season, team):
        ## get the teams previous season average while controlling for errors ##
        ## the first season will not have a pervous season ##
        return self.team_avgs.get('{0}{1}'.format(
            season - 1, team
        ), self.config['init_value'])
    
    def get_prev_season_league_avg(self, season):
        ## get the leagues previous season average while controlling for errors ##
        ## the first season will not have a pervous season ##
        return self.season_avgs.get(season - 1, self.config['init_value'])
    
    ## retrieval functions for QB ##
    def init_qb(self, qb_id, season, team, draft_number, gameday):
        ## initialize qb into storage while also calculating their initial value ##
        ## round is blank (ie undrafted) enter high value ##
        if pd.isnull(draft_number):
            draft_number = self.config['rookie_undrafted_draft_number']
        ## get the previous season averages for the team and league
        prev_season_team_avg = self.get_prev_season_team_avg(season, team)
        prev_season_league_avg = self.get_prev_season_league_avg(season)
        ## calculate the initial value ##
        init_value = min(
            ## value over team's previous average based on draft number ##
            (self.config['rookie_draft_intercept'] + (self.config['rookie_draft_slope'] * math.log(draft_number))) +
            ## team value is regressed to the league average ##
            (
                ((1-self.config['rookie_league_reg']) * prev_season_team_avg) +
                (self.config['rookie_league_reg'] * prev_season_league_avg)
            ),
            ## value is capped at a discount of the league average ##
            ((1+self.config['rookie_league_cap']) * prev_season_league_avg)
        )
        ## write to storage ##
        self.qbs[qb_id] = {
            'current_value' : init_value,
            'current_variance' : init_value,
            'rolling_value' : init_value,
            'starts' : 0,
            'season_starts' : 0,
            'first_game_date' : gameday,
            'first_game_season' : season,
            'last_game_date' : None,
            'last_game_season' : None
        }
        
    def s_curve(self, height, mp, x, direction='down'):
        ## calculate s-curve, which are used for progression discounting and multiplying ##
        if direction == 'down':
            return (
                1 - (1 / (1 + 1.5 ** (
                    (-1 * (x - mp)) *
                    (10 / mp)
                )))
            ) * height
        else:
            return (1-(
                1 - (1 / (1 + 1.5 ** (
                    (-1 * (x - mp)) *
                    (10 / mp)
                )))
            )) * height
    
    def handle_qb_regression(self, qb, season):
        ## regress qb to the league average ##
        ## first, get the previous season average ##
        prev_season_league_avg = self.get_prev_season_league_avg(season)
        ## determine regression amounts based on model curves ##
        league_regression = self.s_curve(
            self.config['player_regression_league_height'],
            self.config['player_regression_league_mp'],
            qb['starts'],
            'down'
        )
        career_regression = self.s_curve(
            self.config['player_regression_career_height'],
            self.config['player_regression_career_mp'],
            qb['starts'],
            'up'
        )
        ## calculate the new value ##
        ## if the qb didnt play much the previous (ie was a backup) this is ##
        ## signal that they are not league average quality ##
        ## In this case, we discount the league average regression portion ##
        league_regression = (
            league_regression *
            self.s_curve(
                1,
                4,
                qb['season_starts'],
                'up'
            )
        )
        ## normalize the combined career and league regression to not exceed 100% ##
        total_regression = league_regression + career_regression
        if total_regression > 1:
            league_regression = league_regression / total_regression
            career_regression = career_regression / total_regression
        ## calculate value ##
        qb['current_value'] = (
            (1 - league_regression - career_regression) * qb['current_value'] +
            (league_regression * prev_season_league_avg) +
            (career_regression * qb['rolling_value'])
        )
        ## update season ##
        ## return the qb object ##
        return qb
    
    def get_qb_value(self, row):
        ## retrieve the current value of the qb before the gamae ##
        ## this takes the entire row as we may need to unpack values to send to
        ## other models ##
        ## get qb from storage ##
        if row['player_id'] not in self.qbs:
            self.init_qb(
                row['player_id'], row['season'], row['team'],
                row['draft_number'], row['gameday']
            )
        qb = self.qbs[row['player_id']]
        ## handle regression ##
        if qb['last_game_season'] is None:
            pass
        elif row['season'] > qb['last_game_season']:
            qb = self.handle_qb_regression(qb, row['season'])
        ## return value ##
        return qb['current_value']
    
    def update_qb_value(self, qb_id, value, gameday, season):
        ## to speed up, minimize the number of times we lookup from storage ##
        qb_ = self.qbs[qb_id]
        ## first store the pre-update value, which i sneeded for rolling variance ##
        old_value = qb_['current_value']
        ## update the qb value after the game ##
        qb_['current_value'] = (
            self.config['player_sf'] * value +
            (1 - self.config['player_sf']) * qb_['current_value']
        )
        ## for rolling value, use a progressively deweighted ewma ##
        ## set rolling sf ##
        rolling_sf = (
            self.config['player_career_sf_base'] +
            self.s_curve(
                self.config['player_career_sf_height'],
                self.config['player_career_sf_mp'],
                qb_['starts'],
                'down'
            )
        )
        qb_['rolling_value'] = (
            rolling_sf * value +
            (1 - rolling_sf) * qb_['rolling_value']
        )
        ## update variance ##
        ## 𝜎2𝑛=(1−𝛼)𝜎2𝑛−1+𝛼(𝑥𝑛−𝜇𝑛−1)(𝑥𝑛−𝜇𝑛) ##
        ## https://stats.stackexchange.com/questions/6874/exponential-weighted-moving-skewness-kurtosis ##
        qb_['current_variance'] = (
            self.config['player_sf'] * (value - old_value) * (value - qb_['current_value']) +
            (1 - self.config['player_sf']) * qb_['current_variance']
        )
        ## update meta ##
        qb_['starts'] += 1
        qb_['season_starts'] += 1
        qb_['last_game_date'] = gameday
        qb_['last_game_season'] = season
        ## write back to storage ##
        self.qbs[qb_id] = qb_
        ## return updated value ##
        return qb_['current_value']
    
    ## function for initing teams ##
    def init_team(self, team):
        ## initialize team into storage ##
        self.teams[team] = {
            'off_value' : self.config['init_value'],
            'def_value' : self.config['init_value']
        }
        ## return the team object ##
        return self.teams[team]
    
    ## functions for getting team def values ##
    def update_league_avg_def(self):
        ## take the average of all team defensive scores ##
        ## use this to update the league average variable ##
        defensive_values = []
        for team, val in self.teams.items():
            defensive_values.append(val['def_value'])
        ## update ##
        self.league_avg_def = numpy.mean(defensive_values)
    
    def handle_team_def_regression(self, team_obj):
        ## simple func for regressing team values to mean
        team_obj['def_value'] = (
            (1 - self.config['team_def_reversion']) * team_obj['def_value'] +
            self.config['team_def_reversion'] * self.league_avg_def
        )
        return team_obj
    
    def get_team_def_value(self, team, week):
        ## get the defensive value of the team ##
        ## if the week has changed, update the league average ##
        if self.current_week != week:
            self.update_league_avg_def()
            self.current_week = week
        ## retrieve the team from storage ##
        if team not in self.teams:
            self.init_team(team)
        team_obj = self.teams[team]
        ## regress if needed ##
        if week == 1:
            team_obj = self.handle_team_def_regression(team_obj)
            ## update in db since this value has now changed ##
            self.teams[team]['def_value'] = team_obj['def_value']
        ## return the teams def value and val vs league ##
        return team_obj['def_value'], team_obj['def_value'] - self.league_avg_def
    
    def update_team_def_value(self, team, value):
        ## update the team value after the game ##
        self.teams[team]['def_value'] = (
            self.config['team_def_sf'] * value +
            (1 - self.config['team_def_sf']) * self.teams[team]['def_value']
        )
        ## return updated value ##
        return self.teams[team]['def_value']
    
    ## functions for getting team off values ##
    def handle_team_off_regression(self, team_obj, qb_val, season):
        ## simple func for regressing team values to the week 1 starter value ##
        team_obj['off_value'] = (
            (1 - self.config['team_off_league_reversion'] - self.config['team_off_qb_reversion']) * team_obj['off_value'] +
            self.config['team_off_qb_reversion'] * qb_val +
            self.config['team_off_league_reversion'] * self.get_prev_season_league_avg(season)
        )
        return team_obj
    
    def get_team_off_value(self, team, qb_val, season):
        ## function for getting the offensive value of the team ##
        ## retrieve the team from storage ##
        if team not in self.teams:
            self.init_team(team)
        team_obj = self.teams[team]
        ## handle offensive regression as needed ##
        if self.current_week == 1:
            team_obj = self.handle_team_off_regression(team_obj, qb_val, season)
            ## update in db since this value has now changed ##
            self.teams[team]['off_value'] = team_obj['off_value']
        ## return off value and adj relative to qb ##
        return team_obj['off_value'], qb_val - team_obj['off_value']
    
    def update_team_off_value(self, team, value):
        ## update the team value after the game ##
        self.teams[team]['off_value'] = (
            self.config['team_off_sf'] * value +
            (1 - self.config['team_off_sf']) * self.teams[team]['off_value']
        )
        ## return updated value ##
        return self.teams[team]['off_value']
    
    ## model functions ##
    def run_model(self):
        ## function that iters through games df and runs model ##
        ## set a start epoch time ##
        start_time = time.time()
        ## clear out any existing values ##
        self.qbs = {} ## storage for most recent QB Data
        self.teams = {} ## storage for most recent team data
        self.data = [] ## storage for all game records ##
        self.league_avg_def = self.config['init_value'] ## each week, we need to calculate the league avg def for adjs ##
        self.current_week = 1 ## track the current week to know when it has changed ##
        ## iterate through games df ##
        for index, row in self.games.iterrows():
            ## get qb value ##
            qb_val = self.get_qb_value(row)
            ## get team def value ##
            team_def_val, team_def_adj = self.get_team_def_value(row['opponent'], row['week'])
            ## get team off value ##
            team_off_val, team_off_adj = self.get_team_off_value(row['team'], qb_val, row['season'])
            ## calc qb adj ##
            qb_adj = qb_val - team_off_val
            ## calc adjusted game value
            adj_val = row['player_VALUE'] - team_def_adj
            ## update qb value ##
            self.update_qb_value(row['player_id'], adj_val, row['gameday'], row['season'])
            ## update team def value ##
            self.update_team_def_value(row['opponent'], row['player_VALUE'])
            ## update team off value ##
            self.update_team_off_value(row['team'], adj_val)
            ## add all values to the row ##
            row['qb_value_pre'] = qb_val
            row['team_value_pre'] = team_off_val
            row['qb_adj'] = qb_adj
            row['opponent_def_value_pre'] = team_def_val
            row['opponent_def_adj'] = team_def_adj
            row['player_VALUE_adj'] = adj_val
            row['qb_value_post'] = self.qbs[row['player_id']]['current_value']
            row['team_value_post'] = self.teams[row['team']]['off_value']
            row['opponent_def_value_post'] = self.teams[row['opponent']]['def_value']
            ## write row to data ##
            self.data.append(row)
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
        df['se'] = (df['qb_value_pre'] - df['player_VALUE_adj']) ** 2
        df['abs_error'] = numpy.absolute(df['qb_value_pre'] - df['player_VALUE_adj'])
        ## only look at data past first season ##
        ## this is to give model time to catch up since we are starting in 1999 ##
        ## and veteran QBs are treated like rookies in that season ##
        df = df[df['season'] >= first_season].copy()
        ## copy config to serve as a record of what was used ##
        record = self.config.copy()
        ## add rmse and mae to record ##
        record['rmse'] = df['se'].mean() ** 0.5
        record['mae'] = df['abs_error'].mean()
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
