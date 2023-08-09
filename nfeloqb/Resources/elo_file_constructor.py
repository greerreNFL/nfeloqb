import pandas as pd
import numpy

class EloConstructor():
    ## this class takes in the updated model data, next weeks games, and ##
    ## the original elo file to create a new elo file in the same format ##
    def __init__(self, games, qb_model, at_wrapper, export_loc):
        self.games = games.copy()
        self.qb_model = qb_model ## an updated QBModel Class object ##
        self.at_wrapper = at_wrapper ## an updated AirtableWrapper Class object ##
        self.export_loc = export_loc ## location to export new file ##
        self.qb_values = pd.DataFrame(qb_model.data)
        self.original_elo_file = pd.read_csv(self.qb_model.original_file_loc, index_col=0) ## original elo file ##
        self.original_elo_cols = self.original_elo_file.columns.to_list()
        self.new_games = None ## games that occured after original file ##
        self.next_games = None ## next weeks games ##
        self.new_file_games = None ## merged new games and next games ##
        self.new_file_data = [] ## formatted rows to be appended to the existing ##
        self.new_file = None
        
    def determine_new_games(self):
        ## could do dynamically, but just assume anything after 2023-02-12 is new ##
        ## this df represents all played games since original elo file ##
        ## was last updated ##
        self.new_games = self.games[
            (self.games['gameday'] > '2023-02-12') &
            (~pd.isnull(self.games['result']))
        ].copy()
        if len(self.new_games) == 0:
            self.new_games = None
        
    def add_qbs_to_new_games(self):
        ## combine model_df, which is flat, with new games ##
        ## elo file is not flat ##
        ## if new games is none, update ##
        if self.new_games is None:
            self.determine_new_games()
        ## if there have been no new games, return without updating ##
        if self.new_games is None:
            return
        ## add home qb ##
        self.new_games = pd.merge(
            self.new_games,
            self.qb_values[[
                'game_id', 'team', 'player_id', 'player_display_name',
                'qb_value_pre', 'qb_adj', 'player_VALUE_adj', 'qb_value_post'
            ]].rename(columns={
                'team' : 'home_team',
                'player_id' : 'qb1_id',
                'player_display_name' : 'qb1',
                'qb_value_pre' : 'qb1_value_pre',
                'qb_adj' : 'qb1_adj',
                'player_VALUE_adj' : 'qb1_game_value',
                'qb_value_post' : 'qb1_value_post'
            }),
            on=['game_id', 'home_team'],
            how='left'
        ) 
        ## add away qb ##
        self.new_games = pd.merge(
            self.new_games,
            self.qb_values[[
                'game_id', 'team', 'player_id', 'player_display_name',
                'qb_value_pre', 'qb_adj', 'player_VALUE_adj', 'qb_value_post'
            ]].rename(columns={
                'team' : 'away_team',
                'player_id' : 'qb2_id',
                'player_display_name' : 'qb2',
                'qb_value_pre' : 'qb2_value_pre',
                'qb_adj' : 'qb2_adj',
                'player_VALUE_adj' : 'qb2_game_value',
                'qb_value_post' : 'qb2_value_post'
            }),
            on=['game_id', 'team'],
            how='left'
        )
    
    def get_next_games(self):
        ## determine the next week of games ##
        unplayed = self.games[
            (pd.isnull(self.games['result']))
        ].copy()
        ## if there are no games, b/c the season is over, stop ##
        if len(unplayed) == 0:
            return None
        ## if there is a next week, filter games ##
        self.next_games = self.games[
            (self.games['season'] == unplayed.iloc[0]['season']) &
            (self.games['week'] == unplayed.iloc[0]['week'])
        ].copy()
    
    def extract_starter_values(self, qb_id, season, team, draft_number, gameday):
        ## helper function that pulls starters current value from the model ##
        ## and does necessary regression ##
        ## first create a random id if qb_id is null, which can happen if QB is not in roster ##
        ## file yet ##
        if pd.isnull(qb_id):
            qb_id = '00-' + str(numpy.random.randint(100000,200000))
        ## create a dummy 'row' so we can use get_qb_value function from model ##
        row = {
            'player_id' : qb_id,
            'season' : season,
            'team' : team,
            'draft_number' : draft_number,
            'gameday' : gameday
        }
        ## get the qb value ##
        qb_value = self.qb_model.get_qb_value(row)
        ## return the value ##
        return qb_value
    
    def add_starters(self):
        ## add starters and values to the new games ##
        ## this also needs to handle season over season regressions ##
        ## update starters ##
        self.at_wrapper.pull_current_starters()
        ## convert starters to a dict ##
        starter_dict = {}
        for index, row in self.at_wrapper.starters_df.iterrows():
            starter_dict[row['team']] = {}
            starter_dict[row['team']]['qb_id'] = row['player_id']
            starter_dict[row['team']]['qb_name'] = row['player_display_name']
            starter_dict[row['team']]['draft_number'] = row['draft_number']
        ## helper func to apply to new games ##
        def apply_starters(row, starter_dict):
            ## home ##
            row['qb1_id'] = starter_dict[row['home_team']]['qb_id']
            row['qb1'] = starter_dict[row['home_team']]['qb_name']
            row['qb1_value_pre'] = self.extract_starter_values(
                row['qb1_id'], row['season'], row['home_team'],
                starter_dict[row['home_team']]['draft_number'], row['gameday']
            )
            row['qb1_game_value'] = numpy.nan
            row['qb1_value_post'] = numpy.nan
            ## away ##
            row['qb2_id'] = starter_dict[row['away_team']]['qb_id']
            row['qb2'] = starter_dict[row['away_team']]['qb_name']
            row['qb2_value_pre'] = self.extract_starter_values(
                row['qb2_id'], row['season'], row['away_team'],
                starter_dict[row['away_team']]['draft_number'], row['gameday']
            )
            row['qb2_game_value'] = numpy.nan
            row['qb2_value_post'] = numpy.nan
            ## return ##
            return row
        ## apply ##
        self.next_games = self.next_games.apply(
            apply_starters,
            starter_dict=starter_dict,
            axis=1
        )
    
    def add_team_values(self):
        ## once next week has been updated with starter values, add team values ##
        ## and make adjustments ##
        ## first set the models current week to the week of next games ##
        self.qb_model.current_week = self.next_games.iloc[0]['week']
        ## helper to add team values ##
        def apply_team_values(row):
            ## home ##
            home_val, home_adj = self.qb_model.get_team_off_value(
                row['home_team'], row['qb1_value_pre'], row['season']
            )
            ## away ##
            away_val, away_adj = self.qb_model.get_team_off_value(
                row['away_team'], row['qb2_value_pre'], row['season']
            )
            ## add adjs to row ##
            row['qb1_adj'] = home_adj
            row['qb2_adj'] = away_adj
            ## return ##
            return row
        ## apply ##
        self.next_games = self.next_games.apply(
            apply_team_values,
            axis=1
        )
    
    def merge_new_and_next(self):
        ## merge new games and next games with logic to handle blanks ##
        if self.new_games is None:
            if self.next_games is None:
                ## if both are none, return none ##
                return
            else:
                self.new_file_games = self.next_games
        else:
            if self.next_games is None:
                self.new_file_games = self.new_games
            else:
                ## merge ##
                ## align columns ##
                self.new_file_games = pd.concat([
                    self.new_games,
                    self.next_games[
                        self.new_games.columns
                    ]
                ])
    
    def format_games_row(self, row):
        ## takes a row from model_df and formats it for the elo file ##
        new_row = {}
        for col in self.original_elo_cols:
            new_row[col] = numpy.nan
        ## add in the values ##
        new_row['date'] = row['gameday']
        new_row['season'] = row['season']
        new_row['team1'] = row['home_team']
        new_row['team2'] = row['away_team']
        new_row['score1'] = row['home_score']
        new_row['score2'] = row['away_score']
        ## qb values ##
        ## each qb value is in VALUE, but needs to be in elo, so multiply by 3.3 ##
        new_row['qb1'] = row['qb1']
        new_row['qb2'] = row['qb2']
        new_row['qb1_value_pre'] = row['qb1_value_pre'] * 3.3
        new_row['qb2_value_pre'] = row['qb2_value_pre'] * 3.3
        new_row['qb1_value_post'] = row['qb1_value_post'] * 3.3
        new_row['qb2_value_post'] = row['qb2_value_post'] * 3.3
        new_row['qb1_adj'] = row['qb1_adj'] * 3.3
        new_row['qb2_adj'] = row['qb2_adj'] * 3.3
        new_row['qb1_game_value'] = row['qb1_game_value'] * 3.3
        new_row['qb2_game_value'] = row['qb2_game_value'] * 3.3
        ## netural locs ##
        if row['location'] == 'Home':
            new_row['neutral'] = 0
        else:
            new_row['neutral'] = 1
        ## playoffs ##
        if row['game_type'] == 'REG':
            new_row['playoff'] = numpy.nan
        else:
            new_row['playoff'] = 1
        ## write row to new file data ##
        self.new_file_data.append(new_row)
    
    def create_new_file(self):
        ## merges original elo file with new games and next games and then saves ##
        ## to the root of the package ##
        if self.new_file_games is None:
            self.new_file = self.original_elo_file
        else:
            ## sort games ##
            self.new_file_games = self.new_file_games.sort_values(
                by=['season', 'week', 'gameday'],
                ascending=[True, True, True]
            ).reset_index(drop=True)
            ## parse rows ##
            for index, row in self.new_file_games.iterrows():
                self.format_games_row(row)
            ## create new df and concat ##
            self.new_file = pd.concat([
                self.original_elo_file,
                pd.DataFrame(self.new_file_data)
            ])
            self.new_file = self.new_file.reset_index(drop=True)
    
    def construct_elo_file(self):
        ## wrapper on the above functions that creates the elo file ##
        print('Constructing elo file...')
        print('     Determining new games...')
        self.determine_new_games()
        if self.new_games is None:
            print('     No new games found')
        else:
            print('          Found {0} new games. Adding QB data...'.format(len(self.new_games)))
            self.add_qbs_to_new_games()
        print('     Determining next games...')
        self.get_next_games()
        if self.next_games is None:
            print('     No next games found')
        else:
            print('          Found {0} next games. Pulling projected starters...'.format(len(self.next_games)))
            self.add_starters()
            print('          Adding team values for adjustments...')
            self.add_team_values()
        print('     Merging new and next games...')
        self.merge_new_and_next()
        if self.new_file_games is None:
            print('     No new games to merge to original elo file. Will not update')
        else:
            print('     Formatting games for elo file...')
            self.create_new_file()
            print('     Saving new elo file...')
            self.new_file.to_csv(
                '{0}/qb_elos.csv'.format(self.export_loc),
                index=False
            )
            print('     Done')
