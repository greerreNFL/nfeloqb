import pandas as pd
import numpy
import pathlib

class Elo():
    ## class for calculating the 538 elo models ##
    def __init__(self, game, qb_df, base=1505, k=20, b=400, reg=.33, reg_vegas=.66, hfa_roll=10, rest=25, no_fan=33, playoff=1.2):
        ## data ##
        self.game = game.copy()
        self.qb_df = qb_df.copy()
        data_folder = pathlib.Path(__file__).parent.parent.resolve()
        self.wt_ratings = pd.read_csv('{0}/Manual Data/wt_ratings.csv'.format(data_folder))
        ## model variables ##
        self.base = base
        self.k = k
        self.b = b
        self.reg = reg
        self.reg_vegas = reg_vegas
        self.hfa_roll = hfa_roll
        self.rest = rest
        self.no_fan = no_fan
        self.playoff = playoff
        ## storage ##
        self.current_elos = {}
        self.elo_records = []
        self.elo_df = None
        ## states ##
        self.current_season = None
        self.current_week = None
        self.current_hfa = None
        ## init ##
        self.format_games()
        self.add_qbs_to_new_games()
        self.init_elos()
                
    def format_games(self):
        ## add fields and HFA to games ##
        hfa = self.game.groupby(['season']).agg(
            avg_home_mov=('result', 'mean')
        ).reset_index()
        ## calc 10 year rolling avg shifted back one year ##
        hfa['avg_home_mov'] = hfa['avg_home_mov'].rolling(self.hfa_roll).mean().shift(1)
        ## fill missing w/ 2.5 ##
        hfa['avg_home_mov'] = hfa['avg_home_mov'].fillna(2.5)
        ## translate to an elo ##
        hfa['hfa'] = hfa['avg_home_mov'] * 25
        ## write last HFA to state ##
        self.current_hfa = hfa['hfa'].iloc[-1]
        ## reduce 2020 by 33 ##
        hfa['hfa'] = numpy.where(
            hfa['season'] == 2020,
            hfa['hfa'] - self.no_fan,
            hfa['hfa']
        )
        ## add to games ##
        self.game = pd.merge(
            self.game,
            hfa[['season', 'hfa']],
            on=['season'],
            how='left'
        )
        ## correct for neutrals ##
        self.game['hfa'] = numpy.where(
            self.game['location'] != 'Home',
            0,
            self.game['hfa']
        )
        ## create byes ##
        self.game['home_bye'] = numpy.where(
            self.game['home_rest'] >= 10,
            1,
            0
        )
        self.game['away_bye'] = numpy.where(
            self.game['away_rest'] >= 10,
            1,
            0
        )
        ## create by dif ##
        self.game['rest_dif'] = (self.game['home_bye'] - self.game['away_bye']) * self.rest
    
    def add_qbs_to_new_games(self):
        ## add qb adjs to the game file ##
        ## add home qb ##
        self.game = pd.merge(
            self.game,
            self.qb_df[[
                'game_id', 'team', 'qb_adj'
            ]].rename(columns={
                'team' : 'home_team',
                'qb_adj' : 'home_qb_elo_adj'
            }),
            on=['game_id', 'home_team'],
            how='left'
        )
        ## add away qb ##
        self.game = pd.merge(
            self.game,
            self.qb_df[[
                'game_id', 'team', 'qb_adj'
            ]].rename(columns={
                'team' : 'away_team',
                'qb_adj' : 'away_qb_elo_adj'
            }),
            on=['game_id', 'away_team'],
            how='left'
        )
        ## gross up to get elos ##
        self.game['home_qb_elo_adj'] = self.game['home_qb_elo_adj'].fillna(0) * 3.3
        self.game['away_qb_elo_adj'] = self.game['away_qb_elo_adj'].fillna(0) * 3.3
        ## add adj ##
        self.game['qb_adj'] = self.game['home_qb_elo_adj'] + self.game['away_qb_elo_adj']
    
    def init_elos(self):
        ## initialize the current elo dict ##
        teams = self.game['home_team'].unique()
        for team in teams:
            self.current_elos[team] = self.base
        ## init states ##
        self.current_season = self.game['season'].min()
        self.current_week = self.game[self.game['season']==self.current_season]['week'].min()
        
    def handle_regression(self):
        ## function that updates all offseason elos and incriments current season ##
        ## incriment seaosn ##
        self.current_season += 1
        ## loop through dict ##
        for team, last_elo in self.current_elos.items():
            ## regress to mean ##
            new_elo = (last_elo * (1-self.reg)) + (self.base * self.reg)
            ## if get vegas ratings ##
            temp = self.wt_ratings[
                (self.wt_ratings['team']==team) &
                (self.wt_ratings['season']==self.current_season)
            ].copy()
            if len(temp) > 0:
                new_elo = (
                    (new_elo * (1-self.reg_vegas)) +
                    (temp.iloc[0]['wt_rating_elo'] * self.reg_vegas)
                )
            ## update dict ##
            self.current_elos[team] = new_elo
        
    def calc_elo_difs(self, record):
        ## function that calcualtes the elo difs for a given record ##
        ## get current elos ##
        home_elo = self.current_elos[record['home_team']]
        away_elo = self.current_elos[record['away_team']]
        ## calc dif ##
        elo_dif = home_elo - away_elo + record['hfa'] + record['qb_adj'] + record['rest_dif']
        elo_dif_ex_qb = home_elo - away_elo + record['hfa'] + record['rest_dif']
        ## calc probs ##
        home_prob = 1 / (10 ** (-elo_dif/self.b) + 1)
        away_prob = 1 - home_prob
        home_prob_ex_qb = 1 / (10 ** (-elo_dif_ex_qb/self.b) + 1)
        away_prob_ex_qb = 1 - home_prob_ex_qb
        ## return ##
        return home_elo, away_elo, home_prob, away_prob, elo_dif, home_prob_ex_qb, away_prob_ex_qb
        
    def update_elos(self, record, home_elo, away_elo, home_prob, away_prob, elo_dif):
        ## function that updates the elos for a given record ##
        ## create result variables
        mov_elo_dif = elo_dif
        away_result = 0
        home_result = 0
        if record['result'] > 0:
            home_result = 1
            away_result = 0
        elif record['result'] == 0:
            home_result = .5
            away_result = .5
        else:
            home_result = 0
            away_result = 1
            mov_elo_dif = -elo_dif
        ## create movmultiplier ##
        ## LN(ABS(PD)+1) * (2.2/((ELOW-ELOL)*.001+2.2))
        mov_mult = (
            numpy.log(abs(record['result']) + 1) *
            (2.2/(mov_elo_dif * .001 + 2.2))
        )
        ## calc new elos ##
        new_home_elo = home_elo + ((self.k * (home_result - home_prob)) * mov_mult)
        new_away_elo = away_elo + ((self.k * (away_result - away_prob)) * mov_mult)
        ## update dict ##
        self.current_elos[record['home_team']] = new_home_elo
        self.current_elos[record['away_team']] = new_away_elo
        ## return ##
        return new_home_elo, new_away_elo

    def handle_game(self, record):
        ## function that handles a single game and returns an elo record ##
        ## first check if regression is needed ##
        if record['season'] > self.current_season:
            self.handle_regression()
        ## get elos and probs ##
        home_elo, away_elo, home_prob, away_prob, elo_dif, home_prob_ex_qb, away_prob_ex_qb = self.calc_elo_difs(record)
        ## if the game has been played, update elos ##
        if pd.isnull(record['result']):
            new_home_elo = numpy.nan
            new_away_elo = numpy.nan
        else:
            new_home_elo, new_away_elo = self.update_elos(
                record, home_elo, away_elo, home_prob, away_prob, elo_dif
            )
        ## return ##
        self.elo_records.append({
            'game_id': record['game_id'],
            'elo1_pre': home_elo,
            'elo2_pre': away_elo,
            'elo_prob1': home_prob_ex_qb,
            'elo_prob2': away_prob_ex_qb,
            'elo1_post': new_home_elo,
            'elo2_post': new_away_elo,
            'qbelo1_pre': home_elo + record['home_qb_elo_adj'],
            'qbelo2_pre': away_elo + record['away_qb_elo_adj'],
            'qbelo_prob1': home_prob,
            'qbelo_prob2': away_prob,
            'qbelo1_post': new_home_elo + record['home_qb_elo_adj'],
            'qbelo2_post': new_away_elo + record['away_qb_elo_adj']
        })

    def run(self):
        ## function that runs the entire model ##
        ## loop through games ##
        for index, record in self.game.iterrows():
            self.handle_game(record)
        ## return ##
        self.elo_df = pd.DataFrame(self.elo_records)
