## built in ##
import pathlib
import json
import os

## external ##
import pandas as pd
import numpy

class MetaConstructor():
    '''
    Creates a df that combines meta data with id mappings across historic 538/nfelo names
    and gsis ids where available
    '''
    def __init__(self,
        players: pd.DataFrame, ## player data from DataLoader
        elo_file: pd.DataFrame, ## new_file from a constructor that has run construct_elo_file()
    ):
        self.players = players
        self.elo_file = elo_file
        ## additional ##
        self.package_loc = pathlib.Path(__file__).parent.parent.parent.resolve()
        self.repl = json.load(open('{0}/nfeloqb/Manual Data/name_id_repl.json'.format(self.package_loc)))
        self.missing_draft_data = pd.read_csv(
            '{0}/nfeloqb/Manual Data/missing_draft_data.csv'.format(self.package_loc),
            index_col=0
        )
        self.gen_file()
        
    def get_538_qbs(self):
        '''
        Creates a list of all qbs that are in the elo file, replaceing where necessary
        to ensure downstream merge
        '''
        flat_df = pd.concat([
            self.elo_file[['qb1']].rename(columns={'qb1' : 'name_id'}).copy(),
            self.elo_file[['qb2']].rename(columns={'qb2' : 'name_id'}).copy()
        ])
        flat_df = flat_df[~pd.isnull(flat_df['name_id'])]
        ## change the name id for mapping ##
        flat_df['name_id'] = flat_df['name_id'].replace(self.repl['elo_repl'])
        flat_df = flat_df.drop_duplicates()
        ## return ##
        return flat_df
    
    def get_fastr_qbs(self):
        '''
        Gets fastr qbs and filters down to relevant columns. Applies mapping
        '''
        ## isolate qbs ##
        qbs = self.players[
            ## QBs ##
            (self.players['position'] == 'QB') |
            ## Or players who used to be qbs and no longer satisfy
            ## the position requirement ##
            (numpy.isin(
                self.players['display_name'],
                [
                    'Tim Tebow', 'Terrelle Pryor', 'Taysom Hill',
                    'Kendall Hinton'
                ]
            ))
        ].copy()
        ## filter to essential fields based on drizzle schema requirements ##
        qbs = qbs[[
            'gsis_id',
            'display_name',
            'football_name',
            'first_name',
            'last_name',
            'short_name',
            'status', 
            'birth_date', 
            'college_name', 
            'height',
            'weight',
            'entry_year',
            'rookie_year',
            'draft_number',
            'draft_round',
            'draft_club',
            'headshot'
        ]].rename(columns={
            'college_name': 'college',
            'draft_club': 'draft_team',
            'headshot': 'headshot_url'
        }).copy()
        ## perform replacement on name ##
        qbs['display_name'] = qbs['display_name'].replace(self.repl['fastr_repl'])
        qbs = qbs.groupby(['gsis_id']).head(1)
        ## return ##
        return qbs

    def add_missing_draft_data(self, df):
        '''
        Adds missing draft data to the fastr df
        '''
        ## load missing draft data ##
        missing_draft = pd.read_csv(
            '{0}/nfeloqb/Manual Data/missing_draft_data.csv'.format(self.package_loc),
            index_col=0
        )
        ## avoid dupes ##
        missing_draft = missing_draft.groupby(['player_id']).head(1)
        ## add missing draft data ##
        df = pd.merge(
            df,
            missing_draft[[
                'player_id',
                'rookie_year',
                'draft_number',
                'entry_year',
                'birth_date'
            ]].rename(columns={
                'player_id' : 'gsis_id',
                'rookie_year' : 'rookie_year_fill',
                'draft_number' : 'draft_number_fill',
                'entry_year' : 'entry_year_fill',
                'birth_date' : 'birth_date_fill'
            }),
            on='gsis_id',
            how='left'
        )
        ## fill in missing data ##
        for col in [
            'rookie_year', 'draft_number', 'entry_year', 'birth_date'
        ]:
            df[col] = df[col].fillna(df[f'{col}_fill'])
        ## drop fill cols ##
        df = df.drop(columns=['rookie_year_fill', 'draft_number_fill', 'entry_year_fill', 'birth_date_fill'])
        ## change birth date to dob ##
        df = df.rename(columns={
            'birth_date' : 'dob'
        })
        ## return ##
        return df
    
    def add_manual_data(self, df):
        '''
        Adds manual data to the condensed df
        '''
        ## load manual data ##
        ## check that its there ##
        if not os.path.exists('{0}/nfeloqb/Manual Data/manual_data.csv'.format(self.package_loc)):
            return df
        ## and that it is not empty ##
        manual_data = pd.read_csv(
            '{0}/nfeloqb/Manual Data/manual_data.csv'.format(self.package_loc),
            index_col=0
        )
        if len(manual_data) == 0:
            return df
        ## prep manual for merge ##
        fill_cols = [col for col in manual_data.columns if col != 'name_id']
        manual_data = manual_data.rename(columns={
            col : f'{col}_fill' for col in fill_cols
        })
        ## merge ##
        df = pd.merge(
            df,
            manual_data[['name_id'] + fill_cols],
            on='name_id',
            how='left'
        )
        ## fill in missing data ##
        for col in fill_cols:
            df[col] = df[col].fillna(df[f'{col}_fill'])
        ## drop fill cols ##
        df = df.drop(columns=fill_cols)
        ## return ##
        return df
    
    def gen_file(self):
        '''
        Calls the methods above to generate the final file
        '''
        ## get 538 qbs ##
        qb_elo = self.get_538_qbs()
        ## get fastr qbs ##
        qbs_fastr = self.get_fastr_qbs()
        ## add missing draft data ##
        qbs_fastr = self.add_missing_draft_data(qbs_fastr)
        ## merge on name ##
        qbs_fastr['name_id'] = qbs_fastr['display_name']
        df = pd.merge(
            qb_elo,
            qbs_fastr,
            on='name_id',
            how='outer'
        )
        ## note misses ##
        elo_misses = len(df[df['gsis_id'].isna()])
        fastr_misses = len(df[df['name_id'].isna()])
        duplicate_gsis = len(df[
            (~pd.isnull(df['gsis_id'])) &
            (df['gsis_id'].duplicated())
        ])
        if elo_misses > 0:
            print(f'WARN: {elo_misses} elo qbs not found in fastr:')
            if elo_misses > 324:
                print('     This is more than expected! Check the elo file.')
        if fastr_misses > 0:
            print(f'ERROR: {fastr_misses} fastr qbs not found in elo')
            for name in df[df['name_id'].isna()]['display_name']:
                print(f'     {name}')
        if duplicate_gsis > 0:
            print(f'ERROR: {duplicate_gsis} duplicate gsis ids in meta data')
            for name_id in df[
                (df['gsis_id'].duplicated()) &
                (~pd.isnull(df['gsis_id']))
            ]['name_id']:
                print(f'     {name_id}')
        ## add manual data ##
        df = self.add_manual_data(df)
        df = df.sort_values(by=['name_id']).reset_index(drop=True)
        ## final clean up of edge cases ##
        df['name_id'] = numpy.where(
            df['gsis_id'] == '00-0035723',
            'Vincent Testaverde',
            df['name_id']
        )
        ## save ##
        df.to_csv(
            '{0}/Other Data/meta_data.csv'.format(self.package_loc)
        )

