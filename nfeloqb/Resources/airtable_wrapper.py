import pandas as pd
import numpy
import requests
import time
import json
import math

class AirtableWrapper():
    ## This class is handles IO for an airtable base that stores
    ## starters for the current week ##
    def __init__(self, model_df, at_config, perform_starter_update=True):
        self.model_df = model_df ## df of qbs and their meta data ##
        self.at_config = at_config ## config for airtable including token, ids, etc ##
        ## unpack config ##
        self.base = self.at_config['base']
        self.qb_table = self.at_config['qb_table']
        self.starter_table = self.at_config['starter_table']
        self.token = self.at_config['token']
        self.qb_fields = self.at_config['qb_fields']
        self.dropdown_field_id = self.at_config['dropdown_field']
        self.base_headers = {
            'Authorization' : 'Bearer {0}'.format(self.token),
            'Content-Type' : 'application/json'
        }
        ## storage for various data sets and vars ##
        self.existing_qbs = None ## qbs already written to db ##
        self.existing_qb_options = None ## qb options in dropdown ##
        self.existing_starters = None ## list of existing starters in AT ##
        self.starters_df = None ## starters in AT, but in df format ##
        self.all_qbs = None ## all qbs in model_df
        self.perform_starter_update = perform_starter_update ## if True, update starters ##

    ## api wrapper functions ##
    def make_post_request(self, base, table, headers, data):
        ## used for creating new records ##
        ## rate limiting ##
        time.sleep(1/4)
        ## formulate url ##
        url = 'https://api.airtable.com/v0/{0}/{1}'.format(base, table)
        requests.post(
            url,
            headers=headers,
            data=json.dumps(data)
        )
    
    def make_patch_request(self, base, table, headers, data):
        ## Used for updating existing records ##
        ## rate limiting ##
        time.sleep(1/4)
        ## formulate url ##
        url = 'https://api.airtable.com/v0/{0}/{1}'.format(base, table)
        resp = requests.patch(
            url,
            headers=headers,
            data=json.dumps(data)
        )
        if resp.status_code != 200:
            print('Error on patch! -- {0} -- {1}'.format(
                resp.status_code,
                resp.content
            ))
    
    def make_get_request(self, base, table, headers, params):
        ## used to for getting records ##
        ## rate limiting ##
        time.sleep(1/4)
        ## formulate url ##
        url = 'https://api.airtable.com/v0/{0}/{1}'.format(base, table)
        resp = requests.get(
            url,
            headers=headers,
            params=params
        )
        return resp
    
    def make_delete_request(self, base, table, headers, params):
        ## used for deleting records ##
        ## rate limiting ##
        time.sleep(1/4)
        ## formulate url ##
        url = 'https://api.airtable.com/v0/{0}/{1}'.format(base, table)
        requests.delete(
            url,
            headers=headers,
            params=params
        )
    
    def make_meta_request(self, base, headers):
        ## request schema of base ##
        ## rate limiting ##
        time.sleep(1/4)
        ## formulate url ##
        url = 'https://api.airtable.com/v0/meta/bases/{0}/tables'.format(base)
        resp = requests.get(
            url,
            headers=headers
        )
        return resp.json()
    
    def make_paginated_get(self, base, table, headers, params):
        ## make first request ##
        all_records = []
        resp = self.make_get_request(base, table, headers, params)
        records = resp.json()
        ## add records to container for initial pull ##
        for record in records['records']:
            all_records.append(record)
        ## init var loops ##
        if 'offset' in records.keys():
            offset = records['offset']
            loops = 0
        else:
            offset = None ## if no offset, no need to paginate ##
            loops = 0
        ## loop ##
        while offset is not None and loops < 50:
            params['offset'] = offset
            resp = self.make_get_request(base, table, headers, params)
            records = resp.json()
            ## add records to container for initial pull ##
            for record in records['records']:
                all_records.append(record)
            ## update var loops ##
            if 'offset' in records.keys():
                offset = records['offset']
                loops += 1
            else:
                offset = None ## if no offset, no need to paginate ##
                loops += 1
        ## return data ##
        return all_records
    
    def data_format(self, datapoint):
        ## translates a NaN to None for airtable ##
        if pd.isnull(datapoint):
            return None
        else:
            return datapoint
    
    def write_chunk(self, base, table, df):
        ## write chunk to airtable ##
        ## container for data to write to airtable ##
        data = {
            'records' : [],
            'typecast' : True
        }
        ## get table cols ##
        table_cols = df.columns.values.tolist()
        ## iterate through chunk and add to data ##
        for index, row in df.iterrows():
            record = {
                'fields' : {}
            }
            for col in table_cols:
                record['fields'][col] = self.data_format(row[col])
            ## append to date ##
            data['records'].append(record)
        ## write to table ##
        self.make_post_request(
            base=base,
            table=table,
            headers=self.base_headers,
            data=data
        )
        
    ## write chunk to airtable ##
    def update_chunk(self, base, table, df, id_col):
        ## container for data to write to airtable ##
        data = {
            'records' : [],
            'typecast' : True
        }
        ## get table cols ##
        table_cols = df.columns.values.tolist()
        ## iterate through chunk and add to data ##
        for index, row in df.iterrows():
            record = {
                'id' : row[id_col],
                'fields' : {}
            }
            for col in table_cols:
                if col == id_col:
                    pass
                else:
                    record['fields'][col] = self.data_format(row[col])
            ## append to date ##
            data['records'].append(record)
        ## write to table ##
        self.make_patch_request(
            base=base,
            table=table,
            headers=self.base_headers,
            data=data
        )
        
    ## perform upsert to airtable ##
    def upsert_chunk(self, base, table, df, upsertFields, key):
        ## container for data to write to airtable ##
        data = {
            'records' : [],
            'performUpsert' : {
                'fieldsToMergeOn' : upsertFields
            }
        }
        ## get table cols ##
        table_cols = df.columns.values.tolist()
        ## control for missing fields ##
        for field in upsertFields:
            if field not in table_cols:
                print('     {0} is not included in data. Upsert will fail...'.format(
                    field
                ))
        ## iterate through chunk and add to data ##
        for index, row in df.iterrows():
            record = {
                'fields' : {}
            }
            for col in table_cols:
                record['fields'][col] = self.data_format(row[col])
            ## append to date ##
            data['records'].append(record)
        ## write to table ##
        self.make_patch_request(
            base=base,
            table=table,
            headers={
                'Authorization' : 'Bearer {0}'.format(key),
                'Content-Type' : 'application/json'
            },
            data=data
        )
        
    ## break df into chunks of 10 and write to airtable ##
    def write_table(self, base, table, df):
        ## break df into chunks of 10 ##
        ## determine size of df ##
        df_len = len(df)
        chunks_needed = math.ceil(df_len / 10)
        ## split ##
        df_chunks = numpy.array_split(df, chunks_needed)
        ## write ##
        for chunk in df_chunks:
            ## turn chunk into record ##
            self.write_chunk(base, table, chunk)
    
    ## break df into chunks of 10 and write to airtable ##
    def update_table(self, base, table, df, id_col):
        ## break df into chunks of 10 ##
        ## determine size of df ##
        df_len = len(df)
        chunks_needed = math.ceil(df_len / 10)
        ## split ##
        df_chunks = numpy.array_split(df, chunks_needed)
        ## write ##
        for chunk in df_chunks:
            ## turn chunk into record ##
            self.update_chunk(base, table, chunk, id_col)
    
    ## fucntional abstractions for wrapper ##
    def get_existing_qbs(self):
        ## gets existing QBs from airtable ##
        ## get existing qbs ##
        qbs_resp = self.make_paginated_get(
            base=self.base,
            table=self.qb_table,
            headers=self.base_headers,
            params={
                ##'fields' : self.qb_fields
            }
        )
        ## container for qbs ##
        qbs = []
        ## iterate through qb response and add to container ##
        for qb in qbs_resp:
            qbs.append(qb['fields']['player_id'])
        ## return ##
        self.existing_qbs = qbs
    
    def get_qb_options(self):
        ## gets a list of QBs that are options in the drop down ##
        ## get base schema ##
        base_schema = self.make_meta_request(
            base=self.base,
            headers=self.base_headers,
        )
        ## parse ##
        options = []
        for table in base_schema['tables']:
            if table['id'] == self.starter_table:
                for field in table['fields']:
                    if field['id'] == self.dropdown_field_id:
                        for option in field['options']['choices']:
                            options.append(option['name'])
        ## return ##
        self.qb_options = options
    
    def get_starters(self):
        ## gets existing QBs from airtable ##
        ## get existing qbs ##
        qbs_resp = self.make_paginated_get(
            base=self.base,
            table=self.starter_table,
            headers=self.base_headers,
            params={
                ##'fields' : self.qb_fields
            }
        )
        ## structure for existing starters, which has a key of the team ##
        ## and values of record id and qb_id ##
        existing_starters = {}
        for record in qbs_resp:
            existing_starters[record['fields']['team']] = {
                'record_id' : record['id'],
                'qb_id' : record['fields']['qb_id']
            }
        ## write ##
        self.existing_starters = existing_starters
    
    def write_qbs(self, qbs_to_write):
        ## write a df containing qb meta to the qb db in airtable ##
        self.write_table(
            base=self.base,
            table=self.qb_table,
            df=qbs_to_write
        )
        
    def write_qb_options(self, qb_options_to_write):
        ## to update an option to the dropdown, you need to create a record ##
        ## with typecase set to true ##
        ## to do this, loop through new options. On the first, create a dummary record ##
        ## on subsequents records, upsert that record ##
        ## on the final, delete the dummy record ##
        ## container for dummy record id ##
        dummy_id = None
        for index, value in enumerate(qb_options_to_write):
            ## create record structure ##
            data = {
                'records' : [
                    {
                        'fields' : {
                            'team' : 'DUMMY',
                            'qb_id' : value
                        }
                    }
                ],
                'typecast' : True
            }
            if index == 0:
                ## if first record, create the dummy ##
                self.make_post_request(
                    base=self.base,
                    table=self.starter_table,
                    headers=self.base_headers,
                    data=data
                )
                ## retrieve record to get id ##
                resp = self.make_get_request(
                    base=self.base,
                    table=self.starter_table,
                    headers=self.base_headers,
                    params={
                        'filterByFormula' : 'team = "DUMMY"'
                    }
                )
                resp = resp.json()
                dummy_id = resp['records'][0]['id']
            else:
                ## update record with dummy id ##
                data['records'][0]['id'] = dummy_id
                ## make a patch request ##
                self.make_patch_request(
                    base=self.base,
                    table=self.starter_table,
                    headers=self.base_headers,
                    data=data
                )
            ## if last record, delete dummy ##
            if index == len(qb_options_to_write) - 1:
                r=self.make_delete_request(
                    base=self.base,
                    table=self.starter_table,
                    headers=self.base_headers,
                    params={
                        'records[]' : dummy_id
                    }
                )
    
    ## model functions ##
    def get_qbs(self):
        ## gets a unique set of QBs from the data file ##
        ## note, this only stores QBs that have made a start ##
        qbs = self.model_df.copy()
        ## get most recent ##
        qbs = qbs.sort_values(
            by=['gameday'],
            ascending=[False]
        ).reset_index(drop=True)
        ## add a field that combines id and display name ##
        qbs['qb_id'] = qbs['player_display_name'] + ' - ' + qbs['player_id']
        qbs = qbs[[
            'qb_id', 'player_id', 'player_display_name',
            'start_number', 'rookie_year', 'entry_year',
            'draft_number'
        ]].groupby(['player_id']).head(1)
        ## return ##
        self.all_qbs = qbs
        
    def get_last_starter(self):
        ## for each team, determines last starter, which is assumed ##
        ## to be the starter for the next week ##
        starters = self.model_df.copy()
        starters = starters.sort_values(
            by=['gameday'],
            ascending=[False]
        ).reset_index(drop=True)
        ## add a field that combines id and display name ##
        starters['qb_id'] = starters['player_display_name'] + ' - ' + starters['player_id']
        starters = starters[[
            'team', 'qb_id',
        ]].groupby(['team']).head(1)
        ## return ##
        return starters
    
    ## actual functions that get called ##
    def update_qb_table(self):
        ## checks qbs in airtable against qbs in data ##
        ## updates the delta ##
        print('Updating QB table...')
        ## get existing qbs ##
        self.get_existing_qbs()
        ## get qbs from data ##
        self.get_qbs()
        ## get delta ##
        delta = self.all_qbs[
            ~numpy.isin(
                self.all_qbs['player_id'],
                self.existing_qbs
            )
        ].copy()
        ## determine write ##
        if len(delta) > 0:
            print('     Found {0} new QBs'.format(len(delta)))
            ## write ##
            self.write_qbs(delta)
            ## update existing qbs so its accurate ##
            for qb in delta['player_id'].unique().tolist():
                self.existing_qbs.append(qb)
        else:
            print('     No new QBs needed')
    
    def update_qb_options(self):
        ## updates the QB option dropdown to reflect QBs in the ##
        ## database ##
        print('Updating QB options...')
        ## update existing options ##
        self.get_qb_options()
        ## determine all values that should be in dropdown ##
        delta = self.all_qbs[
            ~numpy.isin(
                self.all_qbs['qb_id'],
                self.qb_options
            )
        ].copy()
        ## determine write ##
        if len(delta) > 0:
            print('     Found {0} new QB options'.format(len(delta)))
            ## write ##
            self.write_qb_options(delta['qb_id'].unique().tolist())
        else:
            print('     No new QB options needed')
        
    def update_starters(self):
        if not self.perform_starter_update:
            return
        ## reads the starter table in airtable and determines ##
        ## if any starters are different from the previous week ##
        print('Updating starters...')
        ## get last week's starters from AT ##
        self.get_starters()
        existing_starters = self.existing_starters
        ## get this weeks starters from data #
        this_weeks_starters = self.get_last_starter()
        ## structure for holding updates ##
        writes = []
        updates = []
        ## loop through teams ##
        for index, row in this_weeks_starters.iterrows():
            ## get team ##
            team = row['team']
            if team in existing_starters:
                ## if team is in the AT table (it should be) check starter ##
                if existing_starters[team]['qb_id'] != row['qb_id']:
                    ## if starter is not match, create update rec ##
                    updates.append({
                        'id' : existing_starters[team]['record_id'],
                        'qb_id' : row['qb_id'],
                        ## airtable automations dont trigger on API update, so ##
                        ## zero out the fields so it's obvious they need to be updated ##
                        'start_number' : numpy.nan,
                        'rookie_year' : numpy.nan,
                        'entry_year' : numpy.nan,
                        'draft_number' : numpy.nan,
                        'player_display_name' : numpy.nan,
                        'player_id' : numpy.nan
                    })
            else:
                ## if team is not in the AT table, create write rec ##
                writes.append({
                    'team' : team,
                    'qb_id' : row['qb_id']
                })
        ## write if necessary ##
        if len(writes) > 0:
            print('     Found {0} new teams'.format(len(writes)))
            self.write_table(
                base=self.base,
                table=self.starter_table,
                df=pd.DataFrame(writes)
            )
        ## update if necessary ##
        if len(updates) > 0:
            print('     Found {0} updated starters'.format(len(updates)))
            self.update_table(
                base=self.base,
                table=self.starter_table,
                df=pd.DataFrame(updates),
                id_col='id'
            )
    
    def pull_current_starters(self):
        ## pulls the current starters from the airtable ##
        ## and stores as a DF for the elo constructor ##
        qbs_resp = self.make_paginated_get(
            base=self.base,
            table=self.starter_table,
            headers=self.base_headers,
            params={
                ##'fields' : self.qb_fields
            }
        )
        ## structure for existing starters, which has a key of the team ##
        ## and values of record id and qb_id ##
        starters_data = []
        for record in qbs_resp:
            ## control for missing ##
            for field in ['team', 'player_id', 'player_display_name', 'draft_number', 'last_updated']:
                if field not in record['fields']:
                    record['fields'][field] = numpy.nan
            starters_data.append({
                'team' : record['fields']['team'],
                'player_id' : record['fields']['player_id'],
                'player_display_name' : record['fields']['player_display_name'],
                'draft_number' : record['fields']['draft_number'],
                'last_updated' : record['fields']['last_updated']
            })
        ## write ##
        self.starters_df = pd.DataFrame(starters_data)
    
    def get_last_update(self):
        '''
        Returns the timestampe when the starters table was last updated
        '''
        ## get the starters ##
        self.pull_current_starters()
        starters = self.starters_df.copy()
        ## ensure time format ##
        starters['last_updated'] = pd.to_datetime(
            starters['last_updated'],
            errors='coerce',
            utc=True
        )
        ## sort ##
        starters = starters.sort_values(
            by=['last_updated'],
            ascending=[False]
        ).reset_index(drop=True)
        ## return most recent ##
        return starters.iloc[0]['last_updated']
