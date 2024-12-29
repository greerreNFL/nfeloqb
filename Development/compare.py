import pathlib

import pandas as pd
import numpy

import nfelodcm as dcm

def compare_qb_file(
    ext_file_path: str
):
    '''
    Comapres a QB file to the current QB file
    '''
    ## comparison columns ##
    comparison_cols = ['value_pre', 'value_post', 'adj', 'game_value']
    ## establish path ##
    root_loc = pathlib.Path(__file__).parent.parent.resolve()
    ## load files ##
    ext = pd.read_csv(ext_file_path)
    cur = pd.read_csv(
        '{0}/qb_elos.csv'.format(root_loc)
    )
    ## constrain to after the nfelo model took over ##
    ext = ext[ext['season'] >= 2023].copy()
    cur = cur[cur['season'] >= 2023].copy()
    ## flatten ##
    ext = pd.concat([
        ext[[
            'qb1','season','game_id',
            'qb1_value_pre', 'qb1_value_post',
            'qb1_adj', 'qb1_game_value'
        ]].rename(columns={
            'qb1' : 'qb',
            'qb1_value_pre' : 'value_pre',
            'qb1_value_post' : 'value_post',
            'qb1_adj' : 'adj',
            'qb1_game_value' : 'game_value'
        }),
        ext[[
            'qb2','season','game_id',
            'qb2_value_pre', 'qb2_value_post',
            'qb2_adj', 'qb2_game_value'
        ]].rename(columns={
            'qb2' : 'qb',
            'qb2_value_pre' : 'value_pre',
            'qb2_value_post' : 'value_post',
            'qb2_adj' : 'adj',
            'qb2_game_value' : 'game_value'
        })
    ])
    cur = pd.concat([
        cur[[
            'qb1','season','game_id',
            'qb1_value_pre', 'qb1_value_post',
            'qb1_adj', 'qb1_game_value'
        ]].rename(columns={
            'qb1' : 'qb',
            'qb1_value_pre' : 'value_pre',
            'qb1_value_post' : 'value_post',
            'qb1_adj' : 'adj',
            'qb1_game_value' : 'game_value'
        }),
        cur[[
            'qb2','season','game_id',
            'qb2_value_pre', 'qb2_value_post',
            'qb2_adj', 'qb2_game_value'
        ]].rename(columns={
            'qb2' : 'qb',
            'qb2_value_pre' : 'value_pre',
            'qb2_value_post' : 'value_post',
            'qb2_adj' : 'adj',
            'qb2_game_value' : 'game_value'
        })
    ])
    ## rename ##
    for col in comparison_cols:
        ext = ext.rename(columns={col : '{0}_ext'.format(col)})
        cur = cur.rename(columns={col : '{0}_cur'.format(col)})
    ## merge ##
    merged = pd.merge(
        ext,
        cur,
        on=['qb', 'season', 'game_id'],
        how='left'
    )
    ## add start number and season start number ##
    merged['start_number'] = merged.groupby(['qb']).cumcount() + 1
    merged['season_start_number'] = merged.groupby(['qb', 'season']).cumcount() + 1
    ## calc diffs ##
    for col in comparison_cols:
        merged['{0}_diff'.format(col)] = (
            merged['{0}_ext'.format(col)] - merged['{0}_cur'.format(col)]
        )
        merged['{0}_abs_diff'.format(col)] = merged['{0}_diff'.format(col)].abs()
    ## print some stats ##
    print('Total Records: {0}'.format(len(merged)))
    print('Missing vs external: {0}'.format(
        len(merged[merged['value_pre_ext'].isna()]) / len(merged)
    ))
    print('MAE: {0}'.format(
        merged['value_pre_abs_diff'].mean()
    ))
    print('20 Largest differences in pre-game value:')
    for index,row in merged.sort_values(
        by='value_pre_abs_diff',
        ascending=False
    ).head(20).iterrows():
        print('{0}, Start {1}: {2} ({3}{4})'.format(
            row['qb'],
            row['start_number'],
            round(row['value_pre_cur'], 2),
            '+' if row['value_pre_diff'] > 0 else '-',
            round(row['value_pre_diff'], 2)
        ))
    ## write to csv ##
    merged.to_csv(
        '{0}/Development/qb_file_comparison.csv'.format(root_loc),
        index=False
    )
    ## calcualte errors ##
    merged['cur_to_cur_ae'] = numpy.absolute(
        merged['value_pre_cur'] - merged['game_value_cur']
    )
    merged['cur_to_ext_ae'] = numpy.absolute(   
        merged['value_pre_cur'] - merged['game_value_ext']
    )
    merged['ext_to_cur_ae'] = numpy.absolute(
        merged['value_pre_ext'] - merged['game_value_cur']
    )
    merged['ext_to_ext_ae'] = numpy.absolute(
        merged['value_pre_ext'] - merged['game_value_ext']
    )
    # calcualte mae ##
    mae_df = pd.DataFrame([
        {
            'predictor' : 'cur',
            'cur_mae' : merged['cur_to_cur_ae'].mean(),
            'ext_mae' : merged['cur_to_ext_ae'].mean()
        },
        {
            'predictor' : 'ext',
            'cur_mae' : merged['ext_to_cur_ae'].mean(),
            'ext_mae' : merged['ext_to_ext_ae'].mean()
        }
    ])
    ## write to csv ##
    mae_df.to_csv(
        '{0}/Development/qb_file_comparison_mae.csv'.format(root_loc),
        index=False
    )

def compare_to_538():
    '''
    Compares the nfelo QB predictions to the 538 QB predictions for 2009 to 2022
    '''
    ## get the flattened model data ##
    ## establish path ##
    root_loc = pathlib.Path(__file__).parent.parent.resolve()
    qbs = pd.read_csv(
        '{0}/Other Data/weekly_qb_states.csv'.format(root_loc)
    )
    ## get the 538 data ##
    db = dcm.load(['qbelo'])
    qbs_538 = db['qbelo'].copy()
    ## constraint to period where 538 was active and give ##
    ## buffer past 1999 to allow model to catch up from inits ##
    qbs_538 = qbs_538[
        (qbs_538['season'] >= 2002) &
        (qbs_538['season'] <= 2022)
    ].copy()
    ## flatten ##
    qbs_538 = pd.concat([
        qbs_538[[
            'game_id', 'qb1', 'qb1_value_pre', 'qb1_value_post',
            'qb1_game_value'
        ]].rename(columns={
            'qb1' : 'player_name',
            'qb1_value_pre' : 'value_pre_538',
            'qb1_value_post' : 'value_post_538',
            'qb1_game_value' : 'game_value_538'
        }),
        qbs_538[[
            'game_id', 'qb2', 'qb2_value_pre', 'qb2_value_post',
            'qb2_game_value'
        ]].rename(columns={
            'qb2' : 'player_name',
            'qb2_value_pre' : 'value_pre_538',
            'qb2_value_post' : 'value_post_538',
            'qb2_game_value' : 'game_value_538'
        })
    ])
    ## merge ##
    merged = pd.merge(
        qbs_538,
        qbs,
        on=['game_id', 'player_name'],
        how='left'
    )
    ## make adjs to 538 to make comparable ##
    ## translate elo to value ##
    for col in ['value_pre_538', 'value_post_538', 'game_value_538']:
        merged[col] = merged[col] / 3.3
    ## add the def adj ##
    merged['value_pre_538_def_adj'] = merged['value_pre_538'] - merged['opponent_def_value_pre']
    ## add the performance adj ##
    merged['game_value_538_def_adj'] = merged['game_value_538'] + merged['opponent_def_value_pre']
    ## calc maes ##
    merged['f38_to_f38_ae'] = numpy.absolute(
        merged['value_pre_538_def_adj'] - merged['game_value_538']
    )
    merged['f38_to_nfelo_ae'] = numpy.absolute(
        merged['value_pre_538_def_adj'] - merged['value_performance_def_adj']
    )
    merged['nfelo_to_f38_ae'] = numpy.absolute(
        merged['value_pre_def_adj'] - merged['game_value_538']
    )
    merged['nfelo_to_nfelo_ae'] = numpy.absolute(
        merged['value_pre_def_adj'] - merged['value_performance_def_adj']
    )
    ## drop any rows with na values ##
    merged = merged.dropna()
    ## calc maes ##
    mae_df = pd.DataFrame([
        {
            'model' : '538',
            'mae_to_538_value' : merged['f38_to_f38_ae'].mean(),
            'mae_to_nfelo_value' : merged['f38_to_nfelo_ae'].mean()
        },
        {
            'model' : 'nfelo',
            'mae_to_538_value' : merged['nfelo_to_f38_ae'].mean(),
            'mae_to_nfelo_value' : merged['nfelo_to_nfelo_ae'].mean()
        }
    ])
    ## write to csv ##
    merged.to_csv(
        '{0}/Development/qb_file_comparison_538.csv'.format(root_loc),
        index=False
    )
    mae_df.to_csv(
        '{0}/Development/mae_comparison.csv'.format(root_loc),
        index=False
    )
