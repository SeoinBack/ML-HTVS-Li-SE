import pickle

import os
import json
import re
import warnings
import pandas as pd
import numpy as np
from ast import literal_eval

import matplotlib.pyplot as plt
import seaborn as sns
from pymatgen.core.composition import Composition
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram

from tqdm.auto import tqdm

import multiprocessing as multi
from joblib import Memory

import parmap

tqdm.pandas

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('./new_order_data/01_2_substitution_final.csv')

unique_formula = list(set(df['reduced_formula']))

unique_formula_df = pd.DataFrame(columns=['pretty_formula','composition'])
unique_formula_df['pretty_formula'] = unique_formula
unique_formula_df['composition'] = [Composition(i).formula.replace(' ','') for i in unique_formula]

df = unique_formula_df[:]

elements = []
df['elements']=pd.Series()
for i in tqdm(range(len(df))):
    ele_list = Composition(df['composition'][i]).chemical_system.split('-')
    elements.append(ele_list)
df['elements'] = elements

df['elements_tuple'] = [ tuple(sorted(i)) for i in df['elements']]

unique_chemsys = set(df['elements_tuple'])
len(unique_chemsys)

for elements in tqdm(unique_chemsys):
    if type(elements) == str:
        elements = literal_eval(elements)
    list(elements).sort()
    with MPRester() as m:
        entries = m.get_entries_in_chemsys(elements) # get a list of all entries in the given chemsys(= elements list)
    chemsys_str = "".join(elements)
    with open(f'./new_order_data/entries_from_unique_chemsys/{chemsys_str}_all_entries_MP.pickle','wb') as fw:
        pickle.dump(entries, fw)
        
roost_df = pd.DataFrame(columns=['id','composition','target'])
roost_df['id'] = df.index
roost_df['composition'] = df['composition']
roost_df['target']=0

roost_df.to_csv('./new_order_data/02_3_unique_formula_for_roost.csv',index=False) # 이 DataFrame 을 ROOST 로 예측

############### 여기에 form energy 를 roost 로 예측

roost_result_df = pd.read_csv('./new_order_data/02_4_form_e_roost.csv')
roost_result_df

roost_ensemble_mean = roost_result_df.iloc[:,:2].rename(columns={'target':'roost_ensemble_mean'})
roost_ensemble_mean['roost_ensemble_mean'] = [roost_result_df.iloc[i,3:].mean() for i in tqdm(range(len(roost_result_df)))]
roost_ensemble_mean

num_cores = multi.cpu_count() # 8

# memory = Memory('./data/stability_multiprocesser',verbose=0)
# calc_stability_cached = memory.cache(e_above_hull_calculator)

hull_cal_input = list(zip(roost_ensemble_mean['composition'],roost_ensemble_mean['roost_ensemble_mean']))

with multi.Pool(num_cores) as pool:
    e_above_hull_result = list(tqdm(pool.imap(e_above_hull_calculator, hull_cal_input)))

e_above_hull_result

hull_result_df = pd.DataFrame(e_above_hull_result,columns=['composition','e_hull_from_roost'])


merged_df = pd.merge(roost_ensemble_mean, hull_result_df, how='inner',on='composition')

hull_screen_final_df = merged_df[(merged_df['roost_ensemble_mean']<0.1)&(merged_df['e_hull_from_roost']<0.2)] 

df = hull_screen_final_df

roost_df = pd.DataFrame(columns=['id','composition','target'])
roost_df['id'] = df.index
roost_df['composition'] = df['composition']
roost_df['target']=0

############ 여기에 roost gap 예측 부분


roost_band_gap_df = pd.read_csv('./new_order_data/03_2_band_gap_roost.csv') # ROOST 결과 가져오기
roost_band_gap_df.drop(['target'],axis=1,inplace=True)

e_gap_screen_final_df=roost_band_gap_df[roost_band_gap_df['pred_0']>0.5].rename(columns={'pred_0':'roost_band_gap'})
# e gap 기준 0.5
e_gap_screen_final_df['pretty_formula'] = [Composition(i).reduced_formula for i in e_gap_screen_final_df['composition']]


after_egap_hull_df = pd.merge(mp_df,e_gap_screen_final_df,on='composition',how='inner')


df = after_egap_hull_df.drop(['id_x','id_y'],axis=1)


unique_formula_df = df[['pretty_formula','composition']]

num_cores = multi.cpu_count() # 8
comp_list = unique_formula_df['pretty_formula']

memory = Memory('./data/stability_multiprocesser',verbose=0)
calc_stability_cached = memory.cache(get_window_range)

with multi.Pool(num_cores) as pool:
	stability = list(atqdm(pool.imap(calc_stability_cached, comp_list)))

stability_df = pd.DataFrame(stability,columns=['lower_bound','upper_bound','total_range'])
stability_df = pd.concat([unique_formula_df,stability_df],axis=1)
stability_df

screened_by_stability_window_df = stability_df[(stability_df['upper_bound']>1)&(stability_df['total_range']>0.001)]

merged_result = pd.merge(screened_by_stability_window_df,df,on=['composition','pretty_formula'],how='inner')

merged_result = pd.read_csv('./new_order_data/04_2_after_stability_window_screening.csv')
