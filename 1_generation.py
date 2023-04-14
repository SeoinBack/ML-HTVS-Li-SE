from pymatgen.core import Composition, Element, Structure
from pymatgen.ext.matproj import MPRester
import pymatgen.core as mg

import re
import pprint
import pickle
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas
from ast import literal_eval

import multiprocessing as multi
from joblib import Memory
import pickle

import parmap

from htvs_module import substitution, comp_to_form, comp_to_ele

mpr = MPRester('WgPaLVntnoS3X3tdlT')
mg.__version__

data = mpr.query({"elements":'Li','nelements':{ "$gte" : 3 , "$lte" : 4}}, properties=['material_id',"pretty_formula",'elements','formula','cif'])
# MP 에서 원하는 데이터 가져옴
df = pd.DataFrame(data)
# df.to_csv('./new_order_data/01_1_all_Li_material_from_MP.csv',index=False)

for i in tqdm(range(len(df))): 
    formula_dict = literal_eval(df.iloc[i]['formula'])
    df.iloc[i]['formula'] = (list(formula_dict.keys())+list(formula_dict.values()))
    
sub_el_list = ['Li','B','C','N','O','F','Na','Mg','Al','Si','P','S',
               'Cl','Ca','Zn','Ga','Ge','As','Se','Sr','Cd','In',
               'Sn','Te','I','Ba','Hg','Pb'] # 치환할 27개 원소

substituted_list=substitution(df,sub_el_list)

subcomp_mpid_df = pd.DataFrame()
sub_comp_list = []
mpid_list = []
reduced_form_list = []
element = []
for i in tqdm(substituted_list):
    my_comp_list=i[:-1]
    mpid=i[-1]
    reduced_form=Composition(comp_to_form(my_comp_list)).reduced_formula
#     ele_list = Composition(subcomp_mpid_df['reduced_formula'][i]).chemical_system.split('-')
    
    sub_comp_list.append(my_comp_list)
    mpid_list.append(mpid)
    reduced_form_list.append(reduced_form)
#     elements.append(tuple(sorted(ele_list)))
subcomp_mpid_df['sub_comp'] = sub_comp_list
subcomp_mpid_df['mpid'] = mpid_list
subcomp_mpid_df['reduced_formula'] = reduced_form_list
# subcomp_mpid_df['elements'] = elements
# subcomp_mpid_df

# subcomp_mpid_df.to_csv('./new_order_data/01_2_substitution_final.csv',index=False)
# subcomp_mpid_df = pd.read_csv('./new_order_data/01_2_substitution_final.csv')

selected_structure_df = subcomp_mpid_df
selected_structure_df = pd.merge(df[['material_id','cif']],selected_structure_df,left_on='material_id',right_on='mpid',how='right').rename({'cif':'ref_cif'},axis=1)
selected_structure_df

structure_substitution_input_list = [r.to_dict() for _, r in tqdm(selected_structure_df[['mpid','sub_comp','ref_cif']].iterrows())]
with open('./new_order_data/01_3_structure_substitution_input_list.pkl','wb') as f:
    pickle.dump(structure_substitution_input_list,f)
