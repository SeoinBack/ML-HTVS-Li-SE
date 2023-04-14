import pandas as pd
from pymatgen.core.composition import Composition 
from pymatgen.core.structure import Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.core.periodic_table import Element

from tqdm.auto import tqdm
import multiprocessing as multi
from joblib import Memory

from ast import literal_eval

mpr = MPRester('WgPaLVntnoS3X3tdlT')


ref_df = pd.read_csv('./new_order_data/01_1_all_Li_material_from_MP.csv',index_col='material_id')
for i in tqdm(range(len(ref_df))): 
    formula_dict = literal_eval(ref_df.iloc[i]['formula'])
    ref_df.iloc[i]['formula'] = (list(formula_dict.keys())+list(formula_dict.values()))
ref_df.head(5)
# 치환할 parent structure 가져올 때 필요함

all_df = pd.read_csv('./new_order_data/01_2_substitution_final.csv')
all_df['index'] = all_df.index

selected_df = pd.read_csv('./new_order_data/04_2_after_stability_window_screening.csv')

selected_structure_df = pd.merge(all_df, selected_df, left_on='reduced_formula',right_on='pretty_formula', how='inner')

changed_structure_df = pd.DataFrame(changed_structure_list)
changed_structure_df.columns = ['index','sub_structure']

sub_structure_df = pd.merge(selected_structure_df,changed_structure_df,on='index',how='inner')


formula_and_structure_df = sub_structure_df

wyckoffs_map = defaultdict(int)
alphabets = "abcdefghijklmnopqrstuvwxyzA"
for i in range(len(alphabets)):
    wyckoffs_map[alphabets[i]] = i+1

atoms_map = defaultdict(int)
for i in range(1,110):
    symbol = ase.atom.Atom(i).symbol
    atoms_map[symbol] = i
    
# read data from folder


# files = [i for i in os.listdir('/home/ahrehd0506/research/LiB/all_structure_from_MP/') if i.endswith('.POSCAR')] # 중복을 제거할 POSCAR 파일이 들어있는 폴더 지정
# df = pd.DataFrame()
# df['name'] =files


spacegroup_list = []
wyckoffs_list = []
species_list = []

# print(df)
for index, row in tqdm(formula_and_structure_df.iterrows()):
#     filename = row['name']
#     with open('/home/ahrehd0506/research/LiB/all_structure_from_MP/'+filename, 'r') as ftemp: # 여기도 폴더 지정
#     poscar = ftemp.read()
    poscar = row['sub_structure']
    
    # extract properties
    b = be.bulk.BULK()
    b.set_structure_from_file(poscar)
    name = b.get_name()
    spacegroup = b.get_spacegroup()
    wyckoffs = b.get_wyckoff()
    species = b.get_species()
    b.delete()      
    
    # sort in alphabetical order of Wyckoffs
    data = []
    for w,s in zip(wyckoffs, species):
        data.append({'wyckoff' : w, 'specie' : s})
    data.sort(key = lambda entry: entry['wyckoff'])
    wyckoffs = []
    species = []
    for entry in data:
        wyckoffs.append(entry['wyckoff'])
        species.append(entry['specie'])
    wyckoffs = "_".join([str(wyckoffs_map[_]) for _ in wyckoffs])
    species = "_".join([str(atoms_map[_]) for _ in species])
    
    spacegroup_list.append(spacegroup)
    wyckoffs_list.append(wyckoffs)
    species_list.append(species)
    

formula_and_structure_df['spacegroup'] = spacegroup_list
formula_and_structure_df['wyckoffs'] = wyckoffs_list
formula_and_structure_df['species'] = species_list

drop_dupl_df = formula_and_structure_df.drop_duplicates(['reduced_formula','spacegroup','wyckoffs','species'],keep='first')


df = pd.read_csv('./new_order_data/07_1_Liion_comp_528.csv')


r_xrd_list=[]
for i in tqdm(range(len(df))):
    x=df.iloc[i][4:-2]
    xx = [max(x[i*5:(i+1)*5]) for i in range(0,900)]
    norm = Spectrum(range(len(xx)),list(xx))
    norm.normalize('sum',10)
    r_xrd_list.append(list(norm.y))
new_df = df.iloc[:,:4]
new_df['r_xrd'] = r_xrd_list


df_ref = new_df

modified_rxd = []
for j in range(len(df_ref)):
    xrd = literal_eval(df_ref['r_xrd'][j])
    xrd_peak = []
    for i in range(len(xrd)-1):
        if xrd[i] > xrd[i+1] and xrd[i] > xrd[i-1]:
            xrd_peak.append(xrd[i])
        else:
            xrd_peak.append(0.0)
    xrd_peak.append(0.0)

    y = xrd_peak
    x = range(len(y))
    ssp = Spectrum(x,y)

    # fig, ax = plt.subplots(3,1,figsize=(30,15))
    # ax[0].plot(ssp.x,ssp.y)

    ssp.smear(0.2,'gaussian')
    # ax[1].plot(ssp.x,ssp.y)

    ssp.normalize('sum',10)
    # ax[2].plot(ssp.x,ssp.y)
    
    modified_rxd.append(ssp.y)
    
df_ref['modified_xrd'] = modified_rxd

df = pd.read_csv('./new_order_data/06_2_drop_duplicate.csv')

mxrd = []
for ii in tqdm(range(len(df))):
    atoms = read(StringIO(df['sub_structure'][ii]),format='vasp')
    anions = ['O','F','S','Cl','Se','Br','I','Te'] # except 16, 17 group

    if len(set(atoms.get_chemical_symbols())) == 2:
        del atoms[[atom.index for atom in atoms if atom.symbol == 'Li']]
    else:
        del atoms[[atom.index for atom in atoms if not (atom.symbol in anions)]]

    for i in range(len(atoms)):
        atoms[i].symbol = 'S'

#     print(atoms.cell.volume)
    v_offset = (atoms.get_global_number_of_atoms()*40)/atoms.cell.volume
    c_offset = (v_offset**(1/3))

    new_cell=atoms.cell*c_offset

    atoms.set_cell(new_cell,scale_atoms=True)
#     print(atoms.cell.volume)

    struc = AseAtomsAdaptor.get_structure(atoms)
    pa=xrd.XRDCalculator(wavelength='CrKb1').get_pattern(structure=struc,scaled=True,two_theta_range=(0,89.98))

    threshold = 0
    psd_idxs = pa.y > threshold #array of 0 and 1
    pay = pa.y * psd_idxs #zero out all the unnecessary powers

    ra = [round(x,1) for x in np.arange(0,91,0.1)]
    # ra2 = [round(x,2) for x in np.arange(0,90.1,0.5)]
    ry = [0 for _ in (ra)]

    rdf = pd.DataFrame(columns=ra)
    rdf.loc[0]=(ry)
    rdf

    rrdf = rdf[:]
    rx = [round(i,1) for i in pa.x]
    # rx = [round(i*2,0)/2 for i in pa.x]

    # rx = [round((i//0.5)*0.5,2) for i in pa.x]
    for i in range(len(rx)):    
        s=rx[i]
        rrdf[s] = max(rrdf[s][0],pay[i])

    spec = Spectrum(ra[:900],list(rrdf.loc[0])[:900])

#     fig, ax = plt.subplots(4,1,figsize=(30,15))
#     ax[0].plot(spec.x,spec.y)

    ssp = spec
    ssp.smear(0.2,'gaussian')
#     ax[1].plot(ssp.x,ssp.y)

    ssp.normalize('sum',10)
#     ax[2].plot(ssp.x,ssp.y)

    mxrd.append(list(ssp.y))

# x=df.iloc[259][4:-2]
# norm = Spectrum(range(len(x)),list(x))
# norm.normalize('sum',30)
# ax[3].plot(norm.y)
df['mxrd']=mxrd

df_new_only_mxrd = df

df_ref = pd.read_csv('./new_order_data/07_3_mxrd_from_ref.csv')[['formula_id','r_xrd','cond']]
df_ref.columns = ['composition','mxrd','conductivity']

df_new_and_old = pd.concat([df_new_only_mxrd,df_ref]).reset_index()
df_new_and_old.drop(['index'],axis=1,inplace=True)
