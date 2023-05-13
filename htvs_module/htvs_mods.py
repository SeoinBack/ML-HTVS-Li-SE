import ase
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from ast import literal_eval
from tqdm.auto import tqdm
from collections import defaultdict
from scipy.cluster import hierarchy as sch
from pymatgen.ext.matproj import MPRester
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram

from htvs_module.bulk_enumerator.bulk_enumerator import bulk

def dendro(cut,flight2):
    fig = plt.figure(figsize=(30,10))

    ax1 = fig.add_subplot(1,1,1)
    dend1 = sch.linkage(flight2, method='ward')
    cutoff = cut*max(dend1[:,2])
    dend_res1 = sch.dendrogram(dend1, color_threshold=cutoff)
    ax1.set_xticklabels([])
    return dend_res1

def get_indirect_structure(formula_and_structure_df):
    wyckoffs_map = defaultdict(int)
    alphabets = "abcdefghijklmnopqrstuvwxyzA"
    for i in range(len(alphabets)):
        wyckoffs_map[alphabets[i]] = i+1

    atoms_map = defaultdict(int)
    for i in range(1,110):
        symbol = ase.atom.Atom(i).symbol
        atoms_map[symbol] = i

    spacegroup_list = []
    wyckoffs_list = []
    species_list = []

    for index, row in tqdm(formula_and_structure_df.iterrows(),total=len(formula_and_structure_df)):
        poscar = row['sub_structure']

        # extract properties
        b = bulk.BULK()
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
    
    
    
    return formula_and_structure_df

def structure_substitution(row,ref_df): # rowDataFramempid, reference structure, substituted chemical system dictionaly expected
    index = row['index']
    mpid = row['mpid']
    ref_series = ref_df.loc[mpid]
    struc = Structure.from_str(ref_series['cif'],fmt='cif')
    initial_chemsys_list = ref_series['formula'] # struc.composition.chemical_system.split('-')
    final_chemsys_list = literal_eval(row['sub_comp'])
    num_ele = len(final_chemsys_list)//2
    struc.replace_species(dict(zip(initial_chemsys_list[:num_ele],final_chemsys_list[:num_ele])))
    poscar = struc.to(fmt='poscar')
    return [index, poscar]

def get_window_range(argcomp,open_el='Li',allowpmu=False,trypreload=True): # trypredload : pickle data -> True, MP data -> False
    comp = Composition(argcomp)
    entry = VirtualEntry.from_composition(comp)
    oe = open_el
    entry.stabilize(trypreload=trypreload)
    window_list = []
    window_list = entry.get_printable_evolution_profile(oe, allowpmu=False,trypreload=trypreload)
    
    
    return window_list

def substitution(df,sub_elements):
    substituted_list = []
    for j in tqdm(range(len(df))):
        temp_list = df['formula'][j]
        temp_range = list(range(int(len(temp_list)/2))) 
        del temp_range[temp_list.index('Li')] # not Li element ex) ['Li', 'Cd', 'Cu', 1.0, 2.0, 1.0] -> [1, 2]
        
        for i in temp_range:
            temp_range2 = temp_range[:]
            temp_list2 = temp_list[:]
            temp_list2[i]= 'S'
            temp_range2.remove(i) # substitution to S
            for k in temp_range2:
                temp_list3 = temp_list2[:]
                temp_range2.remove(k) 
                for s1 in sub_elements:
                    temp_list3[k] = s1

                    if temp_range2: # quaternary
                        for s2 in sub_elements:
                            temp_list4 = temp_list3[:]
                            temp_list4[temp_range2[0]] = s2
                            substituted_list.append(temp_list4+[df['material_id'][j]]) 
                    else: # ternary
                        temp_list5 = temp_list3[:]
                        substituted_list.append(temp_list5+[df['material_id'][j]])
                        
    return substituted_list

def comp_to_form(tt):
    form = ''
    n_ele = int(len(tt)/2)
    for i in range(n_ele):
        form += str(tt[i]) + str(tt[n_ele+i])
    return form
def comp_to_ele(tt):
    form = ''
    n_ele = int(len(tt)/2)
    for i in range(n_ele):
        form += str(tt[i])
    return form

def e_above_hull_calculator(comp_and_form_energy, mp=False):
    e_above_hull_list_unique = []
    trypreload=not mp
    
    argcomp = comp_and_form_energy[0] # composition from data frame
    roost_form_energy = comp_and_form_energy[1] # roost energy from data frame (eV/atom)??
    comp = Composition(argcomp) # make Composition type data
    sys=[_.symbol for _ in comp.elements] # elements list
    entries = get_PD_entries_new(sys,trypreload=trypreload) # get Phase diagram entries list from MP for the right above elements list
    phase_dia = PhaseDiagram(entries) # make entries list to PhaseDiagram type data 

    decomp = phase_dia.get_decomposition(comp)

    tot_energy_on_hull = sum([e.energy_per_atom * n for e, n in decomp.items()])
    form_e_on_hull = tot_energy_on_hull - (sum([comp[el] * phase_dia.el_refs[el].energy_per_atom for el in comp.elements])/comp.num_atoms)
    e_above_hull = roost_form_energy -  form_e_on_hull
#     e_above_hull_list_unique.append(e_above_hull)
    return [argcomp,e_above_hull]

def get_PD_entries_new(chemsys,trypreload):
        chemsys = list(set(chemsys)) 

        if trypreload:
            chemsys.sort()
            chemsys = "".join(chemsys)
            with open(f'./data/entries_from_unique_chemsys/{chemsys}_all_entries_MP.pickle','rb') as fr:
                entries = pickle.load(fr)               
            
        else:
#             entries = MPRester().get_entries_in_chemsys(chemsys)
            pass
            
        return entries


class VirtualEntry(ComputedEntry):
    def __init__(self, composition, energy, name=None): # class variable: composition, energy, name
        super(VirtualEntry, self).__init__(Composition(composition), energy)
        if name:
            self.name = name

    @classmethod
    def from_composition(cls, comp, energy=0, name=None): 
        return cls(Composition(comp), energy, name=name)



    @property
    def chemsys(self): 
        return [_.symbol for _ in self.composition.elements]

    def get_PD_entries(self, sup_el=None, exclusions=None, trypreload=False):
        """
        :param sup_el: a list for extra element dimension, using str format
        :param exclusions: a list of manually exclusion entries, can use entry name or mp_id
        :param trypreload: If try to reload from cached search results.
        Warning: if set to True, the return result may not be consistent with the updated MP database.
        :return: all related entries to construct phase diagram.
        """

        chemsys = self.chemsys + sup_el if sup_el else self.chemsys 
        chemsys = list(set(chemsys)) 

        if trypreload:
            entries = self.get_PD_entries_from_pickle(chemsys) 
        else:
            entries = MPRester().get_entries_in_chemsys(chemsys)
        entries.append(self) 
        if exclusions:
            entries = [e for e in entries if e.name not in exclusions]
            entries = [e for e in entries if e.entry_id not in exclusions]
        return entries

    @staticmethod
    def get_PD_entries_from_pickle(chemsys):
#         with MPRester() as m:
#             entries = m.get_entries_in_chemsys(chemsys) # get a list of all entries in the given chemsys(= elements list)
        chemsys.sort()
        chemsys = "".join(chemsys)
        with open(f'./data/entries_from_unique_chemsys/{chemsys}_all_entries_MP.pickle','rb') as fr:
            entries = pickle.load(fr)            
        return entries

    def get_PD_entries_from_MP(chemsys):
        #with MPRester() as m:
        entries = MPRester().get_entries_in_chemsys(chemsys) # get a list of all entries in the given chemsys(= elements list)
#         chemsys.sort()
#         chemsys = "".join(chemsys)
#         with open(f'./entries_from_unique_chemsys/{chemsys}_all_entries_MP.pickle','rb') as fr:
#             entries = pickle.load(fr)
            
        return entries

    def get_decomp_entries_and_e_above_hull(self, entries=None, exclusions=None, trypreload=None):
        if not entries:
            entries = self.get_PD_entries(exclusions=exclusions, trypreload=trypreload)
        pd = PhaseDiagram(entries)
        decomp_entries, hull_energy = pd.get_decomp_and_e_above_hull(self)
        return decomp_entries, hull_energy

    def stabilize(self, entries=None,trypreload=False):
        """
        Stabilize an entry by putting it on the convex hull
        """
        decomp_entries, hull_energy = self.get_decomp_entries_and_e_above_hull(entries=entries,trypreload=trypreload)
        self.correction -= (hull_energy * self.composition.num_atoms + 1e-8)
        return None

    def get_phase_evolution_profile(self, oe, allowpmu=False, entries=None,exclusions=None, trypreload=False):
        pd_entries = entries if entries else self.get_PD_entries(sup_el=[oe],exclusions=exclusions,trypreload=trypreload) 
        offset = 30 if allowpmu else 0 
#         for e in pd_entries: # entries list 에 있는 각 물질에 대해
#             if e.composition.is_element and oe in e.composition.keys(): 
#                 e.correction += offset * e.composition.num_atoms 
        pd = PhaseDiagram(pd_entries)
        evolution_profile = pd.get_element_profile(oe, self.composition.reduced_composition)
        el_ref = evolution_profile[0]['element_reference']
        el_ref.correction -= el_ref.composition.num_atoms * offset
        evolution_profile[0]['chempot'] -= offset
        return evolution_profile





    def get_printable_evolution_profile(self, open_el, entries=None, plot_rxn_e=True, allowpmu=False, trypreload=False):
        evolution_profile = self.get_phase_evolution_profile(open_el, entries=entries, allowpmu=allowpmu,trypreload=trypreload)

        PE_list = [list(stage['entries']) for stage in evolution_profile]
        oe_amt_list = [stage['evolution'] for stage in evolution_profile]
        pure_el_ref = evolution_profile[0]['element_reference']

        miu_trans_list = [stage['chempot'] for stage in evolution_profile][1:]  # The first chempot is always useless
        miu_trans_list = sorted(miu_trans_list, reverse=True)
        miu_trans_list = [miu - pure_el_ref.energy_per_atom for miu in miu_trans_list]
        
        
        if not allowpmu:
            mu_h_list = [0] + miu_trans_list
        mu_l_list = mu_h_list[1:] + ['-inf']
        df = pd.DataFrame()
        df['mu_high (eV)'] = mu_h_list
        df['mu_low (eV)'] = mu_l_list
        df['d(n_{})'.format(open_el)] = oe_amt_list
        
        if not (abs(df['d(n_Li)'])<0.02).any():
            return [0,0,0]
        
        start = -list(df[abs(df['d(n_Li)'])<0.02]['mu_high (eV)'])[0]
        end = -list(df[abs(df['d(n_Li)'])<0.02]['mu_low (eV)'])[-1]
        tot_range = abs(start - end)
        window_range = [start,end,tot_range]
        
        return window_range
