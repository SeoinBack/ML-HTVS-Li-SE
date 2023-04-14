import os
import json
import re
import warnings
import pandas
import pickle

import matplotlib.pyplot as plt
from matplotlib import rc
from monty.json import MontyDecoder, MontyEncoder
from pymatgen.core import Composition, SETTINGS, Element
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram, GrandPotentialPhaseDiagram
from pymatgen.analysis.reaction_calculator import ComputedReaction
from pymatgen.entries.computed_entries import ComputedEntry

class VirtualEntry(ComputedEntry):
    def __init__(self, composition, energy, name=None): # class 멤버변수 composition, energy, name
        super(VirtualEntry, self).__init__(Composition(composition), energy)
        if name:
            self.name = name

    @classmethod
    def from_composition(cls, comp, energy=0, name=None): # 입력한 comp 로 class 초기화
        return cls(Composition(comp), energy, name=name)



    @property
    def chemsys(self): # 해당 class 의 composition 에 포함된 원소 종류 str list 를 return
        return [_.symbol for _ in self.composition.elements]

    def get_PD_entries(self, sup_el=None, exclusions=None, trypreload=False):
        """
        :param sup_el: a list for extra element dimension, using str format
        :param exclusions: a list of manually exclusion entries, can use entry name or mp_id
        :param trypreload: If try to reload from cached search results.
        Warning: if set to True, the return result may not be consistent with the updated MP database.
        :return: all related entries to construct phase diagram.
        """

        chemsys = self.chemsys + sup_el if sup_el else self.chemsys # 현재 chemsys + open element (oe 가 없을 경우 chemsys + chemsys) = str list
        chemsys = list(set(chemsys)) # 중복 제거

        if trypreload:
            entries = self.get_PD_entries_from_pickle(chemsys) 
        else:
#             entries = self.get_PD_entries_from_MP(chemsys) # chemsys 의 elements 로 만들 수 있는 entries 를 MP 에서 가져옴 --> 왜 안되는지 알 수 없음
            entries = MPRester().get_entries_in_chemsys(chemsys)
        entries.append(self) # MP 에서 가져온 entries 에 현재 class component 추가 --> 그래서 MP 에 없는 구조도 가능
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
        with open(f'./new_order_data/entries_from_unique_chemsys/{chemsys}_all_entries_MP.pickle','rb') as fr:
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
        # MP 에서 class 의 component 가 가진 element 로 만들 수 있는 모든 entries list 가져옴, self 포함
        offset = 30 if allowpmu else 0 # defalt = 0 이유는 모름
#         for e in pd_entries: # entries list 에 있는 각 물질에 대해
#             if e.composition.is_element and oe in e.composition.keys(): # 그 물질이 단일 원소 물질들 이고, open element 물질을 포함 하면
#                 e.correction += offset * e.composition.num_atoms # correction 값을 바꿔줌 : 근데 지금 offset=0 이라서 의미 없음
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
        df = pandas.DataFrame()
        df['mu_high (eV)'] = mu_h_list
        df['mu_low (eV)'] = mu_l_list
        df['d(n_{})'.format(open_el)] = oe_amt_list
        
        if not (abs(df['d(n_Li)'])<0.02).any():
            return [0,0,0]
        
        start = -list(df[abs(df['d(n_Li)'])<0.02]['mu_high (eV)'])[0]
        end = -list(df[abs(df['d(n_Li)'])<0.02]['mu_low (eV)'])[-1]
        tot_range = abs(start - end)
        window_range = [start,end,tot_range]
        
#         window_range = self.get_evolution_phases_table_string(open_el, pure_el_ref, PE_list, oe_amt_list, miu_trans_list, allowpmu)
        
        return window_range
        

        
#     @classmethod
#     def from_mp(cls, criteria): # 
#         entry = cls.get_mp_entry(criteria)
#         return cls(entry.composition, energy=entry.energy, name=entry.name)

#     @staticmethod
#     def get_mp_entry(criteria): # 
#         """
#         Here always return the lowest energy among all polymorphs.
#         Criteria can be a formula or an mp-id
#         """
#         with MPRester() as m:
#             entries = m.get_entries(criteria)
#         entries = sorted(entries, key=lambda e: e.energy_per_atom)
#         if len(entries) == 0:
#             raise ValueError("MP doesn't have any entry that matches the given formula/MP-id!")
#         return entries[0]



        
#     def get_evolution_phases_table_string(self, open_el, pure_el_ref, PE_list, oe_amt_list, mu_trans_list, allowpmu):
#         if not allowpmu:
#             mu_h_list = [0] + mu_trans_list
#         mu_l_list = mu_h_list[1:] + ['-inf']
#         df = pandas.DataFrame()
#         df['mu_high (eV)'] = mu_h_list
#         df['mu_low (eV)'] = mu_l_list
#         df['d(n_{})'.format(open_el)] = oe_amt_list
        
#         start = list(df[abs(df['d(n_Li)'])<0.02]['mu_high (eV)'])[0]
#         end = list(df[abs(df['d(n_Li)'])<0.02]['mu_low (eV)'])[-1]
#         tot_range = start - end
#         window_range = [start,end,tot_range]

#         return window_range

def get_window_range(argcomp,open_el='Li',allowpmu=False,trypreload=True): # trypredload : pickle 일 때 True, MP 일 때 False
    comp = Composition(argcomp)
    entry = VirtualEntry.from_composition(comp)
    oe = open_el
    entry.stabilize(trypreload=trypreload)
    window_list = []
    window_list = entry.get_printable_evolution_profile(oe, allowpmu=False,trypreload=trypreload)
    
    
    return window_list

