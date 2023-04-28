# first line: 10
def get_window_range(argcomp,open_el='Li',allowpmu=False,trypreload=True): # trypredload : pickle 일 때 True, MP 일 때 False
    comp = Composition(argcomp)
    entry = VirtualEntry.from_composition(comp)
    oe = open_el
    entry.stabilize(trypreload=trypreload)
    window_list = []
    window_list = entry.get_printable_evolution_profile(oe, allowpmu=False,trypreload=trypreload)
    
    
    return window_list
