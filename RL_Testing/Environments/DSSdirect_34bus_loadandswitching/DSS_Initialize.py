"""Network topology and initialization for 34-bus IEEE network. Do not modify."""

import os
import math
import numpy as np
import networkx as nx
from Environments.DSSdirect_34bus_loadandswitching.DSS_CircuitSetup import *

sectional_swt = [
    {'no': 1, 'line': 'L25'},
    {'no': 2, 'line': 'L17'},
    {'no': 3, 'line': 'L30'},
    {'no': 4, 'line': 'L14'},
    {'no': 5, 'line': 'L13'},
]

tie_swt = [
    {'no': 1, 'from node': '828', 'from conn': '.1.2.3', 'to node': '832', 'to conn': '.1.2.3', 'length': 21,  'code': '301', 'name': 'L33'},
    {'no': 2, 'from node': '824', 'from conn': '.1.2.3', 'to node': '848', 'to conn': '.1.2.3', 'length': 32,  'code': '301', 'name': 'L34'},
    {'no': 3, 'from node': '840', 'from conn': '.1.2.3', 'to node': '848', 'to conn': '.1.2.3', 'length': 32,  'code': '301', 'name': 'L35'},
    {'no': 4, 'from node': '814', 'from conn': '.1.2.3', 'to node': '828', 'to conn': '.1.2.3', 'length': 25,  'code': '301', 'name': 'L36'},
]

generators = [
    {'no': 1, 'bus': '820', 'numphase': 1, 'phaseconn': '.1',     'size': 96,  'kV': 14.376, 'Gridforming': 'No'},
    {'no': 2, 'bus': '890', 'numphase': 3, 'phaseconn': '.1.2.3', 'size': 146, 'kV': 4.16,   'Gridforming': 'Yes'},
    {'no': 3, 'bus': '844', 'numphase': 3, 'phaseconn': '.1.2.3', 'size': 144, 'kV': 24.9,   'Gridforming': 'Yes'},
    {'no': 4, 'bus': '816', 'numphase': 3, 'phaseconn': '.1.2.3', 'size': 200, 'kV': 24.9,   'Gridforming': 'Yes'},
]

substatn_id = 'sourcebus'

dispatch_loads = [
    'Load.s860', 'Load.s844', 'Load.s890', 'Load.d808_810sb',
    'Load.d818_820sa', 'Load.d816_824sb', 'Load.d824_828sc',
    'Load.d828_830sa', 'Load.d854_856sb', 'Load.d832_858rc',
]

n_actions = len(sectional_swt) + len(tie_swt) + len(dispatch_loads)  # = 19


def initialize():
    FolderName = os.path.dirname(os.path.realpath(__file__))
    DSSfile    = os.path.join(FolderName, "ieee34Mod1.dss")
    DSSCktobj  = CktModSetup(DSSfile, sectional_swt, tie_swt, generators)
    DSSCktobj.dss.Solution.Solve()
    conv_flag  = 1 if DSSCktobj.dss.Solution.Converged() else 0
    G_init     = graph_struct(DSSCktobj)
    return DSSCktobj, G_init, conv_flag


DSSCktobj, G_init, conv_flag = initialize()

# Normal operating topology (sectionalizing only, tie switches removed)
tie_edges = []
i = DSSCktobj.dss.SwtControls.First()
while i > 0:
    name = DSSCktobj.dss.SwtControls.Name()
    if name[:5] == 'swtie':
        line   = DSSCktobj.dss.SwtControls.SwitchedObj()
        br_obj = Branch(DSSCktobj, line)
        from_b = br_obj.bus_fr.split('.')[0]
        to_b   = br_obj.bus_to.split('.')[0]
        tie_edges.append((from_b, to_b))
    i = DSSCktobj.dss.SwtControls.Next()
G_base = G_init.copy()
G_base.remove_edges_from(tie_edges)

# Generator info
Generator_Buses      = {}
Generator_BlackStart = {}
i = DSSCktobj.dss.Generators.First()
while i > 0:
    elemName     = f'Generator.{DSSCktobj.dss.Generators.Name()}'
    DSSCktobj.dss.Circuit.SetActiveElement(elemName)
    bus_connectn = DSSCktobj.dss.CktElement.BusNames()[0].split('.')[0]
    Generator_Buses[elemName] = bus_connectn
    num = int(elemName[-1]) - 1
    Generator_BlackStart[elemName] = 1 if generators[num]['Gridforming'] == 'Yes' else 0
    i = DSSCktobj.dss.Generators.Next()

Load_Buses = {}
i = DSSCktobj.dss.Loads.First()
while i > 0:
    elemName = f'Load.{DSSCktobj.dss.Loads.Name()}'
    DSSCktobj.dss.Circuit.SetActiveElement(elemName)
    Load_Buses[elemName] = DSSCktobj.dss.CktElement.BusNames()[0].split('.')[0]
    i = DSSCktobj.dss.Loads.Next()

node_list  = list(G_init.nodes())
edge_list  = list(G_init.edges())
nodes_conn = [Bus(DSSCktobj, b).nodes for b in node_list]

gen_buses = np.array(list(Generator_Buses.values()))
gen_elems = list(Generator_Buses.keys())
Gen_Info  = {}
for n in node_list:
    blackstart_flag = 0
    gen_names = [gen_elems[x] for x in np.where(gen_buses == n)[0]]
    if gen_names:
        for g in gen_names:
            blackstart_flag += Generator_BlackStart[g]
        Gen_Info[n] = {'Generators': gen_names, 'Blackstart': blackstart_flag}

AllSwitches = []
i = DSSCktobj.dss.SwtControls.First()
while i > 0:
    name   = DSSCktobj.dss.SwtControls.Name()
    line   = DSSCktobj.dss.SwtControls.SwitchedObj()
    br_obj = Branch(DSSCktobj, line)
    AllSwitches.append({
        'switch name': name,
        'edge name':   line,
        'from bus':    br_obj.bus_fr.split('.')[0],
        'to bus':      br_obj.bus_to.split('.')[0],
        'status':      DSSCktobj.dss.SwtControls.Action() - 1,
    })
    i = DSSCktobj.dss.SwtControls.Next()

SwitchLines = [(s['from bus'], s['to bus']) for s in AllSwitches]

V_nodes = []
for n in node_list:
    V    = Bus(DSSCktobj, n).Vmag
    conn = Bus(DSSCktobj, n).nodes
    V_nodes.append({'name': n, 'Connection': conn, 'Voltage': V})
