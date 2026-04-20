"""Network topology and initialization — identical to Jacob et al. (2024). Do not modify.

NOTE: Copy the following DSS data files into this folder:
  IEEE123Master.dss, IEEE123Loads.DSS, IEEE123Regulators.DSS,
  IEEELineCodes.DSS, Loadshapes.dss, Loadshape_multload.csv, BusCoords.dat
"""

import os
import math
import numpy as np
import networkx as nx
from Environments.DSSdirect_123bus_loadandswitching.DSS_CircuitSetup import *

sectional_swt = [
    {'no': 1,  'line': 'L19'},  {'no': 2,  'line': 'L24'},
    {'no': 3,  'line': 'L26'},  {'no': 4,  'line': 'L43'},
    {'no': 5,  'line': 'L58'},  {'no': 6,  'line': 'L67'},
    {'no': 7,  'line': 'L81'},  {'no': 8,  'line': 'L86'},
    {'no': 9,  'line': 'L101'}, {'no': 10, 'line': 'Sw2'},
    {'no': 11, 'line': 'Sw3'},  {'no': 12, 'line': 'Sw4'},
    {'no': 13, 'line': 'Sw5'},
]

tie_swt = [
    {'no': 1, 'from node': '54',  'from conn': '.1.2.3', 'to node': '94',  'to conn': '.1.2.3', 'length': 0.350,  'code': '6', 'name': 'Sw8'},
    {'no': 2, 'from node': '151', 'from conn': '.1.2.3', 'to node': '300', 'to conn': '.1.2.3', 'length': 0.350,  'code': '6', 'name': 'Sw7'},
    {'no': 3, 'from node': '56',  'from conn': '.1.2.3', 'to node': '92',  'to conn': '.1.2.3', 'length': 0.350,  'code': '6', 'name': 'L120'},
    {'no': 4, 'from node': '83',  'from conn': '.1.2.3', 'to node': '95',  'to conn': '.1.2.3', 'length': 1.975,  'code': '2', 'name': 'L121'},
    {'no': 5, 'from node': '25',  'from conn': '.1.2.3', 'to node': '35',  'to conn': '.1.2.3', 'length': 0.906,  'code': '2', 'name': 'L122'},
    {'no': 6, 'from node': '250', 'from conn': '.1.2.3', 'to node': '300', 'to conn': '.1.2.3', 'length': 1.364,  'code': '3', 'name': 'L123'},
    {'no': 7, 'from node': '51',  'from conn': '.1.2.3', 'to node': '65',  'to conn': '.1.2.3', 'length': 0.600,  'code': '4', 'name': 'L124'},
    {'no': 8, 'from node': '101', 'from conn': '.1.2.3', 'to node': '151', 'to conn': '.1.2.3', 'length': 1.600,  'code': '3', 'name': 'L125'},
    {'no': 9, 'from node': '79',  'from conn': '.1.2.3', 'to node': '450', 'to conn': '.1.2.3', 'length': 1.304,  'code': '5', 'name': 'L126'},
]

generators = [
    {'no': 1,  'bus': '39',  'numphase': 3, 'phaseconn': '.1.2.3', 'size': 250, 'kV': 4.16, 'Gridforming': 'Yes'},
    {'no': 2,  'bus': '46',  'numphase': 3, 'phaseconn': '.1.2.3', 'size': 250, 'kV': 4.16, 'Gridforming': 'Yes'},
    {'no': 3,  'bus': '71',  'numphase': 3, 'phaseconn': '.1.2.3', 'size': 250, 'kV': 4.16, 'Gridforming': 'Yes'},
    {'no': 4,  'bus': '75',  'numphase': 3, 'phaseconn': '.1.2.3', 'size': 250, 'kV': 4.16, 'Gridforming': 'Yes'},
    {'no': 5,  'bus': '79',  'numphase': 3, 'phaseconn': '.1.2.3', 'size': 250, 'kV': 4.16, 'Gridforming': 'Yes'},
    {'no': 6,  'bus': '96',  'numphase': 3, 'phaseconn': '.1.2.3', 'size': 250, 'kV': 4.16, 'Gridforming': 'Yes'},
    {'no': 7,  'bus': '108', 'numphase': 3, 'phaseconn': '.1.2.3', 'size': 250, 'kV': 4.16, 'Gridforming': 'Yes'},
    {'no': 8,  'bus': '11',  'numphase': 3, 'phaseconn': '.1.2.3', 'size': 80,  'kV': 4.16, 'Gridforming': 'No'},
    {'no': 9,  'bus': '33',  'numphase': 3, 'phaseconn': '.1.2.3', 'size': 80,  'kV': 4.16, 'Gridforming': 'No'},
    {'no': 10, 'bus': '56',  'numphase': 3, 'phaseconn': '.1.2.3', 'size': 80,  'kV': 4.16, 'Gridforming': 'No'},
    {'no': 11, 'bus': '82',  'numphase': 3, 'phaseconn': '.1.2.3', 'size': 80,  'kV': 4.16, 'Gridforming': 'No'},
    {'no': 12, 'bus': '91',  'numphase': 3, 'phaseconn': '.1.2.3', 'size': 80,  'kV': 4.16, 'Gridforming': 'No'},
    {'no': 13, 'bus': '104', 'numphase': 3, 'phaseconn': '.1.2.3', 'size': 80,  'kV': 4.16, 'Gridforming': 'No'},
]

substatn_id    = '150'
dispatch_loads = [
    'Load.s82a', 'Load.s84c', 'Load.s92c', 'Load.s88a', 'Load.s69a',
    'Load.s100c', 'Load.s107b', 'Load.s109a', 'Load.s113a', 'Load.s48',
    'Load.s49a', 'Load.s64b', 'Load.s66c', 'Load.s53a', 'Load.s11a',
    'Load.s16c', 'Load.s20a', 'Load.s30c', 'Load.s22b',
]

n_actions = len(sectional_swt) + len(tie_swt) + len(dispatch_loads)   # = 41


def initialize():
    FolderName = os.path.dirname(os.path.realpath(__file__))
    DSSfile    = os.path.join(FolderName, "IEEE123Master.dss")
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
Generator_Buses     = {}
Generator_BlackStart = {}
i = DSSCktobj.dss.Generators.First()
while i > 0:
    elemName     = f'Generator.{DSSCktobj.dss.Generators.Name()}'
    DSSCktobj.dss.Circuit.SetActiveElement(elemName)
    bus_connectn = DSSCktobj.dss.CktElement.BusNames()[0].split('.')[0]
    Generator_Buses[elemName] = bus_connectn
    num = int(elemName[11:]) - 1
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
