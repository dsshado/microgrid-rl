"""State, action, and reward functions — identical to Jacob et al. (2024). Do not modify."""

import numpy as np
import math
from Environments.DSSdirect_123bus_loadandswitching.DSS_Initialize import *


def switchInfo(DSSCktobj):
    AllSwitches = []
    i = DSSCktobj.dss.SwtControls.First()
    while i > 0:
        name   = DSSCktobj.dss.SwtControls.Name()
        line   = DSSCktobj.dss.SwtControls.SwitchedObj()
        DSSCktobj.dss.Circuit.SetActiveElement(line)
        sw_status = DSSCktobj.dss.SwtControls.Action() - 1
        AllSwitches.append({'switch name': name, 'edge name': line, 'status': sw_status})
        i = DSSCktobj.dss.SwtControls.Next()
    return AllSwitches


def get_state(DSSCktobj, G, edgesout):
    Adj_mat = nx.adjacency_matrix(G, nodelist=node_list)

    DSSCktobj.dss.Transformers.First()
    KVA_base = DSSCktobj.dss.Transformers.kVA()

    En_Supply   = 0
    Total_Demand = 0
    for ld in list(DSSCktobj.dss.Loads.AllNames()):
        DSSCktobj.dss.Circuit.SetActiveElement(f"Load.{ld}")
        S     = np.array(DSSCktobj.dss.CktElement.Powers())
        ctidx = 2 * np.array(range(0, min(int(S.size / 2), 3)))
        P = S[ctidx]
        Q = S[ctidx + 1]
        if np.isnan(P).any() or np.isnan(Q).any():
            Power_Supp = 0
        else:
            Power_Supp = sum(P)
        if math.isnan(Power_Supp):
            Power_Supp = 0
        Demand    = float(DSSCktobj.dss.Properties.Value('kW'))
        En_Supply    += Power_Supp
        Total_Demand += Demand

    En_Supply_perc = En_Supply / Total_Demand if Total_Demand != 0 else -1

    Vmagpu    = []
    active_conn = []
    for b in node_list:
        V    = Bus(DSSCktobj, b).Vmag
        conn = Bus(DSSCktobj, b).nodes
        active_conn.append(conn)
        temp_flag = np.isnan(V)
        if np.any(temp_flag):
            V[temp_flag] = 0
            active_conn[node_list.index(b)] = np.array(
                [n for n in conn if not temp_flag[n - 1]]
            )
        Vmagpu.append(V)

    I_flow = []
    for e in G_init.edges(data=True):
        branchname = e[2]['label'][0]
        I_flow.append(Branch(DSSCktobj, branchname).Cap)

    if DSSCktobj.dss.Solution.Converged():
        conv_flag = 1
        Conv_const = 0
    else:
        conv_flag = 0
        Conv_const = 10

    V_viol = Volt_Constr(Vmagpu, active_conn)

    SwitchMasks = []
    for x in SwitchLines:
        SwitchMasks.append(1 if x in edgesout else 0)
    for _ in dispatch_loads:
        SwitchMasks.append(0)

    _clip = lambda x: np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
    return {
        "EnergySupp":           np.array([_clip(En_Supply_perc)], dtype=np.float32),
        "NodeFeat(BusVoltage)": np.array(_clip(Vmagpu),           dtype=np.float32),
        "EdgeFeat(Branchflow)": np.array(_clip(I_flow),           dtype=np.float32),
        "Adjacency":            np.array(Adj_mat.todense(),       dtype=np.float32),
        "VoltageViolation":     np.array([_clip(V_viol)],         dtype=np.float32),
        "ConvergenceViolation": np.array([Conv_const],            dtype=np.float32),
        "ActionMasking":        np.array(SwitchMasks,             dtype=np.float32),
    }


def take_action(action, out_edges):
    DSSCktObj, G_init_local, conv_flag = initialize()
    G_sc = G_init_local.copy()

    switch_actionidx = 0
    i = DSSCktObj.dss.SwtControls.First()
    while i > 0:
        switch_actionidx = i - 1
        cmd = 'o' if action[switch_actionidx] == 0 else 'c'
        DSSCktObj.dss.Text.Command(
            f'Swtcontrol.{DSSCktObj.dss.SwtControls.Name()}.Action={cmd}'
        )
        i = DSSCktObj.dss.SwtControls.Next()

    DSSCktObj.dss.Solution.Solve()

    for load_actionidx in range(switch_actionidx + 1, n_actions):
        loadname = dispatch_loads[load_actionidx - switch_actionidx - 1]
        if action[load_actionidx] == 0:
            DSSCktObj.dss.Circuit.SetActiveElement(loadname)
            DSSCktObj.dss.Text.Command(loadname + '.enabled="False"')
        DSSCktObj.dss.Solution.Solve()

    for o_e in out_edges:
        (u, v) = o_e
        if G_sc.has_edge(u, v):
            G_sc.remove_edge(u, v)
        branch_name = G_init_local.edges[o_e]['label'][0]
        DSSCktObj.dss.Circuit.SetActiveElement(branch_name)
        DSSCktObj.dss.Text.Command(f'Open {branch_name} term=1')
        DSSCktObj.dss.Solution.Solve()

    i = DSSCktObj.dss.SwtControls.First()
    while i > 0:
        line = DSSCktObj.dss.SwtControls.SwitchedObj()
        if DSSCktObj.dss.SwtControls.Action() == 1:
            b_obj = Branch(DSSCktObj, line)
            u = b_obj.bus_fr.split('.')[0]
            v = b_obj.bus_to.split('.')[0]
            if G_sc.has_edge(u, v):
                G_sc.remove_edge(u, v)
        i = DSSCktObj.dss.SwtControls.Next()

    Components   = list(nx.connected_components(G_sc))
    Virtual_Slack = []
    if len(Components) > 1:
        for C in Components:
            if substatn_id not in C:
                Slack_DER = {'name': '', 'kVA': 0}
                for gen_bus, gen_info in Gen_Info.items():
                    if gen_bus in C and gen_info['Blackstart'] == 1:
                        kva_val = 0
                        for gen_name in gen_info['Generators']:
                            DSSCktObj.dss.Circuit.SetActiveElement(gen_name)
                            kva_val += float(DSSCktObj.dss.Properties.Value('kVA'))
                        if kva_val > Slack_DER['kVA']:
                            Slack_DER['kVA']  = kva_val
                            Slack_DER['name'] = 'bus_' + gen_bus
                Virtual_Slack.append(Slack_DER)

    for vs in Virtual_Slack:
        Vs_name = vs['name']
        if Vs_name:
            Vs_locatn  = Vs_name.split('_')[1]
            Vs_MVA     = vs['kVA'] / 1000
            Vs_MVAsc3  = Vs_MVA
            Vs_MVAsc1  = Vs_MVAsc3 / 3
            DSSCktObj.dss.Circuit.SetActiveBus(Vs_locatn)
            Vs_kv = DSSCktObj.dss.Bus.kVBase() * math.sqrt(3)
            DSSCktObj.dss.Text.Command(
                f"New Vsource.{Vs_name} bus1={Vs_locatn} basekV={Vs_kv} phases=3 "
                f"Pu=1.00 angle=30 baseMVA={Vs_MVA} MVAsc3={Vs_MVAsc3} MVAsc1={Vs_MVAsc1} enabled=yes"
            )
            for gens in Gen_Info[Vs_locatn]['Generators']:
                DSSCktObj.dss.Text.Command(gens + '.enabled=no')
    DSSCktObj.dss.Solution.Solve()

    return DSSCktObj, G_sc


def Volt_Constr(Vmagpu, active_conn):
    Vmax, Vmin = 1.10, 0.90
    V_Viol = []
    for i in range(len(active_conn)):
        for phase_co in active_conn[i]:
            if Vmagpu[i][phase_co - 1] < Vmin:
                V_Viol.append(abs(Vmin - Vmagpu[i][phase_co - 1]) / Vmin)
            if Vmagpu[i][phase_co - 1] > Vmax:
                V_Viol.append(abs(Vmagpu[i][phase_co - 1] - Vmax) / Vmax)
    if V_Viol:
        return np.sum(V_Viol) / (len(G_init.nodes()) * 3)
    return 0.0


def get_reward(observ_dict):
    if float(observ_dict['ConvergenceViolation']) > 0 or math.isinf(float(observ_dict['VoltageViolation'])):
        return np.array([0.0])
    return observ_dict['EnergySupp'] - observ_dict['VoltageViolation']
