"""DSS engine setup — identical to Jacob et al. (2024). Do not modify."""

import opendssdirect as dss
import numpy as np
import math
import networkx as nx


class DSS():
    def __init__(self, filename):
        self.filename = filename
        self.dss = dss

    def compile_ckt_dss(self):
        self.dss.Basic.ClearAll()
        self.dss.Text.Command("compile [" + self.filename + "]")


class Bus:
    def __init__(self, DSSCktobj, bus_name):
        Vmag = np.zeros(3)
        DSSCktobj.dss.Circuit.SetActiveBus(bus_name)
        V     = DSSCktobj.dss.Bus.puVmagAngle()
        nodes = np.array(DSSCktobj.dss.Bus.Nodes())
        for indx in range(len(nodes)):
            Vmag[nodes[indx] - 1] = V[int(indx * 2)]
        self.Vmag  = Vmag
        self.nodes = nodes


class Branch:
    def __init__(self, DSSCktobj, branch_fullname):
        DSSCktobj.dss.Transformers.First()
        KVA_base = DSSCktobj.dss.Transformers.kVA()
        KV_base  = DSSCktobj.dss.Transformers.kV()
        I_base   = KVA_base / (math.sqrt(3) * KV_base)

        DSSCktobj.dss.Circuit.SetActiveElement(branch_fullname)
        bus_connections = DSSCktobj.dss.CktElement.BusNames()
        bus1 = bus_connections[0]
        bus2 = bus_connections[1]

        i     = np.array(DSSCktobj.dss.CktElement.CurrentsMagAng())
        ctidx = 2 * np.array(range(0, min(int(i.size / 4), 3)))
        I_mag = i[ctidx]
        I_avg = (np.sum(I_mag)) / I_base

        self.bus_fr = bus1
        self.bus_to = bus2
        self.Cap    = I_avg


def CktModSetup(DSSfile, sectional_swt, tie_swt, generators):
    DSSCktobj = DSS(DSSfile)
    DSSCktobj.compile_ckt_dss()
    DSSCktobj.dss.Text.Command("Set Maxiterations=5000")
    DSSCktobj.dss.Text.Command("Set maxcontroliter=5000")
    DSSCktobj.dss.Basic.AllowForms(0)

    for sline in sectional_swt:
        DSSCktobj.dss.Text.Command(
            f"New swtcontrol.swSec{str(sline['no'])} SwitchedObj=Line.{sline['line']} "
            f"Normal=c SwitchedTerm=1 Action=c"
        )

    for tline in tie_swt:
        if tline['name'] in ('Sw7', 'Sw8'):
            DSSCktobj.dss.Text.Command(
                f"New Line.{tline['name']} Bus1={tline['from node']}{tline['from conn']} "
                f"Bus2={tline['to node']}{tline['to conn']} "
                f"r1=1e-3 r0=1e-3 x1=0.000 x0=0.000 c1=0.000 c0=0.000 Length=0.001"
            )
        else:
            DSSCktobj.dss.Text.Command(
                f"New Line.{tline['name']} Bus1={tline['from node']}{tline['from conn']} "
                f"Bus2={tline['to node']}{tline['to conn']} "
                f"LineCode={tline['code']} Length={str(tline['length'])}"
            )
        DSSCktobj.dss.Text.Command(
            f"New swtcontrol.swTie{str(tline['no'])} SwitchedObj=Line.{tline['name']} "
            f"Normal=o SwitchedTerm=1 Action=o"
        )

    for gen in generators:
        # Grid-forming DERs use Model=3 (constant P, regulated V) so they act as
        # voltage-source inverters and prevent voltage runaway in islanded sections.
        # Grid-feeding DERs use Model=1 (constant P, constant PF).
        mdl = 3 if gen.get('Gridforming', 'No') == 'Yes' else 1
        DSSCktobj.dss.Text.Command(
            f"New Generator.G{str(gen['no'])} bus1={gen['bus']}{gen['phaseconn']} "
            f"Phases={str(gen['numphase'])} Kv={str(gen['kV'])} Kw={str(gen['size'])} Pf=0.8 Model={mdl}"
        )

    # 34-bus network does not use load shapes
    return DSSCktobj


def graph_struct(DSSCktobj):
    G_original = nx.Graph()
    i = DSSCktobj.dss.PDElements.First()
    while i > 0:
        label_edge = []
        e = DSSCktobj.dss.PDElements.Name()
        if e.split('.')[0] in ('Line', 'Transformer'):
            branch_obj = Branch(DSSCktobj, e)
            sr_node    = branch_obj.bus_fr.split('.')[0]
            tar_node   = branch_obj.bus_to.split('.')[0]
            if G_original.has_edge(sr_node, tar_node):
                label_edge = list(G_original.edges[sr_node, tar_node]['label'])
                label_edge.append(e)
                G_original.edges[sr_node, tar_node]['label'] = label_edge
            elif G_original.has_edge(tar_node, sr_node):
                label_edge = list(G_original.edges[tar_node, sr_node]['label'])
                label_edge.append(e)
                G_original.edges[tar_node, sr_node]['label'] = label_edge
            else:
                label_edge.append(e)
                G_original.add_edge(sr_node, tar_node, label=label_edge)
        i = DSSCktobj.dss.PDElements.Next()
    return G_original
