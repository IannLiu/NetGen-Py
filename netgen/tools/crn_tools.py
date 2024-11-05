from netgen.crngen.crn import CoreEdgeReactionModel
import networkx as nx
import os
import pygraphviz as pgv
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from typing import List
import numpy as np


def draw_mol(smiles: str,
             filename: str = None):
    """

    """
    mol = Chem.MolFromSmiles(smiles)
    d2d = Draw.MolDraw2DCairo(-1, -1)
    rdDepictor.Compute2DCoords(mol)
    rdDepictor.StraightenDepiction(mol)

    # Draw.DrawMoleculeACS1996(d2d, mol)
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    d2d.WriteDrawingText(filename)


def get_crn_graph(model: CoreEdgeReactionModel,
                  time: float,
                  reactant_name: List[str],
                  min_percentage: float = 0.1,
                  keep_species: List[str] = None):
    """
    Get core species rate as a networkx object
    :param model: the core reaction model
    :param time: simulation time
    :param reactant_name: the SMILES of reactants
    :param min_percentage: percentage smaller than min_percentage will be removed
    :param keep_species: keeping species and deleting other nodes
    :return:
    """
    reas_name = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in reactant_name]

    sim_results = model.simulate(time=time)
    core_concentration = sim_results['concentration'][:, -1]
    con_dict = {}
    core_species2idx = sim_results['species']
    for s, idx in core_species2idx.items():
        con_dict[s] = core_concentration[idx]

    # organize results using networkx by adding reaction to digraph one by one
    crn = nx.MultiDiGraph()
    for id_key, rxn_obj in model.core.reaction_pool.id_key2reaction.items():
        rate_f = rxn_obj.get_coef()
        rate_r = rxn_obj.get_coef(reverse=True)
        for rsmi in rxn_obj.get_rsmis():
            rate_f *= con_dict[rsmi]
        for psmi in rxn_obj.get_psmis():
            rate_r *= con_dict[psmi]

        net_rate_cst = rate_f - rate_r
        if net_rate_cst > 0:
            rsmis, psmis = rxn_obj.get_rsmis(), rxn_obj.get_psmis()
        else:
            rsmis, psmis = rxn_obj.get_psmis(), rxn_obj.get_rsmis()

        # for every species, we only calculate the consuming rate
        stoi_num = rxn_obj.get_stoichiometric_num
        for rsmi in set(rsmis):
            rate = abs(stoi_num[rsmi] * net_rate_cst)
            for psmi in set(psmis):
                crn.add_edge(rsmi, psmi, rate=rate, id_key=id_key)

    # reorganize the final network
    crn_g = nx.DiGraph()
    # iterating every node
    for node_name in crn.nodes:
        fluxes = []
        reactions = {}  # recoding rate of every node
        for sucr_name in crn.successors(node_name):
            idx = 0
            rate = 0  # rate of every edge transformation
            while crn.has_edge(node_name, sucr_name, idx):
                edge_info = crn.edges[node_name, sucr_name, idx]
                rate += edge_info["rate"]
                idx += 1
                if edge_info['id_key'] not in reactions:
                    reactions[edge_info['id_key']] = edge_info["rate"]
            fluxes.append([sucr_name, rate])
        # add fluxes for every node
        total_flux = np.sum(list(reactions.values()))
        for flux in fluxes:
            percent = flux[1] / total_flux
            if percent > min_percentage:
                crn_g.add_edge(node_name, flux[0], percentage=f'{round(percent * 100, 1)}%')

    # remove rest nodes
    removed_nodes = []
    if keep_species is not None:
        for node in crn_g.nodes:
            if node not in keep_species:
                removed_nodes.append(node)
    for node in removed_nodes:
        crn_g.remove_node(node)

    if len(list(nx.weakly_connected_components(crn_g))) > 1:
        removed_nodes = []
        for g in list(nx.weakly_connected_components(crn_g)):
            has_reactants = False
            for reactant in reas_name:
                if reactant in g:
                    has_reactants = True
            if not has_reactants:
                removed_nodes.append(g)
        if len(removed_nodes) >= 1:
            for nodes in removed_nodes:
                for n in nodes:
                    # print(f'remove node {n}')
                    crn_g.remove_node(n)

    # remove node without predecessors (except for reactant)
    removed_nodes = []
    for node in crn_g.nodes:
        # print(f'node {node} has {len(list(crn_g.predecessors(node)))} predecessors')
        if len(list(crn_g.predecessors(node))) == 0 and node not in reas_name:
            removed_nodes.append(node)
    for node in removed_nodes:
        # print(f'remove node {node}')
        crn_g.remove_node(node)

    return crn_g


def draw_flux_img(rxn_network, mol_img_style, dot_img_name, mol_fig_dir_name):
    """

    :param rxn_network: a networkx object
    :param mol_img_style: the figure style of molecules (png, jpg...), str
    :param dot_img_name: the name of output dot file, str
    :param mol_fig_dir_name: the file name of saved molecule images, str
    :return: None
    """
    node_img_path_dict = {}
    if not os.path.exists(mol_fig_dir_name):
        os.mkdir(mol_fig_dir_name)
    for node in rxn_network.nodes:
        path = os.path.join(mol_fig_dir_name, '.'.join([node, mol_img_style]))
        node_img_path_dict[node] = path
        if not os.path.exists(path):
            draw_mol(smiles=node, filename=path)
    G = pgv.AGraph(directed=True, ranksep=0.2, rankdir="LR", nodesep=0.1)
    for node_name in rxn_network.nodes:
        G.add_node(n=node_name, image=node_img_path_dict[node_name], fixedsize=False, label="", shape='none')
    for edge in rxn_network.edges:
        percentage = float(rxn_network.edges[edge[0], edge[1]]['percentage'][:-1])
        penwidth = percentage * 3.5 / 100 + 0.5
        if percentage < 10:
            color = "#b2b1ae"
        else:
            color = "#b20300"
        G.add_edge(edge[0], edge[1], label=rxn_network.edges[edge[0], edge[1]]['percentage'], fontsize=15,
                   fontsname="Times-Roman", penwidth=penwidth,
                   color=color)
    G.write(f'{dot_img_name}.dot')
    G.layout(prog='dot')
    G.draw(path=f'{dot_img_name}.png')
