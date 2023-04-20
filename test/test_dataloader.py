from process.dataloader import Cyclo23TS as cyclo

dlf = cyclo(process=True)

r_0_graph, r_0_atomtypes, r_0_coords, r_1_graph, r_1_atomtypes, r_1_coords, p_graph, p_atomtypes, p_coords, label, idx = dlf[0]
print("r0 graph info", r_0_graph.keys)
print("coords from graph", r_0_graph.pos)
print("coords elsewhere", r_0_coords)

print("r1 graph info", r_1_graph.keys)
print("coords from graph", r_1_graph.pos)
print("coords elsewhere", r_1_coords)

print("product graph info", p_graph.keys)
print("coords from graph", p_graph.pos)
print("coords elsewhere", p_coords)