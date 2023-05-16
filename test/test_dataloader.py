from process.dataloader import Cyclo23TS as cyclo

dlf = cyclo(process=True)

label, idx, r_0_graph, r_1_graph, p_graph = dlf[0]
print("r0 graph info", r_0_graph.keys)
print("coords from graph", r_0_graph.pos)

print("r1 graph info", r_1_graph.keys)
print("coords from graph", r_1_graph.pos)

print("product graph info", p_graph.keys)
print("coords from graph", p_graph.pos)