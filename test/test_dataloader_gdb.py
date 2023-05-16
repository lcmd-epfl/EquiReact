from process.dataloader import GDB722TS as gdb

dlf =gdb(process=True)

label, idx, label, idx, r, p_graphs = dlf[0]
print("r0 graph info", r_0_graph.keys)
print("coords from graph", r_0_graph.pos)

print("r1 graph info", r_1_graph.keys)
print("coords from graph", r_1_graph.pos)

for p_graph in p_graphs:
    print("p graph info", p_graph.keys)
    print("coords from graph", p_graph.pos)