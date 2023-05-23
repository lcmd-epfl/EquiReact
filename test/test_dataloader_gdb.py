from process.dataloader import GDB722TS as gdb

#dlf =gdb(process=True)
dlf = gdb(process=False)

out = dlf[0]
label = out[0]
print('label', label)
idx = out[1]
print('idx', idx)

reactant_graph = out[2]
print('reactant graph info', reactant_graph.keys)
product_graphs = out[3:]
for product_graph in product_graphs:
    print('product graph', product_graph.keys)
    print('pos', product_graph.pos)