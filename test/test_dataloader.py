from process.dataloader import GDB7RXN as gdb

dlf = gdb(process=False)

r_graphs, r_atomtypes, r_coords, p_graphs, p_atomtypes, p_coords, labels, indices = dlf[0]
print('R graph...')

print("r atomtypes", r_atomtypes[0])
print("r atomtypes shape", r_atomtypes[0].shape)
print("r coords shape", r_coords[0].shape)

print("r graph info", r_graphs[0].keys)