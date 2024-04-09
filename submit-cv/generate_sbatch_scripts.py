import numpy as np

fname = 'gdb_config.dat'


run_config = """--device cuda \\
--experiment_name final-cv-80 \\
--CV 1 \\
--num_epochs 512 \\
--seed ${SEED} \\
--logdir /scratch/izar/briling/cv \\"""

base_config = np.loadtxt(fname, skiprows=1, dtype=str)
base_config = {key: val.strip('"') for key, val in base_config}
normal_base_config = '\n'.join([f'--{key} {val} \\' for key, val in base_config.items()])
true_base_config = '\n'.join((normal_base_config, '--two_layers_atom_diff \\', '--atom_mapping \\'))
rxnmapper_base_config = '\n'.join((true_base_config, '--rxnmapper \\'))
#cross_base_config =  TODO use different lr and weight decay?

base_configs = {'normal': normal_base_config, 'true':true_base_config, 'rxnmapper':rxnmapper_base_config}

base_name = f"ns{base_config['n_s']}-nv{base_config['n_v']}-d{base_config['distance_emb_dim']}-l{base_config['n_conv_layers']}-{base_config['graph_mode']}-{base_config['combine_mode']}-{base_config['sum_mode']}"

header=f"""#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem={8 if base_config['dataset']=='cyclo' else 4}GB
#SBATCH --time=24:00:00
#SBATCH --job-name=equireact
#SBATCH --array=0-9
#SBATCH --exclude=i39

SEED=$(( $SLURM_ARRAY_TASK_ID + 123 ))

module purge
conda activate equireact
python -c 'import torch; print(torch.cuda.is_available())'
wandb enabled

python train.py \\"""


# 1) random, dft, noH
for key, config in base_configs.items():
    extra_name = f"{base_config['dataset']}-random-noH-dft-{key}"
    wandb_name = f"cv10-{extra_name}-{base_name}"
    extra_config=f"--splitter random \\ \n--noH \\ \n--wandb_name {wandb_name} \\"
    with open(f'{extra_name}.sbatch', 'w') as f:
        print("\n".join((header, run_config, config, extra_config)), file=f)
        print(file=f)


# 2) scaffold, dft, noH
for key, config in base_configs.items():
    extra_name = f"{base_config['dataset']}-scaffold-noH-dft-{key}"
    wandb_name = f"cv10-{extra_name}-{base_name}"
    extra_config=f"--splitter scaffold \\ \n--noH \\ \n--wandb_name {wandb_name} \\"
    with open(f'{extra_name}.sbatch', 'w') as f:
        print("\n".join((header, run_config, config, extra_config)), file=f)
        print(file=f)


# 3) random, xtb, noH
for key, config in base_configs.items():
    extra_name = f"{base_config['dataset']}-random-noH-xtb-{key}"
    wandb_name = f"cv10-{extra_name}-{base_name}"
    extra_config=f"--splitter random \\ \n--noH \\ \n--xtb \\ \n--wandb_name {wandb_name} \\"
    with open(f'{extra_name}.sbatch', 'w') as f:
        print("\n".join((header, run_config, config, extra_config)), file=f)
        print(file=f)


# 4) random, xtb subset, noH
for key, config in base_configs.items():
    extra_name = f"{base_config['dataset']}-random-noH-sub-{key}"
    wandb_name = f"cv10-{extra_name}-{base_name}"
    extra_config=f"--splitter random \\ \n--noH \\ \n--xtb_subset \\ \n--wandb_name {wandb_name} \\"
    with open(f'{extra_name}.sbatch', 'w') as f:
        print("\n".join((header, run_config, config, extra_config)), file=f)
        print(file=f)


# 5) random, dft, withH
for key, config in base_configs.items():
    extra_name = f"{base_config['dataset']}-random-withH-dft-{key}"
    wandb_name = f"cv10-{extra_name}-{base_name}"
    extra_config=f"--splitter random \\ \n--wandb_name {wandb_name} \\"
    with open(f'{extra_name}.sbatch', 'w') as f:
        print("\n".join((header, run_config, config, extra_config)), file=f)
        print(file=f)


# 6) random, dft, noH, invariant
for key, config in base_configs.items():
    extra_name = f"{base_config['dataset']}-inv-random-noH-dft-{key}"
    wandb_name = f"cv10-{extra_name}-{base_name}"
    extra_config=f"--splitter random \\ \n--noH \\ \n--invariant \\ \n--wandb_name {wandb_name} \\"
    with open(f'{extra_name}.sbatch', 'w') as f:
        print("\n".join((header, run_config, config, extra_config)), file=f)
        print(file=f)


# 7) scaffold, dft, noH, invariant
for key, config in base_configs.items():
    extra_name = f"{base_config['dataset']}-inv-scaffold-noH-dft-{key}"
    wandb_name = f"cv10-{extra_name}-{base_name}"
    extra_config=f"--splitter scaffold \\ \n--noH \\ \n--invariant \\ \n--wandb_name {wandb_name} \\"
    with open(f'{extra_name}.sbatch', 'w') as f:
        print("\n".join((header, run_config, config, extra_config)), file=f)
        print(file=f)
