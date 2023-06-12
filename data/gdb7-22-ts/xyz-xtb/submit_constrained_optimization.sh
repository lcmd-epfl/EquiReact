#!/bin/bash

# We assume running this from the script directory

job_directory=$(pwd)
input=${1}
name="${1%%.py}"
script="${name}.py"
output="${job_directory}/${name}.out"
tmpdir='$SLURM_TMPDIR'
curdir='$SLURM_SUBMIT_DIR'
checker='$?'
scontrol show node -d > nodelist.dat
for i in $(grep -n 'CoresPerSocket=24' nodelist.dat | cut -d ' ' -f 1  ) ; do echo $i | cut -d ':' -f 2 | cut -d '=' -f 2 >> nodenames.dat ; done


echo "#!/bin/sh
#SBATCH --job-name=${name}
#SBATCH --mem=96gb
#SBATCH --cpus-per-task=24
#SBATCH --tasks=1
#SBATCH -N 1
#SBATCH --nodelist=node33
#SBATCH -o ${output}
hostname
cd ${tmpdir}
echo ${tmpdir}
cp ${curdir}/${script} ${tmpdir}
cp ${curdir}/.pysisyphusrc ${tmpdir}
mkdir opt_min
find ${curdir}/../xyz_structures/ -name '*.xyz' -exec cp {} ${tmpdir}/ \;
if [ ${checker} -eq 0 ]; then
   python ${script} 
fi
find opt_min/ -name '*.xyz' -exec mv {} ${curdir}/opt_min/ \;
if [ ${checker} -eq 0 ]; then
   rm -rf ${tmpdir}/opt_min/
fi
find . ! -name '*' -type f -exec rm -f {} +
exit " > ${name}.job

sbatch ${name}.job
#cp ${curdir}/opt_min/*.xyz opt_min
#cp ${curdir}/opt_ts/*.xyz opt_ts
