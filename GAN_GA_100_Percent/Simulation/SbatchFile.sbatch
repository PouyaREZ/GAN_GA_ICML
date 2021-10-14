#!/bin/bash
#SBATCH --job-name=RQ3_Run_3_5_2021
#SBATCH --output=RQ3_Run_3_5_2021.%j.out
#SBATCH --error=RQ3_Run_3_5_2021.%j.err
#SBATCH --time=0-6:00:00
#SBATCH -p cee
#SBATCH -c 12
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pouyar@stanford.edu
module load python/3.6.1
python3.6 -m scoop --host $HOSTNAME -vv -n 12 Main.py
