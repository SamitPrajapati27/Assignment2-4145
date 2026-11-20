#!/bin/bash
#SBATCH --job-name=a2_partB
#SBATCH --partition=Centaurus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=data/partB_%j.log

export OMP_PROC_BIND=true
export OMP_PLACES=cores
export B_STAR=256

mkdir -p data
gcc --version > data/system_info_partB.txt
lscpu >> data/system_info_partB.txt

make partB

HEADER="part,pr,pc,p,b,M,K,N,verified,min_s,median_s,avg_s,max_s,flops_per_s"
echo "$HEADER" > data/partB_results.csv

for sz in 1024 2048; do
    for b in 0 $B_STAR; do
        ./partB 0 0 $b $sz $sz $sz >> data/partB_results.csv
    done
done

for sz in 1024 2048; do
    for b in 0 $B_STAR; do
        for p in 1 2 4 8 12 18 36; do
            ./partB $p 1 $b $sz $sz $sz >> data/partB_results.csv
        done
    done
done

for sz in 1024 2048; do
    for grid in "1 2" "2 2" "2 4" "3 4" "3 6" "6 6"; do
        ./partB $grid $B_STAR $sz $sz $sz >> data/partB_results.csv
    done
done
./partB 1 36 $B_STAR 2048 2048 2048 >> data/partB_results.csv
./partB 6 6 $B_STAR 1000 1024 1536 >> data/partB_results.csv