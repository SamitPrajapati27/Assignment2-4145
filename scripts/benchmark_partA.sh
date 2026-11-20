#!/bin/bash
#SBATCH --job-name=a2_partA
#SBATCH --partition=Centaurus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=data/partA_%j.log

export B_STAR=256

mkdir -p data
gcc --version > data/system_info_partA.txt
lscpu >> data/system_info_partA.txt
echo "b* = $B_STAR" > data/b_star.txt

make partA

HEADER="part,pr,pc,p,b,M,K,N,verified,min_s,median_s,avg_s,max_s,flops_per_s"
echo "$HEADER" > data/partA_sweep.csv
echo "$HEADER" > data/partA_results.csv

./partA 0 0 0 1024 1024 1024 >> data/partA_sweep.csv
for b in 0 16 32 64 128 256; do
    ./partA 0 0 $b 2048 2048 2048 >> data/partA_sweep.csv
done

./partA 0 0 0 1024 1024 1024 >> data/partA_results.csv
./partA 0 0 0 2048 2048 2048 >> data/partA_results.csv

for sz in 1024 2048; do
    for b in 0 $B_STAR; do
        for p in 1 2 4 8 12 18 36; do
            ./partA $p 1 $b $sz $sz $sz >> data/partA_results.csv
        done
    done
done

for sz in 1024 2048; do
    for grid in "1 2" "2 2" "2 4" "3 4" "3 6" "6 6"; do
        ./partA $grid $B_STAR $sz $sz $sz >> data/partA_results.csv
    done
done
./partA 1 36 $B_STAR 2048 2048 2048 >> data/partA_results.csv
./partA 6 6 $B_STAR 1000 1024 1536 >> data/partA_results.csv