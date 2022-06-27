#!/bin/sh

echo trial 22 seed 0
python3 main.py --trial 22 --seed 0 --max_timesteps 200000 -lr 3e-4
echo trial 22 seed 1
python3 main.py --trial 22 --seed 1 --max_timesteps 200000 -lr 3e-4
echo trial 22 seed 2
python3 main.py --trial 22 --seed 2 --max_timesteps 200000 -lr 3e-4
echo trial 22 seed 3
python3 main.py --trial 22 --seed 3 --max_timesteps 200000 -lr 3e-4
echo trial 22 seed 4
python3 main.py --trial 22 --seed 4 --max_timesteps 200000 -lr 3e-4

echo trial 23 seed 0
python3 main.py --trial 23 --seed 0 --max_timesteps 200000 -lr 4e-4
echo trial 23 seed 1
python3 main.py --trial 23 --seed 1 --max_timesteps 200000 -lr 4e-4
echo trial 23 seed 2
python3 main.py --trial 23 --seed 2 --max_timesteps 200000 -lr 4e-4
echo trial 23 seed 3
python3 main.py --trial 23 --seed 3 --max_timesteps 200000 -lr 4e-4
echo trial 23 seed 4
python3 main.py --trial 23 --seed 4 --max_timesteps 200000 -lr 4e-4

echo trial 24 seed 0
python3 main.py --trial 24 --seed 0 --max_timesteps 200000 -lr 5e-4
echo trial 24 seed 1
python3 main.py --trial 24 --seed 1 --max_timesteps 200000 -lr 5e-4
echo trial 24 seed 2
python3 main.py --trial 24 --seed 2 --max_timesteps 200000 -lr 5e-4
echo trial 24 seed 3
python3 main.py --trial 24 --seed 3 --max_timesteps 200000 -lr 5e-4
echo trial 24 seed 4
python3 main.py --trial 24 --seed 4 --max_timesteps 200000 -lr 5e-4

echo trial 25 seed 0
python3 main.py --trial 25 --seed 0 --max_timesteps 200000 -lr 6e-4
echo trial 25 seed 1
python3 main.py --trial 25 --seed 1 --max_timesteps 200000 -lr 6e-4
echo trial 25 seed 2
python3 main.py --trial 25 --seed 2 --max_timesteps 200000 -lr 6e-4
echo trial 25 seed 3
python3 main.py --trial 25 --seed 3 --max_timesteps 200000 -lr 6e-4
echo trial 25 seed 4
python3 main.py --trial 25 --seed 4 --max_timesteps 200000 -lr 6e-4

echo trial 26 seed 0
python3 main.py --trial 26 --seed 0 --max_timesteps 200000 -lr 7e-4
echo trial 26 seed 1
python3 main.py --trial 26 --seed 1 --max_timesteps 200000 -lr 7e-4
echo trial 26 seed 2
python3 main.py --trial 26 --seed 2 --max_timesteps 200000 -lr 7e-4
echo trial 26 seed 3
python3 main.py --trial 26 --seed 3 --max_timesteps 200000 -lr 7e-4
echo trial 26 seed 4
python3 main.py --trial 26 --seed 4 --max_timesteps 200000 -lr 7e-4

python3 plot_graph.py --trial 22
python3 main.py -l 1 --trial 22 --seed 0
python3 main.py -l 1 --trial 22 --seed 1
python3 main.py -l 1 --trial 22 --seed 2
python3 main.py -l 1 --trial 22 --seed 3
python3 main.py -l 1 --trial 22 --seed 4

python3 plot_graph.py --trial 23
python3 main.py -l 1 --trial 23 --seed 0
python3 main.py -l 1 --trial 23 --seed 1
python3 main.py -l 1 --trial 23 --seed 2
python3 main.py -l 1 --trial 23 --seed 3
python3 main.py -l 1 --trial 23 --seed 4

python3 plot_graph.py --trial 24
python3 main.py -l 1 --trial 24 --seed 0
python3 main.py -l 1 --trial 24 --seed 1
python3 main.py -l 1 --trial 24 --seed 2
python3 main.py -l 1 --trial 24 --seed 3
python3 main.py -l 1 --trial 24 --seed 4

python3 plot_graph.py --trial 25
python3 main.py -l 1 --trial 25 --seed 0
python3 main.py -l 1 --trial 25 --seed 1
python3 main.py -l 1 --trial 25 --seed 2
python3 main.py -l 1 --trial 25 --seed 3
python3 main.py -l 1 --trial 25 --seed 4

python3 plot_graph.py --trial 26
python3 main.py -l 1 --trial 26 --seed 0
python3 main.py -l 1 --trial 26 --seed 1
python3 main.py -l 1 --trial 26 --seed 2
python3 main.py -l 1 --trial 26 --seed 3
python3 main.py -l 1 --trial 26 --seed 4