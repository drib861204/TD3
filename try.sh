#!/bin/sh

echo trial 0 seed 0
python3 main.py --trial 0 --seed 0 --max_timesteps 200000 -lr 1e-3
echo trial 0 seed 1
python3 main.py --trial 0 --seed 1 --max_timesteps 200000 -lr 1e-3
echo trial 0 seed 2
python3 main.py --trial 0 --seed 2 --max_timesteps 200000 -lr 1e-3
echo trial 0 seed 3
python3 main.py --trial 0 --seed 3 --max_timesteps 200000 -lr 1e-3
echo trial 0 seed 4
python3 main.py --trial 0 --seed 4 --max_timesteps 200000 -lr 1e-3

echo trial 1 seed 0
python3 main.py --trial 1 --seed 0 --max_timesteps 200000 -lr 2e-3
echo trial 1 seed 1
python3 main.py --trial 1 --seed 1 --max_timesteps 200000 -lr 2e-3
echo trial 1 seed 2
python3 main.py --trial 1 --seed 2 --max_timesteps 200000 -lr 2e-3
echo trial 1 seed 3
python3 main.py --trial 1 --seed 3 --max_timesteps 200000 -lr 2e-3
echo trial 1 seed 4
python3 main.py --trial 1 --seed 4 --max_timesteps 200000 -lr 2e-3

echo trial 2 seed 0
python3 main.py --trial 2 --seed 0 --max_timesteps 200000 -lr 3e-3
echo trial 2 seed 1
python3 main.py --trial 2 --seed 1 --max_timesteps 200000 -lr 3e-3
echo trial 2 seed 2
python3 main.py --trial 2 --seed 2 --max_timesteps 200000 -lr 3e-3
echo trial 2 seed 3
python3 main.py --trial 2 --seed 3 --max_timesteps 200000 -lr 3e-3
echo trial 2 seed 4
python3 main.py --trial 2 --seed 4 --max_timesteps 200000 -lr 3e-3
