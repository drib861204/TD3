#!/bin/sh

echo trial 3 seed 0
python3 main.py --trial 3 --seed 0 --max_timesteps 200000 -lr 1e-4
echo trial 3 seed 1
python3 main.py --trial 3 --seed 1 --max_timesteps 200000 -lr 1e-4
echo trial 3 seed 2
python3 main.py --trial 3 --seed 2 --max_timesteps 200000 -lr 1e-4
echo trial 3 seed 3
python3 main.py --trial 3 --seed 3 --max_timesteps 200000 -lr 1e-4
echo trial 3 seed 4
python3 main.py --trial 3 --seed 4 --max_timesteps 200000 -lr 1e-4

echo trial 4 seed 0
python3 main.py --trial 4 --seed 0 --max_timesteps 200000 -lr 4e-4
echo trial 4 seed 1
python3 main.py --trial 4 --seed 1 --max_timesteps 200000 -lr 4e-4
echo trial 4 seed 2
python3 main.py --trial 4 --seed 2 --max_timesteps 200000 -lr 4e-4
echo trial 4 seed 3
python3 main.py --trial 4 --seed 3 --max_timesteps 200000 -lr 4e-4
echo trial 4 seed 4
python3 main.py --trial 4 --seed 4 --max_timesteps 200000 -lr 4e-4

echo trial 5 seed 0
python3 main.py --trial 5 --seed 0 --max_timesteps 200000 -lr 7e-4
echo trial 5 seed 1
python3 main.py --trial 5 --seed 1 --max_timesteps 200000 -lr 7e-4
echo trial 5 seed 2
python3 main.py --trial 5 --seed 2 --max_timesteps 200000 -lr 7e-4
echo trial 5 seed 3
python3 main.py --trial 5 --seed 3 --max_timesteps 200000 -lr 7e-4
echo trial 5 seed 4
python3 main.py --trial 5 --seed 4 --max_timesteps 200000 -lr 7e-4
