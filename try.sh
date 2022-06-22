#!/bin/sh

echo trial 6 seed 0
python3 main.py --trial 6 --seed 0 --max_timesteps 200000
echo trial 6 seed 1
python3 main.py --trial 6 --seed 1 --max_timesteps 200000
echo trial 6 seed 2
python3 main.py --trial 6 --seed 2 --max_timesteps 200000
echo trial 6 seed 3
python3 main.py --trial 6 --seed 3 --max_timesteps 200000
echo trial 6 seed 4
python3 main.py --trial 6 --seed 4 --max_timesteps 200000
