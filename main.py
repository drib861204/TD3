import numpy as np
import torch
import gym
import argparse
import os
import utils
import TD3
import OurDDPG
import DDPG
import matplotlib.pyplot as plt
from Pendulum_v3_mirror import *  # added by Ben
import time


def transient_response(eval_env, state_action_log):
	# print(np.shape(state_action_log)[0])
	fig, axs = plt.subplots(4)
	fig.suptitle('TD3 Transient Response')
	t = np.linspace(0, eval_env.dt * (state_action_log.shape[0] - 1), state_action_log.shape[0])
	axs[0].plot(t[1:], state_action_log[1:, 0])
	axs[3].plot(t[1:], state_action_log[1:, 1])
	axs[1].plot(t[1:], state_action_log[1:, 2])
	axs[2].plot(t[1:], state_action_log[1:, 3] * eval_env.max_torque)
	axs[0].set_ylabel('q1(rad)')
	axs[1].set_ylabel('q2 dot(rad/s)')
	axs[2].set_ylabel('torque(Nm)')
	axs[3].set_ylabel('q1 dot(rad/s)')
	axs[2].set_xlabel('time(s)')
	# axs[0].set_ylim([-0.01, 0.06])
	# axs[0].set_ylim([-pi-0.5,pi+0.5])
	axs[1].set_ylim([-34, 34])
	# axs[2].set_ylim([-12, 12])

	plt.savefig(f"runs/rwip{args.trial}/fig/response{args.seed}")
	plt.show()

	print("e_ss=", state_action_log[-1, 0])
	print("u_ss=", state_action_log[-1, 3] * eval_env.max_torque)
	print("q1_min=", min(state_action_log[1:, 0]))
	print("q1_min_index=", np.argmin(state_action_log[1:, 0]))
	print("OS%=", min(state_action_log[1:, 0]) / (eval_env.ang * pi / 180))
	print("q1_a=", eval_env.ang * pi / 180 * 0.9)
	print("q1_b=", eval_env.ang * pi / 180 * 0.1)
	print("q1_c=", eval_env.ang * pi / 180 * 0.1)
	print("q1_d=", -eval_env.ang * pi / 180 * 0.1)
	min_a = 100
	min_b = 100
	min_c = 100
	min_d = 100
	t_a = 100
	t_b = 100
	t_c = 100
	t_d = 100
	for i in range(1, np.shape(state_action_log)[0]):
		tr_a = eval_env.ang * pi / 180 * 0.9
		tr_b = eval_env.ang * pi / 180 * 0.1
		tr_c = eval_env.ang * pi / 180 * 0.1
		tr_d = -eval_env.ang * pi / 180 * 0.1
		diff_a = abs(state_action_log[i, 0] - tr_a)
		diff_b = abs(state_action_log[i, 0] - tr_b)
		diff_c = abs(state_action_log[i, 0] - tr_c)
		diff_d = abs(state_action_log[i, 0] - tr_d)
		if diff_a < min_a:
			min_a = diff_a
			t_a = i * eval_env.dt
		if diff_b < min_b:
			min_b = diff_b
			t_b = i * eval_env.dt
		if diff_c < min_c:
			min_c = diff_c
			t_c = i * eval_env.dt
		if diff_d < min_d:
			min_d = diff_d
			t_d = i * eval_env.dt
	print("[min_a, t_a, min_b, t_b]=", [min_a, t_a, min_b, t_b])
	print("rising time=", t_b - t_a)
	print("[min_c, t_c, min_d, t_d]=", [min_c, t_c, min_d, t_d])
	print("settling time=", t_c, "or", t_d)


def timer(start, end):
    """ Helper to print training time """
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nTraining Time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=3):
	# eval_env = gym.make(env_name)
	# eval_env.seed(seed + 100)
	if args.load_model:
		eval_env = Pendulum(1)
	else:
		eval_env = Pendulum(0)

	avg_reward = 0.

	for i in range(eval_episodes):

		# print(_)

		rep = 0
		state_action_log = np.zeros((1, 4))

		state, done = eval_env.reset('render'), False

		# print("eval_state",state)

		time_duration = 5 #second
		rep_max = time_duration/eval_env.dt

		while not done and rep < rep_max:
			rep += 1

			if args.load_model: # while training, don't render
				eval_env.render(i + 1)

			if state[2] >= eval_env.wheel_max_speed or state[2] <= -eval_env.wheel_max_speed:
				action = np.array([0])
			else:
				action = policy.select_action(np.array(state))
			# print("eval_action",action)
			state, reward, done, _ = eval_env.step(action)
			state_for_render = eval_env.state

			avg_reward += reward
			# print("eval_reward", reward)

			state_action = np.append(state_for_render, action)
			state_action_log = np.concatenate((state_action_log, np.asmatrix(state_action)), axis=0)

		if args.load_model:
			transient_response(eval_env, state_action_log)

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward}")
	print("---------------------------------------")
	return avg_reward

def run():
	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

	# Evaluate untrained policy
	# evaluations = [eval_policy(policy, args.env, args.seed)]

	state, done = env.reset(None), False

	# print("state",state)

	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	for t in range(int(args.max_timesteps)):
		episode_timesteps += 1

		if state[2] >= env.wheel_max_speed or state[2] <= -env.wheel_max_speed:
			action = np.array([0])
		else:
			# Select action randomly or according to policy
			if t < args.start_timesteps:
				# action = env.action_space.sample()
				action = np.random.uniform(low=-1, high=1, size=action_dim)
				'''action_test = (
					policy.select_action(np.array(state))
					+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
				).clip(-max_action, max_action)'''
			else:
				action = (
					policy.select_action(np.array(state))
					+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
				).clip(-max_action, max_action)


		# Perform action
		next_state, reward, done, _ = env.step(action)

		#print("test")
		# print(next_state)
		# print("reward",reward)
		# print(done)

		# done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
		done_bool = float(done) if episode_timesteps < 500 else 0

		# Store data in replay buffer
		# print(next_state)
		# print(reward)
		# print(done)
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		if done or episode_timesteps % 500 == 0:
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward}")
			log_f.write('{},{},{}\n'.format(episode_num, t, episode_reward))
			log_f.flush()

			# Reset environment
			state, done = env.reset(None), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1
			#if args.save_model: policy.save(f"runs/rwip{args.trial}/{file_name}")
			if not args.load_model: policy.save(f"runs/rwip{args.trial}/{file_name}")


# Evaluate episode
		#if (t + 1) % args.eval_freq == 0:
		#	evaluations.append(eval_policy(policy, args.env, args.seed))
		#	np.save(f"runs/rwip{args.trial}/log/{file_name}", evaluations)
		#	if args.save_model: policy.save(f"runs/rwip{args.trial}/{file_name}")



if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3", type=str)                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="rwip", type=str)                    # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)                        # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)          # Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)                 # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=5e4, type=int)             # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1, type=float)              # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)                # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)               # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)                   # Target network update rate
	parser.add_argument("--policy_noise", default=0.2, type=float)            # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5, type=float)              # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)                 # Frequency of delayed policy updates
	parser.add_argument("--save_model", default=1, type=int)                  # Save model and optimizer parameters, not in use now, only use load_model
	parser.add_argument("-l", "--load_model", default=0, type=int)            # 0: training; 1: testing
	parser.add_argument("--trial", default=0, type=int, help="trial")
	parser.add_argument("-lr", default=2e-3, type=float, help="learning rate")
	args = parser.parse_args()

	# print(args.save_model)

	file_name = f"rwip{args.trial}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	log_dir = f"runs/rwip{args.trial}/log/"
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	current_num_files = next(os.walk(log_dir))[2]
	run_num = len(current_num_files)
	log_f_name = log_dir + f"/TD3_log_{run_num}.csv"
	log_f = open(log_f_name,"w+")
	log_f.write('episode,timestep,raw_reward\n')

	t0 = time.time()

	# if args.save_model and not os.path.exists(f"runs/rwip{args.trial}"):
	#	os.makedirs(f"runs/rwip{args.trial}")

	# env = gym.make(args.env)
	env = Pendulum(0)

	# Set seeds
	# env.seed(args.seed)
	# env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		kwargs["lr"] = args.lr
		policy = TD3.TD3(**kwargs)
	elif args.policy == "OurDDPG":
		policy = OurDDPG.DDPG(**kwargs)
	elif args.policy == "DDPG":
		policy = DDPG.DDPG(**kwargs)

	if args.load_model:
		#policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"runs/rwip{args.trial}/{file_name}")
		eval_policy(policy, args.env, args.seed)

	#elif args.load_model == "train":
	else:
		run()

	'''
	elif args.load_model == "cont": # continue training from previous policy model (modification pending)
		policy.load(f"runs/rwip{args.trial}/{file_name}")

		replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

		# Evaluate untrained policy
		evaluations = [eval_policy(policy, args.env, args.seed)]

		state, done = env.reset(None), False

		# print("state",state)

		episode_reward = 0
		episode_timesteps = 0
		episode_num = 0

		for t in range(int(args.max_timesteps)):

			episode_timesteps += 1

			# Select action randomly or according to policy
			if t < args.start_timesteps:
				# action = env.action_space.sample()
				action = np.random.uniform(low=-1, high=1, size=action_dim)
				action_test = (
					policy.select_action(np.array(state))
					+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
				).clip(-max_action, max_action)
			else:
				action = (
					policy.select_action(np.array(state))
					+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
				).clip(-max_action, max_action)

			# Perform action
			next_state, reward, done, _ = env.step(action)

			#print("test")
			# print(next_state)
			# print("reward",reward)
			# print(done)

			# done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
			done_bool = float(done) if episode_timesteps < 200 else 0

			# Store data in replay buffer
			# print(next_state)
			# print(reward)
			# print(done)
			replay_buffer.add(state, action, next_state, reward, done_bool)

			state = next_state
			episode_reward += reward

			# Train agent after collecting sufficient data
			if t >= args.start_timesteps:
				policy.train(replay_buffer, args.batch_size)

			if done or episode_timesteps % 200 == 0:
				# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
				print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward}")
				# Reset environment
				state, done = env.reset(None), False
				episode_reward = 0
				episode_timesteps = 0
				episode_num += 1

			# Evaluate episode
			if (t + 1) % args.eval_freq == 0:
				evaluations.append(eval_policy(policy, args.env, args.seed))
				np.save(f"runs/rwip{args.trial}/log/{file_name}", evaluations)
				if args.save_model: policy.save(f"runs/rwip{args.trial}/{file_name}")
	'''

	t1 = time.time()
	timer(t0, t1)
	log_f.close()
