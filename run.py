import os
import sys
import subprocess
import numpy as np
from time import time, sleep
import argparse
import json
import copy
import tensorflow as tf
import gym
import gym_auv
import gym_auv.reporting
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv, SubprocVecEnv
import stable_baselines.ddpg.policies
import stable_baselines.td3.policies
from stable_baselines.ddpg import AdaptiveParamNoiseSpec, NormalActionNoise, LnMlpPolicy
from stable_baselines import PPO2, DDPG, TD3, A2C, ACER, ACKTR
from sklearn.model_selection import ParameterGrid
from shapely import speedups

speedups.enable()
DIR_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def _preprocess_custom_envconfig(rawconfig):
    custom_envconfig = dict(zip(args.envconfig[::2], args.envconfig[1::2]))
    for key in custom_envconfig:
        try:
            custom_envconfig[key] = float(custom_envconfig[key])
            if (custom_envconfig[key] == int(custom_envconfig[key])):
                custom_envconfig[key] = int(custom_envconfig[key])
        except ValueError:
            pass
    return custom_envconfig

def create_env(env_id, envconfig, test_mode=False, render_mode='2d', pilot=None, verbose=False):
    if pilot:
        env = gym.make(env_id, env_config=envconfig, test_mode=test_mode, render_mode=render_mode, pilot=pilot, verbose=verbose)
    else:
        env = gym.make(env_id, env_config=envconfig, test_mode=test_mode, render_mode=render_mode, verbose=verbose)
    return env

def make_mp_env(env_id, rank, envconfig, seed=0, pilot=None):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = create_env(env_id, envconfig, pilot=pilot)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


def play_scenario(env, recorded_env, args, agent=None):
    # if args.video:
    #     print('Recording enabled')
    #     recorded_env = VecVideoRecorder(env, args.video_dir, record_video_trigger=lambda x: x == 0,
    #         video_length=args.video_length, name_prefix=args.video_name
    #     )

    from pyglet.window import key

    key_input = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    autopilot = False

    print('Playing scenario: ', env)
    # print('KEY BINDINGS')
    # print('Lambda control: K, L')
    # print('Toggle autopilot: A')
    # print('Restart: R')
    # print('Quit: Q')

    def key_press(k, mod):
        nonlocal autopilot
        if k == key.DOWN:  key_input[0] = -1
        if k == key.UP:    key_input[0] = 1
        if k == key.LEFT:  key_input[1] = 0.5
        if k == key.RIGHT: key_input[1] = -0.5
        if k == key.NUM_2: key_input[2] = -1
        if k == key.NUM_1: key_input[2] = 1
        if k == key.J: key_input[3] = -1
        if k == key.U: key_input[3] = 1
        if k == key.I: key_input[4] = -1
        if k == key.K: key_input[4] = 1
        if k == key.O: key_input[5] = -1
        if k == key.P: key_input[5] = 1
        if k == key.NUM_4: key_input[6] = -1
        if k == key.NUM_3: key_input[6] = 1
        if k == key.A: autopilot = not autopilot

    def key_release(k, mod):
        nonlocal restart, quit
        if k == key.R:
            restart = True
            print('Restart')
        if k == key.Q:
            quit = True
            print('quit')
        if k == key.UP:    key_input[0] = -1
        if k == key.DOWN:  key_input[0] = -1
        if k == key.LEFT and key_input[1] != 0: key_input[1] = 0
        if k == key.RIGHT and key_input[1] != 0: key_input[1] = 0
        if k == key.NUM_2 and key_input[2] != 0: key_input[2] = 0
        if k == key.NUM_1 and key_input[2] != 0: key_input[2] = 0
        if k == key.U and key_input[3] != 0: key_input[3] = 0
        if k == key.J and key_input[3] != 0: key_input[3] = 0
        if k == key.I and key_input[4] != 0: key_input[4] = 0
        if k == key.K and key_input[4] != 0: key_input[4] = 0
        if k == key.O and key_input[5] != 0: key_input[5] = 0
        if k == key.P and key_input[5] != 0: key_input[5] = 0
        if k == key.NUM_4 and key_input[6] != 0: key_input[6] = 0
        if k == key.NUM_3 and key_input[6] != 0: key_input[6] = 0

    viewer = env.viewer2d if args.render in {'both', '2d'} else env.viewer3d
    viewer.window.on_key_press = key_press
    viewer.window.on_key_release = key_release

    try:
        while True:
            t = time()
            restart = False
            steps = 0
            quit = False
            if (args.env == 'PathGeneration-v0'):
                a = np.array([5.0, 5.0, 1.0, 1.0])
            elif (args.env == 'PathColavControl-v0'):
                a = np.array([0.0])
            else:
                a = np.array([0.0, 0.0])

            obs = None
            while True:
                t, dt = time(), time()-t
                if args.env == 'PathGeneration-v0':
                    a[0] += key_input[1]
                    a[1] = max(0, key_input[0], a[1] + 0.1*key_input[0])
                    a[2] += 0.1*key_input[2]
                    print('Applied action: ', a)
                    sleep(1)
                elif (args.env == 'PathColavControl-v0'):
                    a[0] = 0.1*key_input[1]
                else:
                    a[0] = key_input[0]
                    a[1] = key_input[1]
                    try:
                        env.rewarder.params["lambda"] = np.clip(np.power(10, np.log10(env.rewarder.params["lambda"]) + key_input[2]*0.05), 0, 1)
                        env.rewarder.params["eta"] = np.clip(env.rewarder.params["eta"] + key_input[6]*0.02, 0, 4)
                    except KeyError:
                        pass
                    if args.render in {'3d', 'both'}:
                        env.viewer3d.camera_height += 0.15*key_input[3]
                        env.viewer3d.camera_height = max(0, env.viewer3d.camera_height)
                        env.viewer3d.camera_distance += 0.3*key_input[4]
                        env.viewer3d.camera_distance = max(1, env.viewer3d.camera_distance)
                        env.viewer3d.camera_angle += 0.3*key_input[5]

                    elif args.render == '2d':
                        env.viewer2d.camera_zoom += 0.1*key_input[4]
                        env.viewer2d.camera_zoom = max(0, env.viewer2d.camera_zoom)

                if autopilot and agent is not None:
                    if obs is None:
                        a = np.array([0.0, 0.0])
                    else:
                        a, _ = agent.predict(obs, deterministic=True)
                obs, r, done, info = env.step(a)
                if args.verbose > 0:
                    print(', '.join('{:.1f}'.format(x) for x in obs) + '(size {})'.format(len(obs)))
                recorded_env.render()
                steps += 1

                if quit: raise KeyboardInterrupt
                if done or restart: break

            env.reset()
            gym_auv.reporting.report(env, report_dir='../logs/play_results/')


    except KeyboardInterrupt:
        pass

def main(args):
    envconfig_string = args.envconfig
    custom_envconfig = _preprocess_custom_envconfig(args.envconfig) if args.envconfig is not None else {}
    env_id = 'gym_auv:' + args.env
    env_name = env_id.split(':')[-1] if ':' in env_id else env_id
    envconfig = gym_auv.SCENARIOS[env_name]['config'] if env_name in gym_auv.SCENARIOS else {}
    envconfig.update(custom_envconfig)

    NUM_CPU = 8
    EXPERIMENT_ID = str(int(time())) + args.algo.lower()
    model = {
        'ppo': PPO2,
        'ddpg': DDPG,
        'td3': TD3,
        'a2c': A2C,
        'acer': ACER,
        'acktr': ACKTR
    }[args.algo.lower()]

    if (args.mode == 'play'):
        agent = model.load(args.agent) if args.agent is not None else None
        envconfig_play = envconfig.copy()
        envconfig_play['show_indicators'] = True
        #envconfig_play['autocamera3d'] = False
        env = create_env(env_id, envconfig_play, test_mode=True, render_mode=args.render, pilot=args.pilot, verbose=args.verbose)
        print('Created environment instance')

        if args.scenario:
            env.load(args.scenario)
        vec_env = DummyVecEnv([lambda: env])
        recorded_env = VecVideoRecorder(vec_env, args.video_dir, record_video_trigger=lambda x: x == 0,
            video_length=args.video_length, name_prefix=args.video_name
        )
        play_scenario(env, recorded_env, args, agent=agent)
        recorded_env.env.close()

    elif (args.mode == 'enjoy'):
        agent = model.load(args.agent)
        # params = agent.get_parameters()
        # policy_weights = [
        #     params['model/pi_fc0/w:0'],
        #     params['model/pi_fc1/w:0'],
        #     params['model/pi/w:0']
        # ]
        # policy_biases = [
        #     params['model/pi_fc0/b:0'],
        #     params['model/pi_fc1/b:0'],
        #     params['model/pi/b:0']
        # ]
        # for param in params:
        #     print(param, params[param].shape)


        env = create_env(env_id, envconfig, test_mode=True, render_mode=args.render, pilot=args.pilot)
        if (args.scenario):
            env.load(args.scenario)
        vec_env = DummyVecEnv([lambda: env])
        recorded_env = VecVideoRecorder(vec_env, args.video_dir, record_video_trigger=lambda x: x==0 or x%1000 == 0,
            video_length=args.video_length, name_prefix=(args.env if args.video_name == 'auto' else args.video_name)
        )
        obs = recorded_env.reset()
        state = None
        done = [False for _ in range(vec_env.num_envs)]
        for i in range(args.video_length):
            if args.recurrent:
                action, _states = agent.predict(observation=obs, state=state, mask=done, deterministic=not args.stochastic)
                state = _states
            else:
                action, _states = agent.predict(obs, deterministic=not args.stochastic)
            obs, reward, done, info = recorded_env.step(action)
            recorded_env.render()
            if (args.env == 'PathGeneration-v0'):
                sleep(1)
        recorded_env.env.close()

    elif (args.mode == 'train'):
        figure_folder = os.path.join(DIR_PATH, 'logs', 'figures', args.env, EXPERIMENT_ID)
        os.makedirs(figure_folder, exist_ok=True)
        scenario_folder = os.path.join(figure_folder, 'scenarios')
        os.makedirs(scenario_folder, exist_ok=True)
        video_folder = os.path.join(DIR_PATH, 'logs', 'videos', args.env, EXPERIMENT_ID)
        video_length = 8000
        os.makedirs(video_folder, exist_ok=True)
        agent_folder = os.path.join(DIR_PATH, 'logs', 'agents', args.env, EXPERIMENT_ID)
        os.makedirs(agent_folder, exist_ok=True)
        tensorboard_log = os.path.join(DIR_PATH, 'logs', 'tensorboard', args.env, EXPERIMENT_ID)
        tensorboard_port = 6006

        if (args.nomp or model == DDPG or model == TD3):
            num_cpu = 1
            vec_env = DummyVecEnv([lambda: create_env(env_id, envconfig, pilot=args.pilot)])
        else:
            num_cpu = NUM_CPU
            vec_env = SubprocVecEnv([make_mp_env(env_id, i, envconfig, pilot=args.pilot) for i in range(num_cpu)])

            #n_envs = envconfig['n_ships']
            #vec_env = SubprocVecEnv([make_mp_env(env_id, i, envconfig, pilot=args.pilot) for i in range(n_envs)])

        if (args.agent is not None):
            agent = model.load(args.agent)
            agent.set_env(vec_env)
        else:
            if (model == PPO2):
                if args.recurrent:
                    hyperparams = {
                        # 'n_steps': 1024,
                        # 'nminibatches': 32,
                        # 'lam': 0.95,
                        # 'gamma': 0.99,
                        # 'noptepochs': 10,
                        # 'ent_coef': 0.0,
                        # 'learning_rate': 0.0003,
                        # 'cliprange': 0.2,
                        'n_steps': 1024,
                        'nminibatches': 1,
                        'lam': 0.98,
                        'gamma': 0.999,
                        'noptepochs': 4,
                        'ent_coef': 0.01,
                        'learning_rate': 2e-3,
                    }
                    class CustomLSTMPolicy(MlpLstmPolicy):
                        def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, **_kwargs):
                            super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                            net_arch=[256, 256, 'lstm', dict(vf=[64], pi=[64])],
                            **_kwargs)

                    agent = PPO2(CustomLSTMPolicy,
                        vec_env, verbose=True, tensorboard_log=tensorboard_log,
                        **hyperparams
                    )
                else:
                    hyperparams = {
                        # 'n_steps': 1024,
                        # 'nminibatches': 32,
                        # 'lam': 0.95,
                        # 'gamma': 0.99,
                        # 'noptepochs': 10,
                        # 'ent_coef': 0.0,
                        # 'learning_rate': 0.0003,
                        # 'cliprange': 0.2,
                        'n_steps': 1024,
                        'nminibatches': 32,
                        'lam': 0.98,
                        'gamma': 0.999,
                        'noptepochs': 4,
                        'ent_coef': 0.01,
                        'learning_rate': 2e-4,
                    }
                    #policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[64, 64, 64])
                    #policy_kwargs = dict(net_arch=[64, 64, 64])
                    #layers = [256, 128, 64, 32, 16, 8]
                    layers = [64, 64]
                    policy_kwargs = dict(net_arch = [dict(vf=layers, pi=layers)])
                    agent = PPO2(MlpPolicy,
                        vec_env, verbose=True, tensorboard_log=tensorboard_log,
                        **hyperparams, policy_kwargs=policy_kwargs
                    )
            elif (model == DDPG):
                hyperparams = {
                    'memory_limit': 1000000,
                    'normalize_observations': True,
                    'normalize_returns': False,
                    'gamma': 0.98,
                    'actor_lr': 0.00156,
                    'critic_lr': 0.00156,
                    'batch_size': 256,
                    'param_noise': AdaptiveParamNoiseSpec(initial_stddev=0.287, desired_action_stddev=0.287)
                }
                agent = DDPG(LnMlpPolicy,
                    vec_env, verbose=True, tensorboard_log=tensorboard_log, **hyperparams
                )
            elif (model == TD3):
                action_noise = NormalActionNoise(mean=np.zeros(2), sigma=0.1*np.ones(2))
                agent = TD3(stable_baselines.td3.MlpPolicy,
                    vec_env, verbose=True, tensorboard_log=tensorboard_log, action_noise=action_noise
                )
            elif model == A2C:
                hyperparams = {
                    'n_steps': 5,
                    'gamma': 0.995,
                    'ent_coef': 0.00001,
                    'learning_rate': 2e-4,
                }
                layers = [64, 64]
                policy_kwargs = dict(net_arch = [dict(vf=layers, pi=layers)])
                agent = A2C(MlpPolicy,
                    vec_env, verbose=True, tensorboard_log=tensorboard_log,
                    **hyperparams, policy_kwargs=policy_kwargs
                )
            elif model == ACER:
                agent = ACER(MlpPolicy, vec_env, verbose=True, tensorboard_log=tensorboard_log)
            elif model == ACKTR:
                agent = ACKTR(MlpPolicy, vec_env, verbose=True, tensorboard_log=tensorboard_log)

        print('Training {} agent on "{}"'.format(args.algo.upper(), env_id))

        n_updates = 0
        n_episodes = 0
        def callback(_locals, _globals):
            nonlocal n_updates
            nonlocal n_episodes

            sys.stdout.write('Training update: {}\r'.format(n_updates))
            sys.stdout.flush()

            _self = _locals['self']
            vec_env = _self.get_env()

            class Struct(object): pass
            report_env = Struct()
            report_env.history = []
            report_env.config = envconfig
            report_env.nsensors = report_env.config["n_sensors_per_sector"]*report_env.config["n_sectors"]
            report_env.sensor_angle = 2*np.pi/(report_env.nsensors + 1)
            report_env.last_episode = vec_env.get_attr('last_episode')[0]
            report_env.config = vec_env.get_attr('config')[0]

            env_histories = vec_env.get_attr('history')
            for episode in range(max(map(len, env_histories))):
                for env_idx in range(len(env_histories)):
                    if (episode < len(env_histories[env_idx])):
                        report_env.history.append(env_histories[env_idx][episode])
            report_env.episode = len(report_env.history) + 1

            total_t_steps = _self.get_env().get_attr('total_t_steps')[0]*num_cpu
            agent_filepath = os.path.join(agent_folder, str(total_t_steps) + '.pkl')

            if model == PPO2:
                recording_criteria = n_updates % 10 == 0
                report_criteria = True
                _self.save(agent_filepath)
            elif model == A2C or model == ACER or model == ACKTR:
                save_criteria = n_updates % 100 == 0
                recording_criteria = n_updates % 1000 == 0
                report_criteria = True
                if save_criteria:
                    _self.save(agent_filepath)
            elif model == DDPG or model == TD3:
                save_criteria = n_updates % 10000 == 0
                recording_criteria = n_updates % 50000 == 0
                report_criteria = report_env.episode > n_episodes
                if (save_criteria):
                    _self.save(agent_filepath)

            if report_env.last_episode is not None and len(report_env.history) > 0 and report_criteria:
                try:
                    gym_auv.reporting.plot_last_episode(report_env, fig_dir=scenario_folder, fig_prefix=args.env + '_ep_{}'.format(report_env.episode))
                    gym_auv.reporting.report(report_env, report_dir=figure_folder)
                    #vec_env.env_method('save', os.path.join(scenario_folder, '_ep_{}'.format(report_env.episode)))
                except OSError as e:
                    print("Ignoring reporting OSError:")
                    print(repr(e))

            if recording_criteria:
                if (args.pilot):
                    cmd = 'python run.py enjoy {} --agent "{}" --video-dir "{}" --video-name "{}" --video-length {} --algo {} --pilot {} --envconfig {}{}'.format(
                        args.env, agent_filepath, video_folder, args.env + '-' + str(total_t_steps), video_length, args.algo, args.pilot, envconfig_string,
                        ' --recurrent' if args.recurrent else ''
                    )
                else:
                    cmd = 'python run.py enjoy {} --agent "{}" --video-dir "{}" --video-name "{}" --video-length {} --algo {} --envconfig {}{}'.format(
                        args.env, agent_filepath, video_folder, args.env + '-' + str(total_t_steps), video_length, args.algo, envconfig_string,
                        ' --recurrent' if args.recurrent else ''
                    )
                subprocess.Popen(cmd)

            n_episodes = report_env.episode
            n_updates += 1

        agent.learn(
            total_timesteps=10000000,
            tb_log_name='log',
            callback=callback
        )

    elif (args.mode in ['policyplot', 'vectorfieldplot', 'streamlinesplot']):
        figure_folder = os.path.join(DIR_PATH, 'logs', 'plots', args.env, EXPERIMENT_ID)
        os.makedirs(figure_folder, exist_ok=True)
        agent = PPO2.load(args.agent)

        if args.testvals:
            testvals = json.load(open(args.testvals, 'r'))
            valuegrid = list(ParameterGrid(testvals))
            for valuedict in valuegrid:
                customconfig = envconfig.copy()
                customconfig.update(valuedict)
                env = create_env(env_id, envconfig, test_mode=True, pilot=args.pilot)
                valuedict_str = '_'.join((key + '-' + str(val) for key, val in valuedict.items()))

                print('Running {} test for {}...'.format(args.mode, valuedict_str))

                if args.mode == 'policyplot':
                    gym_auv.reporting.plot_actions(env, agent, fig_dir=figure_folder, fig_prefix=valuedict_str)
                elif args.mode == 'vectorfieldplot':
                    gym_auv.reporting.plot_vector_field(env, agent, fig_dir=figure_folder, fig_prefix=valuedict_str)
                elif args.mode == 'streamlinesplot':
                    gym_auv.reporting.plot_streamlines(env, agent, fig_dir=figure_folder, fig_prefix=valuedict_str)

        else:
            env = create_env(env_id, envconfig, test_mode=True, pilot=args.pilot)
            with open(os.path.join(figure_folder, 'config.json'), 'w') as f:
                json.dump(env.config, f)

            if args.mode == 'policyplot':
                gym_auv.reporting.plot_actions(env, agent, fig_dir=figure_folder)
            elif args.mode == 'vectorfieldplot':
                gym_auv.reporting.plot_vector_field(env, agent, fig_dir=figure_folder)
            elif args.mode == 'streamlinesplot':
                gym_auv.reporting.plot_streamlines(env, agent, fig_dir=figure_folder)


        print('Output folder: ', figure_folder)

    elif (args.mode == 'test'):
        figure_folder = os.path.join(DIR_PATH, 'logs', 'tests', args.env, EXPERIMENT_ID)
        scenario_folder = os.path.join(figure_folder, 'scenarios')
        os.makedirs(figure_folder, exist_ok=True)
        os.makedirs(scenario_folder, exist_ok=True)

        if (not args.onlyplot):
            agent = model.load(args.agent)

        def create_test_env(envconfig=envconfig):
            print('Creating test environment: ' + env_id)
            env = create_env(env_id, envconfig, test_mode=True, render_mode=args.render if args.video else None, pilot=args.pilot)
            vec_env = DummyVecEnv([lambda: env])
            if args.video:
                recorded_env = VecVideoRecorder(vec_env, args.video_dir, record_video_trigger=lambda x: x == 0,
                video_length=args.video_length, name_prefix=args.video_name
            )
            active_env = recorded_env if args.video else vec_env

            return env, active_env

        failed_tests = []
        def run_test(id, reset=True, report_dir=figure_folder, scenario=None, max_t_steps=None):
            nonlocal failed_tests

            if scenario is not None:
                env, active_env = create_test_env()
                obs = active_env.reset()
                env.load(args.scenario)
                print('Loaded', args.scenario)
            else:
                env, active_env = create_test_env()
                if reset:
                    obs = active_env.reset()
                else:
                    obs = env.observe()

            gym_auv.reporting.plot_scenario(env, fig_dir=scenario_folder, fig_postfix=id, show=args.onlyplot)
            if args.onlyplot:
                return
            cumulative_reward = 0
            t_steps = 0
            if max_t_steps is None:
                done = False
            else:
                done = t_steps > max_t_steps

            while not done:
                action, _states = agent.predict(obs, deterministic=not args.stochastic)
                obs, reward, done, info = active_env.step(action)
                if args.video:
                    active_env.render()
                t_steps += 1
                cumulative_reward += reward[0]
                report_msg = '{:<20}{:<20}{:<20.2f}{:<20.2%}{:<20}{:<20.2f}{:<20.2f}\r'.format(
                    id, t_steps, cumulative_reward, info[0]['progress'], info[0]['collisions'], info[0]['cross_track_error'], info[0]['heading_error']*180/np.pi)
                sys.stdout.write(report_msg)
                sys.stdout.flush()

            gym_auv.reporting.report(env, report_dir=report_dir)
            gym_auv.reporting.plot_last_episode(env, fig_dir=scenario_folder, fig_prefix=(args.env + '_' + id))
            env.save(os.path.join(scenario_folder, id))
            if (info[0]['collisions']):
                failed_tests.append(id)
                with open(os.path.join(figure_folder, 'failures.txt'), 'w') as f:
                    f.write(', '.join(map(str, failed_tests)))

            return copy.deepcopy(env.last_episode)

        print('Testing scenario "{}" for {} episodes.\n '.format(args.env, args.episodes))
        report_msg_header = '{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}'.format('Episode', 'Timesteps', 'Cum. Reward', 'Progress', 'Collisions', 'CT-Error [m]', 'H-Error [deg]')
        print(report_msg_header)
        print('-'*len(report_msg_header))

        if args.testvals:
            testvals = json.load(open(args.testvals, 'r'))
            valuegrid = list(ParameterGrid(testvals))

        if args.scenario:
            if args.testvals:
                episode_dict = {}
                for valuedict in valuegrid:
                    customconfig = envconfig.copy()
                    customconfig.update(valuedict)
                    env, active_env = create_test_env(envconfig=customconfig)
                    valuedict_str = '_'.join((key + '-' + str(val) for key, val in valuedict.items()))

                    colorval = -np.log10(valuedict['reward_lambda']) #should be general

                    rep_subfolder = os.path.join(figure_folder, valuedict_str)
                    os.makedirs(rep_subfolder, exist_ok=True)
                    for episode in range(args.episodes):
                        last_episode = run_test(valuedict_str + '_ep' + str(episode), report_dir=rep_subfolder)
                        episode_dict[valuedict_str] = [last_episode, colorval]
                print('Plotting all')
                gym_auv.reporting.plot_last_episode(env, fig_dir=scenario_folder, fig_prefix=(args.env + '_all_agents'), episode_dict=episode_dict)

            else:
                env, active_env = create_test_env()
                run_test("ep0", reset=True, scenario=args.scenario)

        else:
            if args.testvals:
                episode_dict = {}
                agent_index = 1
                for valuedict in valuegrid:
                    customconfig = envconfig.copy()
                    customconfig.update(valuedict)
                    env, active_env = create_test_env(envconfig=customconfig)
                    valuedict_str = '_'.join((key + '-' + str(val) for key, val in valuedict.items()))

                    colorval = np.log10(valuedict['reward_lambda']) #should be general

                    rep_subfolder = os.path.join(figure_folder, valuedict_str)
                    os.makedirs(rep_subfolder, exist_ok=True)
                    for episode in range(args.episodes):
                         last_episode = run_test(valuedict_str + '_ep' + str(episode), report_dir=rep_subfolder)
                    episode_dict['Agent ' + str(agent_index)] = [last_episode, colorval]
                    agent_index += 1

                gym_auv.reporting.plot_last_episode(env, fig_dir=figure_folder, fig_prefix=(args.env + '_all_agents'), episode_dict=episode_dict)
            else:
                env, active_env = create_test_env()
                for episode in range(args.episodes):
                    run_test('ep' + str(episode))

        if args.video and active_env:
            active_env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode',
        help='Which program mode to run.',
        choices=['play', 'train', 'enjoy', 'test', 'policyplot', 'vectorfieldplot', 'streamlinesplot'],
    )
    parser.add_argument(
        'env',
        help='Name of the gym environment to run.',
        choices=gym_auv.SCENARIOS.keys()
    )
    parser.add_argument(
        '--agent',
        help='Path to the RL agent to simulate.',
    )
    parser.add_argument(
        '--video-dir',
        help='Dir for output video.',
        default='../logs/videos/'
    )
    parser.add_argument(
        '--video-name',
        help='Name of output video.',
        default='auto'
    )
    parser.add_argument(
        '--algo',
        help='RL algorithm to use.',
        default='ppo',
        choices=['ppo', 'ddpg', 'td3', 'a2c', 'acer', 'acktr']
    )
    parser.add_argument(
        '--render',
        help='Rendering mode to use.',
        default='2d',
        choices=['2d', '3d', 'both'] #'both' currently broken
    )
    parser.add_argument(
        '--video-length',
        help='Timesteps to simulate.',
        type=int,
        default=1000
    )
    parser.add_argument(
        '--episodes',
        help='Number of episodes to simulate in test mode.',
        type=int,
        default=1
    )
    parser.add_argument(
        '--video',
        help='Record video for test mode.',
        action='store_true'
    )
    parser.add_argument(
        '--onlyplot',
        help='Skip simulations, only plot scenario.',
        action='store_true'
    )
    parser.add_argument(
        '--scenario',
        help='Path to scenario file containing environment data to be loaded.',
    )
    parser.add_argument(
        '--verbose',
        help='Print debugging information.',
        action='store_true'
    )
    parser.add_argument(
        '--envconfig',
        help='Override environment config parameters.',
        nargs='*'
    )
    parser.add_argument(
        '--nomp',
        help='Only use single CPU core for training.',
        action='store_true'
    )
    parser.add_argument(
        '--stochastic',
        help='Use stochastic actions.',
        action='store_true'
    )
    parser.add_argument(
        '--recurrent',
        help='Use RNN for policy network.',
        action='store_true'
    )
    parser.add_argument(
        '--pilot',
        help='If training in a controller environment, this is the pilot agent to control.',
    )
    parser.add_argument(
        '--testvals',
        help='Path to JSON file containing config values to test.',
    )
    args = parser.parse_args()

    from win10toast import ToastNotifier
    toaster = ToastNotifier()
    try:
        main(args)
        toaster.show_toast("run.py", "Program is done", duration=10)
    except Exception as e:
        toaster.show_toast("run.py", "Program has crashed", duration=10)
        raise e
