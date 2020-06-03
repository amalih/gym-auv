import numpy as np
import gym

import gym_auv.utils.geomutils as geom
import gym_auv.utils.helpers as helpers
from gym_auv.objects.vessel import Vessel
from gym_auv.objects.path import RandomCurveThroughOrigin, Path
from gym_auv.objects.obstacles import PolygonObstacle, VesselObstacle, CircularObstacle
from gym_auv.environment import BaseEnvironment
import shapely.geometry, shapely.errors
from gym_auv.objects.rewarder import MultiRewarder, ColavRewarder

import gym_auv.rendering.render2d as render2d
import gym_auv.rendering.render3d as render3d


from stable_baselines import PPO2


import os
dir_path = os.path.dirname(os.path.realpath(__file__))
MAX_VESSELS = 6

class MultiAgent_PPO(BaseEnvironment):

    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second': render2d.FPS
    }


    def __init__(self, env_config, test_mode=False, render_mode='2d', verbose=False):
        """
        The __init__ method declares all class atributes and calls
        the self.reset() to intialize them properly.

        Parameters
        ----------
            env_config : dict
                Configuration parameters for the environment.
                The default values are set in __init__.py
            test_mode : bool
                If test_mode is True, the environment will not be autonatically reset
                due to too low cumulative reward or too large distance from the path.
            render_mode : {'2d', '3d', 'both'}
                Whether to use 2d or 3d rendering. 'both' is currently broken.
            verbose
                Whether to print debugging information.
        """

        self.test_mode = test_mode
        self.render_mode = render_mode
        self.verbose = verbose
        self.config = env_config

        # Setting dimension of observation vector
        self.n_observations = len(Vessel.NAVIGATION_FEATURES) + 3*self.config["n_sectors"] + ColavRewarder.N_INSIGHTS

        self.episode = 0
        self.total_t_steps = 0
        self.t_step = 0
        self.history = []


        # Declaring attributes
        #self.obstacles = None
        self.main_vessel = None
        #self.agent = None

        #self.path = None

        self.reached_goal = None
        self.collision = None
        self.progress = None
        self.cumulative_reward = None
        self.last_reward = None
        self.last_episode = None
        self.rng = None
        self._tmp_storage = None

        self._action_space = gym.spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )
        self._observation_space = gym.spaces.Box(
            low=np.array([-1]*self.n_observations),
            high=np.array([1]*self.n_observations),
            dtype=np.float32
        )

        # Initializing rendering
        self._viewer2d = None
        self._viewer3d = None
        if self.render_mode == '2d' or self.render_mode == 'both':
            render2d.init_env_viewer(self)
        if self.render_mode == '3d' or self.render_mode == 'both':
            render3d.init_env_viewer(self, autocamera=self.config["autocamera3d"])

        #self.agent = PPO2.load('C:/Users/amalih/Documents/gym-auv-master/logs/agents/MovingObstacles-v0/1589625657ppo/6547288.pkl')
        self.agent = PPO2.load('C:/Users/amalih/Documents/gym-auv-master/logs/agents/MovingObstacles-v0/1590746004ppo/4641392.pkl')

        #'C:/Users/amalih/OneDrive - NTNU/github/logs/agents/MultiAgentPPO-v0/1064190.pkl'

        self.rewarder_dict = {}

        self.reset()

    @property
    def path(self):
        return self.main_vessel.path
    @property
    def obstacles(self):
        return self.main_vessel.obstacles

    @property
    def vessel(self):
        return self.main_vessel

    def _generate(self):

        print('In GENERATE in MA')

        self.main_vessel = Vessel(self.config, width=self.config["vessel_width"])
        self.rewarder_dict[self.main_vessel.index] = ColavRewarder(self.main_vessel)
        self.rewarder = self.rewarder_dict[self.main_vessel.index]

        prog = 0
        self.path_prog_hist = np.array([prog])
        self.max_path_prog = prog

        print(f'Ownship created!')
        self.moving_obstacles = [self.main_vessel]
        self.static_obstacles = []
        self._vessel_count = 1

        #Adding moving obstacles (ships)
        curr_vessel_count = self._vessel_count
        for i in range(curr_vessel_count ,curr_vessel_count+MAX_VESSELS-3):
            ship = Vessel(self.config, width=self.config["vessel_width"], index=i, vessel_pos=self.main_vessel.position)
            self.rewarder_dict[ship.index] = ColavRewarder(ship)
            self.moving_obstacles.append(ship)
            print(f'Ship {i} has been created')
            self._vessel_count += 1


        for ship in self.moving_obstacles:
            other_ships = [x for x in self.moving_obstacles if x.index != ship.index]
            ship.obstacles = np.hstack([self.static_obstacles, other_ships])

        #Adding static obstacles
        for _ in range(8):
            obstacle = CircularObstacle(*helpers.generate_obstacle(self.rng, self.path, self.vessel, displacement_dist_std=500))
            self.static_obstacles.append(obstacle)


        print('Exiting GENERATE in MA')

    def step(self, action:list) -> (np.ndarray, float, bool, dict):
        """
        Steps the environment by one timestep. Returns observation, reward, done, info.

        Parameters
        ----------
        action : np.ndarray
        [thrust_input, torque_input].

        Returns
        -------
        obs : np.ndarray
        Observation of the environment after action is performed.
        reward : double
        The reward for performing action at his timestep.
        done : bool
        If True the episode is ended, due to either a collision or having reached the goal position.
        info : dict
        Dictionary with data used for reporting or debugging
        """

        #print(f'STEP {self.t_step}')

        action[0] = (action[0] + 1)/2 # Done to be compatible with RL algorithms that require symmetric action spaces
        if np.isnan(action).any(): action = np.zeros(action.shape)
        self.main_vessel.step(action)

        if self.agent is None:
            [obst.update_without_agent(self.config["t_step_size"]) for obst in self.moving_obstacles if obst.index != 0]

        else:
            for ship in self.moving_obstacles:
                if ship.index != 0:
                    obs = ship.observe()
                    reward = self.rewarder_dict[ship.index].calculate()
                    insight = self.rewarder_dict[ship.index].insight()
                    #print(f'Reward for ship {ship.index}: {reward} -- lambda: {insight}')
                    obs = np.concatenate([insight,obs])
                    action, _states = self.agent.predict(obs, deterministic=True)
                    action[0] = (action[0] + 1)/2
                    ship.step(action)
                    #print(f'action taken for ship {ship.index}: {action}')




        #Adding moving obstacles (ships)
        if len(self.main_vessel.reachable_vessels) < MAX_VESSELS-3:
            curr_vessel_count = self._vessel_count
            for i in range(curr_vessel_count ,curr_vessel_count+1):
                ship = Vessel(self.config, width=self.config["vessel_width"], index=i, vessel_pos=self.main_vessel.position)
                self.rewarder_dict[ship.index] = ColavRewarder(ship)
                self.moving_obstacles.append(ship)
                print(f'Ship {i} has been created')
                self._vessel_count += 1

            for ship in self.moving_obstacles:
                other_ships = [x for x in self.moving_obstacles if x.index != ship.index]
                ship.obstacles = np.hstack([self.static_obstacles, other_ships])

        self.t_step += 1

        # If the environment is dynamic, calling self.update will change it.
        self._update()


        # Getting observation vector
        obs = self.observe()
        vessel_data = self.main_vessel.req_latest_data()
        self.collision = vessel_data['collision']
        self.reached_goal = vessel_data['reached_goal']
        self.progress = vessel_data['progress']

        # Receiving agent's reward
        reward = self.rewarder.calculate()
        self.last_reward = reward
        self.cumulative_reward += reward

        info = {}
        info['collision'] = self.collision
        info['reached_goal'] = self.reached_goal
        info['progress'] = self.progress

        # Testing criteria for ending the episode
        done = self._isdone()
        self._save_latest_step()


        return (obs, reward, done, info)

    def _update(self):

        self.moving_obstacles = self.main_vessel.inrange_vessels
        self.moving_obstacles.append(self.main_vessel)

        for ship in self.moving_obstacles:
            other_ships = [x for x in self.moving_obstacles if x.index != ship.index]
            ship.obstacles = np.hstack([self.static_obstacles, other_ships])

        if (not self.t_step % 10_000 or self.agent is None) and not self.test_mode:
            directory = 'c:/users/amalih/onedrive - ntnu/github/logs/agents/MultiAgentPPO-v0/'
            latest_subdir = max([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime)
            latest_subdir = latest_subdir + '/'

            try:
                latest_agent = max([os.path.join(latest_subdir,d) for d in os.listdir(latest_subdir)], key=os.path.getmtime)
                print(latest_agent)
                self.agent = PPO2.load(latest_agent)

            except ValueError:
                pass

    def observe(self):
        reward_insight = self.rewarder.insight()
        navigation_states = self.main_vessel.navigate(self.path)
        sector_closenesses, sector_velocities = self.main_vessel.perceive(self.obstacles)
        #print(f'Lambda for main ship: {reward_insight}')

        obs = np.concatenate([reward_insight, navigation_states, sector_closenesses, sector_velocities])
        return (obs)
