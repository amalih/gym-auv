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
        self.main_vessel = None

        self.reached_goal = None
        self.collision = None
        self.progress = None
        self.last_episode = None
        self.rng = None
        self._tmp_storage = None


        # Initializing rendering
        self._viewer2d = None
        self._viewer3d = None
        if self.render_mode == '2d' or self.render_mode == 'both':
            render2d.init_env_viewer(self)
        if self.render_mode == '3d' or self.render_mode == 'both':
            render3d.init_env_viewer(self, autocamera=self.config["autocamera3d"])


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

        print(f'Ownvessel created!')
        self.moving_obstacles = [self.main_vessel]
        self.static_obstacles = []
        self._vessel_count = 1

        #Adding moving obstacles (vessels)
        curr_vessel_count = self._vessel_count
        for i in range(curr_vessel_count, curr_vessel_count+MAX_VESSELS-1):
            vessel = Vessel(self.config, width=self.config["vessel_width"], index=i, vessel_pos=self.main_vessel.position)
            self.rewarder_dict[vessel.index] = ColavRewarder(vessel)
            self.moving_obstacles.append(vessel)
            print(f'vessel {i} has been created')
            self._vessel_count += 1


        for vessel in self.moving_obstacles:
            other_vessels = [x for x in self.moving_obstacles if x.index != vessel.index]
            vessel.obstacles = np.hstack([self.static_obstacles, other_vessels])

        #Adding static obstacles
        for _ in range(8):
            obstacle = CircularObstacle(*helpers.generate_obstacle(self.rng, self.path, self.vessel, displacement_dist_std=500))
            self.static_obstacles.append(obstacle)


        print('Exiting GENERATE in MA')

    def _take_action(self, actions) -> (np.ndarray, float, bool, dict):
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

        i = 0
        for vessel in self.moving_obstacles:
            vessel.step(actions[i])
            i += 1

        # If the environment is dynamic, calling self.update will change it.
        self._update()

        # Getting observation vector
        new_states, rewards, dones = self.observe()

        self.t_step += 1

        # Saving episode
        self._save_latest_step()


        return (new_states, rewards, dones, info)

    # Removes vessels far away and adds new ones if less than MAX_VESSELS
    def _update(self):

        temp_moving_obstacles = [self.main_vessel]

        for vessel in self.moving_distances:
            if len(vessel.reachable_vessels):
                temp_moving_obstacles.append(vessel)
        self.moving_obstacles = temp_moving_obstacles

        while len(self.moving_obstacles) < MAX_VESSELS:
            new_vessel_count = self._vessel_count + 1
            vessel = Vessel(self.config, width=self.config["vessel_width"], index=new_vessel_count, vessel_pos=self.main_vessel.position)
            self.rewarder_dict[vessel.index] = ColavRewarder(vessel)
            self.moving_obstacles.append(vessel)
            print(f'vessel {i} has been created')
            self._vessel_count += 1

        for vessel in self.moving_obstacles:
            other_vessels = [x for x in self.moving_obstacles if x.index != vessel.index]
            vessel.obstacles = np.hstack([self.static_obstacles, other_vessels])

        self.moving_obstacles.sort(key=lambda x: x.index)

        return self.moving_obstacles

    # Returns observations and rewards for all moving obstacles
    def _get_obs_rew(self):
        obs_vec = []
        rew_vec = []
        done_vec = []
        self.moving_obstacles.sort(key=lambda x: x.index)
        for vessel in self.moving_obstacles:
            obs, done = vessel.observe()
            reward, insight = self.rewarder_dict[vessel.index].calculate()
            obs = np.concatenate([insight,obs])
            obs_vec.append(obs)
            rew_vec.append(reward)
            done_vec.append(done)

        return (obs_vec, rew_vec, done_vec)
