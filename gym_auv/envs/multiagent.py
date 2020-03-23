import numpy as np
import gym

import gym_auv.utils.geomutils as geom
import gym_auv.utils.helpers as helpers
from gym_auv.objects.vessel import Vessel
from gym_auv.objects.path import RandomCurveThroughOrigin, Path
from gym_auv.objects.obstacles import PolygonObstacle, VesselObstacle, CircularObstacle
from gym_auv.environment import ASV_Scenario
import shapely.geometry, shapely.errors
from gym_auv.objects.rewarder import MultiRewarder

import gym_auv.rendering.render2d as render2d
import gym_auv.rendering.render3d as render3d


import os
dir_path = os.path.dirname(os.path.realpath(__file__))

class MultiAgent(ASV_Scenario):

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
        self.n_observations = len(Vessel.NAVIGATION_STATES) + 4*self.config["n_sectors"]

        self.episode = 0
        self.total_t_steps = 0
        self.t_step = 0
        self.history = []


        # Declaring attributes
        #self.obstacles = None
        self.vessel = None
        #self.path = None

        self.reached_goal = None
        self.collision = None
        self.progress = None
        self.cumulative_reward = None
        self.last_reward = None
        self.last_episode = None
        self.rng = None
        self._tmp_storage = None

        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([-1]*self.n_observations),
            high=np.array([1]*self.n_observations),
            dtype=np.float32
        )

        # Initializing rendering
        self.viewer2d = None
        self.viewer3d = None
        if self.render_mode == '2d' or self.render_mode == 'both':
            render2d.init_env_viewer(self)
        if self.render_mode == '3d' or self.render_mode == 'both':
            render3d.init_env_viewer(self, autocamera=self.config["autocamera3d"])

        self.reset()

    @property
    def path(self):
        return self.vessel.path
    @property
    def obstacles(self):
        return self.vessel.obstacles

    def _generate(self):

        print('In GENERATE in MA')

        self.vessel = Vessel(self.config, width=self.config["vessel_width"])
        prog = 0
        self.path_prog_hist = np.array([prog])
        self.max_path_prog = prog
        self.rewarder = MultiRewarder(self.vessel)

        print(f'Ownship created!')
        self.moving_obstacles = [self.vessel]
        self.static_obstacles = []

        #Adding static obstacles
        #for _ in range(8):
        #   obstacle = CircularObstacle(*helpers.generate_obstacle(self.rng, self.vessel.path, self.vessel))
        #   self.static_obstacles.append(obstacle)

        #Adding moving obstacles (ships)
        for i in range(1,20):
            #obst_speed = np.random.random()
            ship = Vessel(self.config, width=self.config["vessel_width"], index=i)
            self.moving_obstacles.append(ship)
            print(f'Ship {i} has been created')

        for i in range(1,5):
            #obst_speed = np.random.random()
            ship = Vessel(self.config, width=self.config["vessel_width"], index=i, path_length=600)
            self.moving_obstacles.append(ship)
            print(f'Ship {i} has been created')

        for ship in self.moving_obstacles:
            other_ships = [x for x in self.moving_obstacles if x.index != ship.index]
            #ship.obstacles.extend(other_ships)
            ship.obstacles = np.hstack([self.static_obstacles, other_ships])

        for ship in self.moving_obstacles:
            print(f'{ship.index} - Collision: {ship.collision}')

        #self._update()

        print('Exiting GENERATE in MA')

    def _update(self):
        #print('In UPDATE in MA')
        dt = self.config["t_step_size"]
        #print(f'Moving obstacles: {[x.index for x in self.moving_obstacles]}')
        [obst.update(dt) for obst in self.moving_obstacles if obst.index != 0]
        valid_ships = []
        for ship in self.moving_obstacles:
            if not ship.collision:
                valid_ships.append(ship)

        self.moving_obstacles = valid_ships

        for ship in self.moving_obstacles:
            other_ships = [x for x in self.moving_obstacles if x.index != ship.index]
            #ship.obstacles.extend(other_ships)
            ship.obstacles = np.hstack([self.static_obstacles, other_ships])
        #print('Exiting UPDATE in MA')

    def observe(self):
        navigation_states, reached_goal, progress = self.vessel.navigate(self.path)
        sector_closenesses, sector_velocities, sector_moving_obstacles, collision = self.vessel.perceive(self.obstacles)

        obs = np.concatenate([navigation_states, sector_closenesses, sector_velocities, sector_moving_obstacles])
        return (obs, collision, reached_goal, progress)
