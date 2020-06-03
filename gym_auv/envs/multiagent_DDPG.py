import numpy as np
import gym

import gym_auv.utils.geomutils as geom
import gym_auv.utils.helpers as helpers
from gym_auv.objects.vessel import Vessel
from gym_auv.objects.path import RandomCurveThroughOrigin, Path
from gym_auv.objects.obstacles import PolygonObstacle, VesselObstacle, CircularObstacle
from gym_auv.environment import BaseEnvironment
import shapely.geometry, shapely.errors
from gym_auv.objects.rewarder import MultiRewarder

import gym_auv.rendering.render2d as render2d
import gym_auv.rendering.render3d as render3d



import os
dir_path = os.path.dirname(os.path.realpath(__file__))

#class MultiAgent_DDPG(gym.GoalEnv):
class MultiAgent_DDPG(BaseEnvironment):

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
        self.n_observations = len(Vessel.NAVIGATION_FEATURES) + 4*self.config["n_sectors"]

        self.episode = 0
        self.total_t_steps = 0
        self.t_step = 0
        self.history = []


        # Declaring attributes
        #self.obstacles = None
        self.main_vessel = None

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
        self.viewer2d = None
        self.viewer3d = None
        if self.render_mode == '2d' or self.render_mode == 'both':
            render2d.init_env_viewer(self)
        if self.render_mode == '3d' or self.render_mode == 'both':
            render3d.init_env_viewer(self, autocamera=self.config["autocamera3d"])


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
        prog = 0
        self.path_prog_hist = np.array([prog])
        self.max_path_prog = prog
        self.rewarder = MultiRewarder(self.main_vessel)

        print(f'Ownship created!')
        self.moving_obstacles = [self.main_vessel]
        self.static_obstacles = []
        self.queued_vessels = []

        #Adding static obstacles
        #for _ in range(8):
        #   obstacle = CircularObstacle(*helpers.generate_obstacle(self.rng, self.main_vessel.path, self.main_vessel))
        #   self.static_obstacles.append(obstacle)

        self._vessel_count = 1
        #Adding moving obstacles (ships)
        curr_vessel_count = self._vessel_count
        for i in range(curr_vessel_count ,curr_vessel_count+5):
            #obst_speed = np.random.random()
            ship = Vessel(self.config, width=self.config["vessel_width"], index=i, vessel_pos=self.main_vessel.position)
            self.moving_obstacles.append(ship)
            print(f'Ship {i} has been created')
            self._vessel_count += 1


        for ship in self.moving_obstacles:
            other_ships = [x for x in self.moving_obstacles if x.index != ship.index]
            #ship.obstacles.extend(other_ships)
            ship.obstacles = np.hstack([self.static_obstacles, other_ships])


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
        print('IN STEP')
        if len(self.queued_vessels) == 0:
            current_vessel = self.main_vessel
        else:
            current_vessel = self.queued_vessels.pop(0)

        current_index = current_vessel.index
        print(f'Current vessel is ship {current_index}')

        #[vessel.update_without_agent(self.config["t_step_size"]) for vessel in self.moving_obstacles if vessel.index != current_index]

        action[0] = (action[0] + 1)/2
        current_vessel.step(action)

        reward = self.rewarder.calculate(current_vessel)
        self.cumulative_reward += reward


        vessel_data = self.main_vessel.req_latest_data()
        self.collision = vessel_data['collision']
        self.reached_goal = vessel_data['reached_goal']
        self.progress = vessel_data['progress']

        info = {}
        info['collision'] = self.collision
        info['reached_goal'] = self.reached_goal
        info['progress'] = self.progress

        done = self._isdone()
        self._save_latest_step()

        self.moving_obstacles = self.main_vessel.nearby_vessels

        #Adding moving obstacles (ships)
        if not self.t_step % 150:
            #print(f'Time step: {self.t_step}, position of vessel: {self.main_vessel.position}')
            curr_vessel_count = self._vessel_count
            for i in range(curr_vessel_count ,curr_vessel_count+5):
                #obst_speed = np.random.random()
                ship = Vessel(self.config, width=self.config["vessel_width"], index=i, vessel_pos=self.main_vessel.position)

                self.moving_obstacles.append(ship)
                print(f'Ship {i} has been created')
                self._vessel_count += 1


            for ship in self.moving_obstacles:
                other_ships = [x for x in self.moving_obstacles if x.index != ship.index]
                #ship.obstacles.extend(other_ships)
                ship.obstacles = np.hstack([self.static_obstacles, other_ships])

        if len(self.queued_vessels) == 0:
            self.queued_vessels = [x for x in self.moving_obstacles if x.index != 0]
            next_vessel = self.main_vessel
        else:
            next_vessel = self.queued_vessels[0]
        obs = next_vessel.observe()

        self.t_step += 1
        print('EXITIG STEP')

        return (obs, reward, done, info)





        # [obst.update(self.config["t_step_size"]) for obst in self.moving_obstacles if obst.index != 0]
        #
        #
        # action[0] = (action[0] + 1)/2 # Done to be compatible with RL algorithms that require symmetric action spaces
        # if np.isnan(action).any(): action = np.zeros(action.shape)
        # self.main_vessel.step(action)
        #
        #
        # # Getting observation vector
        # obs = self.observe()
        # vessel_data = self.main_vessel.req_latest_data()
        # self.collision = vessel_data['collision']
        # self.reached_goal = vessel_data['reached_goal']
        # self.progress = vessel_data['progress']
        #
        # # Receiving agent's reward
        # reward = self.rewarder.calculate()
        # self.last_reward = reward
        # self.cumulative_reward += reward
        #
        # info = {}
        # info['collision'] = self.collision
        # info['reached_goal'] = self.reached_goal
        # info['progress'] = self.progress
        #
        # # Testing criteria for ending the episode
        # done = self._isdone()
        #
        # self._save_latest_step()
        # # If the environment is dynamic, calling self.update will change it.
        # self._update()
        #
        # self.t_step += 1
        #
        # #Adding moving obstacles (ships)
        # if not self.t_step % 100:
        #     #print(f'Time step: {self.t_step}, position of vessel: {self.main_vessel.position}')
        #     curr_vessel_count = self._vessel_count
        #     for i in range(curr_vessel_count ,curr_vessel_count+5):
        #         #obst_speed = np.random.random()
        #         ship = Vessel(self.config, width=self.config["vessel_width"], index=i, vessel_pos=self.main_vessel.position)
        #
        #         self.moving_obstacles.append(ship)
        #         print(f'Ship {i} has been created')
        #         self._vessel_count += 1
        #
        #
        #     for ship in self.moving_obstacles:
        #         other_ships = [x for x in self.moving_obstacles if x.index != ship.index]
        #         #ship.obstacles.extend(other_ships)
        #         ship.obstacles = np.hstack([self.static_obstacles, other_ships])
        #
        # return (obs, reward, done, info)

    def _update(self):
        valid_ships = [self.main_vessel]
        for ship in self.moving_obstacles:
            if (not ship.collision) and ship.index != 0:# and ship.reachable :
                valid_ships.append(ship)
    #    print(f'Time: {self.t_step}')
        print([x.index for x in valid_ships])

        self.moving_obstacles = valid_ships

        for ship in self.moving_obstacles:
            other_ships = [x for x in self.moving_obstacles if x.index != ship.index]
            #ship.obstacles.extend(other_ships)
            ship.obstacles = np.hstack([self.static_obstacles, other_ships])
        #print('Exiting UPDATE in MA')

    def observe(self):
        navigation_states = self.main_vessel.navigate(self.path)
        sector_closenesses, sector_velocities, sector_moving_obstacles = self.main_vessel.perceive(self.obstacles)

        obs = np.concatenate([navigation_states, sector_closenesses, sector_velocities, sector_moving_obstacles])
        return (obs)
