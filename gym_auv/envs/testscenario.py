import numpy as np
import gym

import gym_auv.utils.geomutils as geom
from gym_auv.objects.vessel import Vessel
from gym_auv.objects.path import RandomCurveThroughOrigin, Path
from gym_auv.objects.obstacles import CircularObstacle, VesselObstacle
from gym_auv.environment import BaseEnvironment
from stable_baselines import PPO2
from gym_auv.objects.rewarder import ColavRewarder

import os
dir_path = os.path.dirname(os.path.realpath(__file__))

TERRAIN_DATA_PATH = './resources/terrain.npy'

class TestScenario1(BaseEnvironment):
    def _generate(self):
        self.path = Path([[0, 1100], [0, 1100]])

        init_state = self.path(0)
        init_angle = self.path.get_direction(0)

        self.vessel = Vessel(self.config, np.hstack([init_state, init_angle]))
        prog = self.path.get_closest_arclength(self.vessel.position)
        self.path_prog_hist = np.array([prog])
        self.max_path_prog = prog

        obst_arclength = 30
        for o in range(20):
            obst_radius = 10 + 10*o**1.5
            obst_arclength += obst_radius*2 + 30
            obst_position = self.path(obst_arclength)
            self.obstacles.append(CircularObstacle(obst_position, obst_radius))

class TestScenario2(BaseEnvironment):
    def _generate(self):

        waypoint_array = []
        for t in range(500):
            x = t*np.cos(t/100)
            y = 2*t
            waypoint_array.append([x, y])

        waypoints = np.vstack(waypoint_array).T
        self.path = Path(waypoints)

        init_state = self.path(0)
        init_angle = self.path.get_direction(0)

        self.vessel = Vessel(self.config, np.hstack([init_state, init_angle]))
        prog = self.path.get_closest_arclength(self.vessel.position)
        self.path_prog_hist = np.array([prog])
        self.max_path_prog = prog

        obst_arclength = 30
        obst_radius = 5
        while True:
            obst_arclength += 2*obst_radius
            if (obst_arclength >= self.path.length):
                break

            obst_displacement_dist = 140 - 120 / (1 + np.exp(-0.005*obst_arclength))

            obst_position = self.path(obst_arclength)
            obst_displacement_angle = self.path.get_direction(obst_arclength) - np.pi/2
            obst_displacement = obst_displacement_dist*np.array([
                np.cos(obst_displacement_angle),
                np.sin(obst_displacement_angle)
            ])

            self.obstacles.append(CircularObstacle(obst_position + obst_displacement, obst_radius))
            self.obstacles.append(CircularObstacle(obst_position - obst_displacement, obst_radius))

class TestScenario3(BaseEnvironment):
    def _generate(self):
        waypoints = np.vstack([[0, 0], [0, 500]]).T
        self.path = Path(waypoints)

        init_state = self.path(0)
        init_angle = self.path.get_direction(0)

        self.vessel = Vessel(self.config, np.hstack([init_state, init_angle]))
        prog = self.path.get_closest_arclength(self.vessel.position)
        self.path_prog_hist = np.array([prog])
        self.max_path_prog = prog

        N_obst = 20
        N_dist = 100
        for n in range(N_obst + 1):
            obst_radius = 25
            angle = np.pi/4 +  n/N_obst * np.pi/2
            obst_position = np.array([np.cos(angle)*N_dist, np.sin(angle)*N_dist])
            self.obstacles.append(CircularObstacle(obst_position, obst_radius))

class TestScenario4(BaseEnvironment):
    def _generate(self):
        waypoints = np.vstack([[0, 0], [0, 500]]).T
        self.path = Path(waypoints)

        init_state = self.path(0)
        init_angle = self.path.get_direction(0)

        self.vessel = Vessel(self.config, np.hstack([init_state, init_angle]))
        prog = self.path.get_closest_arclength(self.vessel.position)
        self.path_prog_hist = np.array([prog])
        self.max_path_prog = prog

        N_obst = 20
        N_dist = 100
        for n in range(N_obst+1):
            obst_radius = 25
            angle = n/N_obst * 2*np.pi
            if (abs(angle < 3/2*np.pi) < np.pi/12):
                continue
            obst_position = np.array([np.cos(angle)*N_dist, np.sin(angle)*N_dist])
            self.obstacles.append(CircularObstacle(obst_position, obst_radius))

class TwoVessel_HeadOn(BaseEnvironment):
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

    #    self.agent = PPO2.load('C:/Users/amalih/Documents/gym-auv-master/logs/agents/MovingObstacles-v0/1589625657ppo/6547288.pkl')
        #self.agent = PPO2.load('C:/Users/amalih/Documents/gym-auv-master/logs/agents/MovingObstacles-v0/1590746004ppo/2927552.pkl')
    #    self.agent = PPO2.load('C:/Users/amalih/Documents/gym-auv-master/logs/agents/MovingObstacles-v0/1590827849ppo/4070808.pkl')
        #'C:/Users/amalih/OneDrive - NTNU/github/logs/agents/MultiAgentPPO-v0/1064190.pkl'

        #self.agent = PPO2.load('C:/Users/amalih/Documents/gym-auv-master/logs/agents/MovingObstacles-v0/1590705511ppo/4425456.pkl')
        #self.agent = PPO2.load('C:/Users/amalih/Documents/gym-auv-master/gym-auv-master/logs/agents/MovingObstacles-v0/1589130704ppo/6916896.pkl')
        #self.agent = PPO2.load('C:/Users/amalih/Documents/gym-auv-master/gym-auv-master/logs/agents/MovingObstacles-v0/1589031909ppo/1760568.pkl')
        self.agent = PPO2.load('C:/Users/amalih/OneDrive - NTNU/github/logs/agents/MultiAgentPPO-v0/1591171914ppo/79288.pkl')

        self.rewarder_dict = {}

        self.reset()
        print('Init done')

    def _generate(self):

        waypoints1 = np.vstack([[0, 0], [0, 500]]).T
        path1 = Path(waypoints1)

        init_pos1 = path1(0)
        init_angle1 = path1.get_direction(0)
        init_state1 = np.hstack([init_pos1, init_angle1])

        self.main_vessel = Vessel(self.config, init_state=init_state1, init_path=path1, width=2) #self.config["vessel_width"])
        self.main_vessel.path = path1
        self.rewarder_dict[self.main_vessel.index] = ColavRewarder(self.main_vessel)
        self.rewarder = self.rewarder_dict[self.main_vessel.index]

        prog = 0
        self.path_prog_hist = np.array([prog])
        self.max_path_prog = prog
        self.moving_obstacles = [self.main_vessel]

        #Adding moving obstacle

        waypoints2 = np.vstack([[0, 150], [0,-400]]).T
        path2 = Path(waypoints2)

        init_pos2 = path2(0)
        init_angle2 = path2.get_direction(0)
        init_state2 = np.hstack([init_pos2, init_angle2])

        vessel = Vessel(self.config, init_state=init_state2, init_path=path2,  index=1, width=2) #self.config["vessel_width"])
        self.rewarder_dict[vessel.index] = ColavRewarder(vessel)
        self.moving_obstacles.append(vessel)
        vessel.path = path2

        for vessel in self.moving_obstacles:
            other_vessels = [x for x in self.moving_obstacles if x.index != vessel.index]
            vessel.obstacles = np.hstack([other_vessels])

        print('Generated vessels!')

        #self._update()

    @property
    def path(self):
        return self.main_vessel.path
    @property
    def obstacles(self):
        return self.main_vessel.obstacles

    @property
    def vessel(self):
        return self.main_vessel

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


        action[0] = (action[0] + 1)/2 # Done to be compatible with RL algorithms that require symmetric action spaces
        if np.isnan(action).any(): action = np.zeros(action.shape)
        self.main_vessel.step(action)

        for vessel in self.moving_obstacles:
            if vessel.index != 0:
                obs = vessel.observe()
                reward = self.rewarder_dict[vessel.index].calculate()
                insight = self.rewarder_dict[vessel.index].insight()
                #print(f'Reward for vessel {vessel.index}: {reward} -- lambda: {insight}')
                obs = np.concatenate([insight,obs])
                action, _states = self.agent.predict(obs, deterministic=True)
                action[0] = (action[0] + 1)/2
                vessel.step(action)

        # Testing criteria for ending the episode
        done = self._isdone()
        self._save_latest_step()

        # Getting observation vector
        obs = self.observe()
        vessel_data = self.main_vessel.req_latest_data()
        self.collision = vessel_data['collision']
        self.reached_goal = vessel_data['reached_goal']
        self.progress = vessel_data['progress']

        # Receiving agent's reward
        reward = self.rewarder.calculate()
        self.last_reward = reward
        #self.cumulative_reward += reward

        info = {}
        info['collision'] = self.collision
        info['reached_goal'] = self.reached_goal
        info['progress'] = self.progress

        self.t_step += 1

        return (obs, reward, done, info)

class EmptyScenario(BaseEnvironment):

    def _generate(self):
        waypoints = np.vstack([[25, 10], [25, 200]]).T
        self.path = Path(waypoints)

        init_state = self.path(0)
        init_angle = self.path.get_direction(0)

        self.vessel = Vessel(self.config, np.hstack([init_state, init_angle]), width=self.config["vessel_width"])
        prog = self.path.get_closest_arclength(self.vessel.position)
        self.path_prog_hist = np.array([prog])
        self.max_path_prog = prog

        if self.render_mode == '3d':
            self.all_terrain = np.zeros((50, 50), dtype=float)
            self._viewer3d.create_world(self.all_terrain, 0, 0, 50, 50)

class DebugScenario(BaseEnvironment):
    def _generate(self):
        waypoints = np.vstack([[250, 100], [250, 300]]).T
        self.path = Path(waypoints)

        init_state = self.path(0)
        init_angle = self.path.get_direction(0)

        self.vessel = Vessel(self.config, np.hstack([init_state, init_angle]), width=self.config["vessel_width"])
        prog = self.path.get_closest_arclength(self.vessel.position)
        self.path_prog_hist = np.array([prog])
        self.max_path_prog = prog

        self.obstacles = []
        self.vessel_obstacles = []

        for vessel_idx in range(5):
            other_vessel_trajectory = []
            trajectory_shift = self.rng.rand()()*2*np.pi
            trajectory_radius = self.rng.rand()()*40 + 30
            trajectory_speed = self.rng.rand()()*0.003 + 0.003
            for i in range(10000):
                #other_vessel_trajectory.append((10*i, (250, 400-10*i)))
                other_vessel_trajectory.append((1*i, (
                    250 + trajectory_radius*np.cos(trajectory_speed*i + trajectory_shift),
                    150 + 70*vessel_idx + trajectory_radius*np.sin(trajectory_speed*i + trajectory_shift)
                )))
            other_vessel_obstacle = VesselObstacle(width=6, trajectory=other_vessel_trajectory)

            self.obstacles.append(other_vessel_obstacle)
            self.vessel_obstacles.append(other_vessel_obstacle)

        for vessel_idx in range(5):
            other_vessel_trajectory = []
            trajectory_start = self.rng.rand()()*200 + 150
            trajectory_speed = self.rng.rand()()*0.03 + 0.03
            trajectory_shift = 10*self.rng.rand()()
            for i in range(10000):
                other_vessel_trajectory.append((i, (245 + 2.5*vessel_idx + trajectory_shift, trajectory_start-10*trajectory_speed*i)))
            other_vessel_obstacle = VesselObstacle(width=6, trajectory=other_vessel_trajectory)

            self.obstacles.append(other_vessel_obstacle)
            self.vessel_obstacles.append(other_vessel_obstacle)

        if self.render_mode == '3d':
            self.all_terrain = np.load(TERRAIN_DATA_PATH)[1950:2450, 5320:5820]/7.5
            #terrain = np.zeros((500, 500), dtype=float)

            # for x in range(10, 40):
            #     for y in range(10, 40):
            #         z = 0.5*np.sqrt(max(0, 15**2 - (25.0-x)**2 - (25.0-y)**2))
            #         terrain[x][y] = z
            self._viewer3d.create_world(self.all_terrain, 0, 0, 500, 500)
