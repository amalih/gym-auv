"""
This module implements an AUV that is simulated in the horizontal plane.
"""
import numpy as np
import numpy.linalg as linalg
from itertools import islice, chain, repeat
import shapely.geometry, shapely.errors, shapely.strtree, shapely.ops, shapely.prepared

from gym_auv.objects.path import RandomCurveThroughOrigin, Path, RandomCurveFromEdge
import gym_auv.utils.constants as const
import gym_auv.utils.geomutils as geom
from gym_auv.objects.obstacles import *
from gym.utils import seeding
import math

from stable_baselines import PPO2

from gym_auv.objects.path import Path

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2

import glob
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

deg2rad = math.pi/180

def _odesolver45(f, y, h):
    """Calculate the next step of an IVP of a time-invariant ODE with a RHS
    described by f, with an order 4 approx. and an order 5 approx.
    Parameters:
        f: function. RHS of ODE.
        y: float. Current position.
        h: float. Step length.
    Returns:
        q: float. Order 2 approx.
        w: float. Order 3 approx.
    """
    s1 = f(y)
    s2 = f(y+h*s1/4.0)
    s3 = f(y+3.0*h*s1/32.0+9.0*h*s2/32.0)
    s4 = f(y+1932.0*h*s1/2197.0-7200.0*h*s2/2197.0+7296.0*h*s3/2197.0)
    s5 = f(y+439.0*h*s1/216.0-8.0*h*s2+3680.0*h*s3/513.0-845.0*h*s4/4104.0)
    s6 = f(y-8.0*h*s1/27.0+2*h*s2-3544.0*h*s3/2565+1859.0*h*s4/4104.0-11.0*h*s5/40.0)
    w = y + h*(25.0*s1/216.0+1408.0*s3/2565.0+2197.0*s4/4104.0-s5/5.0)

    q = y + h*(16.0*s1/135.0+6656.0*s3/12825.0+28561.0*s4/56430.0-9.0*s5/50.0+2.0*s6/55.0)
    return w, q

def _standardize_intersect(intersect):
    if intersect.is_empty:
        return []
    elif isinstance(intersect, shapely.geometry.LineString):
        return [shapely.geometry.Point(intersect.coords[0])]
    elif isinstance(intersect, shapely.geometry.Point):
        return [intersect]
    else:
        return list(intersect.geoms)

def _feasibility_pooling(x, width, theta):
    N_sensors = x.shape[0]
    sort_idx = np.argsort(x)
    for idx in sort_idx:
        #if self.sensor_moving_measurements[idx]:
        #    required_width = 2*width
        #else:
        #    required_width = width
        surviving = x > x[idx] + width
        d = x[idx]*theta
        opening_width = 0
        opening_span = 0
        opening_start = -theta*(N_sensors-1)/2
        found_opening = False
        for isensor, sensor_survives in enumerate(surviving):
            if sensor_survives:
                opening_width += d
                opening_span += theta
                if opening_width > width:
                    opening_center = opening_start + opening_span/2
                    if abs(opening_center) < theta*(N_sensors-1)/4:
                        found_opening = True
            else:
                opening_width += 0.5*d
                opening_span += 0.5*theta
                if opening_width > width:
                    opening_center = opening_start + opening_span/2
                    if abs(opening_center) < theta*(N_sensors-1)/4:
                        found_opening = True
                opening_width = 0
                opening_span = 0
                opening_start = -theta*(N_sensors-1)/2 + isensor*theta

        if not found_opening:
            return max(0, x[idx])

    return max(0, np.max(x))

def _simulate_sensor(self, sensor_angle, p0_point, sensor_range, obstacles):
    sensor_endpoint = (
        p0_point.x + np.cos(sensor_angle)*sensor_range,
        p0_point.y + np.sin(sensor_angle)*sensor_range
    )
    sector_ray = shapely.geometry.LineString([p0_point, sensor_endpoint])

    obst_intersections = [sector_ray.intersection(elm.boundary) for elm in obstacles]
    obst_intersections = list(map(_standardize_intersect, obst_intersections))
    obst_references = list(chain.from_iterable(repeat(obstacles[i], len(obst_intersections[i])) for i in range(len(obst_intersections))))
    obst_intersections = list(chain(*obst_intersections))

    if obst_intersections:
        measured_distance, intercept_idx = min((float(p0_point.distance(elm)), i) for i, elm in enumerate(obst_intersections))
        obstacle = obst_references[intercept_idx]
        if not obstacle.static:
            obst_speed_homogenous = geom.to_homogeneous([obstacle.dx, obstacle.dy])
            obst_speed_rel_homogenous = geom.Rz(-sensor_angle - np.pi/2).dot(obst_speed_homogenous)
            obst_speed_vec_rel = geom.to_cartesian(obst_speed_rel_homogenous)
            situation = 0

                #     ship_domain = self.find_ship_domain(obstacle)
                #     print(f'Ship domain for obstacle {obstacle.index}: {ship_domain} (position: {self.position})')
                #     #if ship_domain.contains(shapely.geometry.Point(self.x, self.y)):
                #         norm_dot_prod = (self.velocity/self.speed) @ (obstacle.velocity/obstacle.speed)
                #         cross_prod = np.cross(self.velocity/self.speed), (obstacle.velocity/obstacle.speed))
                #         print(f'Own velocities:{self.velocity} -- Ship {obstacle.index} velocities: {obstacle.velocity}')
                #         #norm_dot_prod = dot_prod/(abs(self.speed)*abs(obstacle.speed))
                #         #print(f'Ship is inside of ship domain, and the norm_dot_prod = {norm_dot_prod}')
                #         print(f'Norm_dot_prod = {norm_dot_prod} -- Cross_prod = {cross_prod}')
                #         if norm_dot_prod >= np.cos(5*deg2rad)
                #             situation = 1 #overtaking
                #         elif norm_dot_prod <= np.cos(175*deg2rad):
                #             situation = 2 #head-on
                #         else:
                #             if cross_prod < 0:
                #                 situation = 3 #crossing, stand on
                #             if cross_prod > 0:
                #                 situation = 4 #crossing, give way
                #
                #pass

            #moving_obstacle = True
        else:
            obst_speed_vec_rel = (0, 0)
            #moving_obstacle = False

    else:
        measured_distance = sensor_range
        obst_speed_vec_rel = (0, 0)
        situation = 0
        #moving_obstacle = False



    return (measured_distance, obst_speed_vec_rel) #, situation)

def distance(v1, v2):
    return math.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2)

class Vessel():

    NAVIGATION_FEATURES = [
        'surge_velocity',
        'sway_velocity',
        'yaw_rate',
        'look_ahead_heading_error',
        'heading_error',
        'cross_track_error'
    ]

    def __init__(self, config:dict, init_state:np.ndarray=None, init_path=None, width:float=4, index:int=None, path_length:float=800, vessel_pos=None) -> None:
        """
        Initializes and resets the vessel.

        Parameters
        ----------
        config : dict
            Dictionary containing the configuration parameters for
            the vessel
        init_state : np.ndarray
            The initial attitude of the veHssel [x, y, psi], where
            psi is the initial heading of the AUV.
        width : float
            The distance from the center of the AUV to its edge
            in meters.
        """
        self.path = None

        if index != None:
            self.index = index
        else:
            self.index = 0

        if init_path is not None:
            self.path == init_path

        if init_state is None:
            # Initializing path
            #if self.rng is None:
            self.seed()
            nwaypoints = int(np.floor(4*self.rng.rand() + 2))
            if vessel_pos is None:
                self.path = RandomCurveThroughOrigin(self.rng, nwaypoints, length=path_length)
            else:

                self.path = RandomCurveFromEdge(self.rng, nwaypoints, vessel_pos)

            # Initializing vessel
            init_loc = self.path(0)
            init_angle = self.path.get_direction(0)
            init_state = np.hstack([init_loc, init_angle])


        self.config = config
        self.static = False
        self.valid = True
        self.agent = None
        #self.reachable = True


        # Initializing private attributes
        self._n_sectors = self.config["n_sectors"]
        self._n_sensors = self.config["n_sensors_per_sector"]*self.config["n_sectors"]
        self._d_sensor_angle = 2*np.pi/(self._n_sensors)
        self._sensor_angles = np.array([-np.pi + (i + 1)*self._d_sensor_angle for i in range(self._n_sensors)])
        self._sector_angles = []
        self._n_sensors_per_sector = [0]*self._n_sectors
        self._sector_start_indeces = [0]*self._n_sectors
        self._sector_end_indeces = [0]*self._n_sectors
        self._sensor_internal_indeces = []
        self._sensor_interval = max(1, int(1/self.config["sensor_frequency"]))
        self._sensor_range = self.config["sensor_range"]
        self._sector_obst_dynamic = [0]*self._n_sectors
        self._sensor_frequency = self.config["sensor_frequency"]
        if self.index != 0:
            self._sensor_frequency *= 0.7


        self._width = width
        self._points = [
            (-self._width/2, -self._width/2),
            (-self._width/2, self._width/2),
            (self._width/2, self._width/2),
            (3/2*self._width, 0),
            (self._width/2, -self._width/2)
            ]

        # Calculating sensor partitioning
        last_isector = -1
        tmp_sector_angle_sum = 0
        tmp_sector_sensor_count = 0
        for isensor in range(self._n_sensors):
            isector = self.config["sector_partition_fun"](self, isensor)
            angle = self._sensor_angles[isensor]
            if isector == last_isector:
                tmp_sector_angle_sum += angle
                tmp_sector_sensor_count += 1
            else:
                if last_isector > -1:
                    self._sector_angles.append(tmp_sector_angle_sum/tmp_sector_sensor_count)
                last_isector = isector
                self._sector_start_indeces[isector] = isensor
                tmp_sector_angle_sum = angle
                tmp_sector_sensor_count = 1
            self._n_sensors_per_sector[isector] += 1
        self._sector_angles.append(tmp_sector_angle_sum/tmp_sector_sensor_count)
        self._sector_angles = np.array(self._sector_angles)
        #print(self._sector_angles)


        for isensor in range(self._n_sensors):
            isector = self.config["sector_partition_fun"](self, isensor)
            isensor_internal = isensor - self._sector_start_indeces[isector]
            self._sensor_internal_indeces.append(isensor_internal)

        for isector in range(self._n_sectors):
            self._sector_end_indeces[isector] = self._sector_start_indeces[isector] + self._n_sensors_per_sector[isector]

        # Calculating feasible closeness
        if self.config["sensor_log_transform"]:
            self._get_closeness = lambda x: 1 - np.clip(np.log(1 + x)/np.log(1 + self.config["sensor_range"]), 0, 1)
        else:
            self._get_closeness = lambda x: 1 - np.clip(x/self.config["sensor_range"], 0, 1)

        # Initializing vessel to initial position
        self.reset(init_state)

    def reset(self, init_state:np.ndarray) -> None:
        """
        Resets the vessel to the specified initial state.

        Parameters
        ----------
        init_state : np.ndarray
            The initial attitude of the veHssel [x, y, psi], where
            psi is the initial heading of the AUV.
        """
        if self.index == 1:
            init_speed = [0, 0, 0]
        else:
            init_speed = [0,0,0]
        self.init_state = np.array(init_state, dtype=np.float64)
        self.init_speed = np.array(init_speed, dtype=np.float64)



        self._collision = False
        self._progress = None
        self.smoothed_torque_change = 0
        self.smoothed_torque = 0
        self.risk = 0

        self._state = np.hstack([init_state, init_speed])
        self._prev_states = np.vstack([self._state])
        self._input = [0, 0]
        self._prev_inputs =np.vstack([self._input])
        self._last_sensor_dist_measurements = np.ones((self._n_sensors,))*self.config["sensor_range"]
        self._last_sensor_speed_measurements = np.zeros((self._n_sensors,2))
        #self._last_sensor_moving_measurements = np.zeros((self._n_sensors,))
        self._last_sector_dist_measurements = np.zeros((self._n_sectors,))
        #self._last_sector_moving_measurements = np.zeros((self._n_sectors,))
        self._last_sector_feasible_dists = np.zeros((self._n_sectors,))
        self._last_sensor_situations = np.zeros((self._n_sectors,))
        self._last_navi_state_dict = dict((state, 0) for state in Vessel.NAVIGATION_FEATURES)


        self._reached_goal = False
        self._step_counter = 0
        self._perceive_counter = 0
        self._nearby_obstacles = []

        self._boundary = self.calculate_boundary()
        self._init_boundary = copy.deepcopy(self._boundary)



    def seed(self, seed=None):
        self.rng, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action:list) -> None:

        """
        Simulates the vessel one step forward after applying the given action.

        Parameters
        ----------
        action : np.ndarray[thrust_input, torque_input]
        """
        self.input = np.array([self._thrust_surge(action[0]), self._moment_steer(action[1])])
        w, q = _odesolver45(self._state_dot, self._state, self.config["t_step_size"])

        self._state = q
        self._state[2] = geom.princip(self._state[2])

        self._prev_states = np.vstack([self._prev_states,self._state])
        self._prev_inputs = np.vstack([self._prev_inputs,self._input])

        self._step_counter += 1
        self._boundary = self.calculate_boundary()

    def _state_dot(self, state):
        psi = state[2]
        nu = state[3:]

        tau = np.array([self.input[0], 0, self.input[1]])

        eta_dot = geom.Rzyx(0, 0, geom.princip(psi)).dot(nu)
        nu_dot = const.M_inv.dot(
            tau
            #- const.D.dot(nu)
            - const.N(nu).dot(nu)
        )
        state_dot = np.concatenate([eta_dot, nu_dot])
        return state_dot

    def _thrust_surge(self, surge):
        surge = np.clip(surge, 0, 1)
        return surge*const.THRUST_MAX_AUV

    def _moment_steer(self, steer):
        steer = np.clip(steer, -1, 1)
        return steer*const.MOMENT_MAX_AUV

    def perceive(self, obstacles:list) -> (np.ndarray, np.ndarray):
        """
        Simulates the sensor suite and returns observation arrays of the environment.

        Returns
        -------
        sector_closenesses : np.ndarray
        sector_velocities : np.ndarray
        """

        # Initializing variables
        sensor_range = self.config["sensor_range"]
        p0_point = shapely.geometry.Point(*self.position)

        # Loading nearby obstacles, i.e. obstacles within the vessel's detection range
        if self._step_counter % self.config["sensor_interval_load_obstacles"] == 0:
            self._nearby_obstacles = list(filter(
                lambda obst: float(p0_point.distance(obst.boundary)) - self._width < sensor_range, obstacles
            ))

        if self.index == 0:
            self.reachable_vessels = list(filter(
                lambda obst: float(p0_point.distance(obst.boundary)) - self._width < sensor_range + 5 and isinstance(obst,Vessel), obstacles
            ))
            self.inrange_vessels = list(filter(
                lambda obst: float(p0_point.distance(obst.boundary)) - self._width < sensor_range + 20 and isinstance(obst,Vessel), obstacles
            ))

        if not self._nearby_obstacles:
            self.reachable = False
            self.last_sensor_dist_measurements = np.ones((self._n_sensors,))*self._sensor_range
            sector_feasible_distances = np.ones((self._n_sectors,))*self._sensor_range
            sector_closenesses = np.zeros((self._n_sectors,))
            sector_velocities = np.zeros((2*self._n_sectors,))
            #sector_moving_measurements = np.zeros((self._n_sectors,))

            collision = False

        else:
            #print(f'Current position, self: {self.position}')
            #print(f'Current obstacles and rel distances: {[self.calculate_distance(obstacle) for obstacle in self._nearby_obstacles]}')
            #print(f'Current positions: {[obstacle.position for obstacle in self._nearby_obstacles]}')
            for obstacle in self._nearby_obstacles:
                if self.index == 0 and isinstance(obstacle,Vessel):
                    #ship.reachable = True
                    if obstacle.speed != 0:
                        # print('--------------------------')
                        # print(f'For ship {obstacle.index}:')
                         obstacle.risk = self.calculate_risk(obstacle)
                         print(f'For ship {obstacle.index} of distance {self.calculate_distance(obstacle)},  risk = {obstacle.risk}')

            # Simulating all sensors using _simulate_sensor subroutine
            sensor_angles_ned = self._sensor_angles + self.heading
            activate_sensor = lambda i: (i % self._sensor_interval) == (self._perceive_counter % self._sensor_interval)
            sensor_sim_args = (p0_point, sensor_range, self._nearby_obstacles)
            sensor_output_arrs = list(map(
                lambda i: _simulate_sensor(self, sensor_angles_ned[i], *sensor_sim_args) if activate_sensor(i) else (

                    self._last_sensor_dist_measurements[i],
                    self._last_sensor_speed_measurements[i]
                    #self._last_sensor_situations[i]
                ),
                range(self._n_sensors)

            ))
            sensor_dist_measurements, sensor_speed_measurements = zip(*sensor_output_arrs) #, sensor_situations = zip(*sensor_output_arrs)
            #sensor_moving_measurements = np.array(sensor_moving_measurements)
            sensor_dist_measurements = np.array(sensor_dist_measurements)
            sensor_speed_measurements = np.array(sensor_speed_measurements)
            self._last_sensor_dist_measurements = sensor_dist_measurements
            self._last_sensor_speed_measurements = sensor_speed_measurements
            #self._last_sensor_moving_measurements = sensor_moving_measurements
            #self._last_sensor_situations = sensor_situations

            # Partitioning sensor readings into sectors
            sector_dist_measurements = np.split(sensor_dist_measurements, self._sector_start_indeces[1:])
            sector_speed_measurements = np.split(sensor_speed_measurements, self._sector_start_indeces[1:], axis=0)
            #sector_situations = np.split(sensor_situations, self._sector_start_indeces[1:])


            # Deciding whether there is a moving obstacle in sector
            #sector_moving_measurements = [max(sector_moving_measurements[x]) for x in range(len(sector_moving_measurements))]

            # Performing feasibility pooling
            sector_feasible_distances = np.array(list(
                map(lambda x: _feasibility_pooling(x, 2*self._width, self._d_sensor_angle), sector_dist_measurements)
            ))

            # Calculating feasible closeness
            sector_closenesses = self._get_closeness(sector_feasible_distances)

            # Retrieving obstacle speed for closest obstacle within each sector
            closest_obst_sensor_indeces = list(map(np.argmin, sector_dist_measurements))
            sector_velocities = np.concatenate(
                [sector_speed_measurements[i][closest_obst_sensor_indeces[i]] for i in range(self._n_sectors)]
            )

            # Testing if vessel has collided
            collision = any(
                float(p0_point.distance(obst.boundary)) - self._width <= 0 for obst in self._nearby_obstacles
            )

            #print(f'Ship {self.index} collided, nearby moving obstacles: {[x.index for x in self._nearby_obstacles if (not x.static and float(p0_point.distance(x.boundary)) - self._width <= 0)]}')

        self._last_sector_dist_measurements = sector_closenesses
        self._last_sector_feasible_dists = sector_feasible_distances
    #    self._last_sector_moving_measurements = sector_moving_measurements
        self._perceive_counter += 1
        self._collision = collision


        return (sector_closenesses, sector_velocities) #, sector_moving_measurements)
        #return (sector_closenesses, sector_velocities)
        #upstream/master

    def calculate_distance(self, vessel):
        return math.sqrt((vessel.x - self.x)**2 + (vessel.y - self.y)**2)

    def calculate_rel_sensor_speeds(self):
        return [math.sqrt((self._last_sensor_speed_measurements[i][0]-self.x)**2 + (self._last_sensor_speed_measurements[i][1]-self.y)**2) for i in self._n_sensors]

    def calculate_TCPA_DCPA(self, vessel):
        rel_dist = self.calculate_distance(vessel)

        #rel_vel= math.sqrt((vessel.dx - self.dx)**2 + (vessel.dy - self.dy)**2)
        #print(f'Vessel velocity: {vessel.velocity} -- Self velocity: {self.velocity}')
        #rel_vel = self.speed*math.sqrt(1 + (vessel.speed/self.speed)**2 - 2*(vessel.speed/self.speed)*np.cos(self.course-vessel.course))
        #rel_course = np.arccos((self.speed - vessel.speed*(np.cos(self.course-vessel.course)))/rel_vel)

        rel_vel = math.sqrt(np.dot(self.velocity,self.velocity) + np.dot(vessel.velocity,vessel.velocity) - 2*np.dot(self.velocity,vessel.velocity)*np.cos(vessel.course - self.course - math.pi)) # From Li and Pang (2013)

        if self.course < vessel.course:
            rel_course = self.course - np.arccos((rel_vel**2 + self.speed**2 - vessel.speed**2)/(2*rel_vel*self.speed))
        else:
            rel_course = self.course + np.arccos((rel_vel**2 + self.speed**2 - vessel.speed**2)/(2*rel_vel*self.speed))

        target_azimuth = np.arctan2(vessel.y, vessel.x)
        #self_azimuth = np.arctan2(self.y,self.x)
        theta_target = target_azimuth - self.heading
        #self_azimuth_inv = np.arctan2(self.x,self.y)
        #print('----------------')
        #print(f'Rel. dist.: {rel_dist} -- Rel. vel: {rel_vel} -- Rel. course: {rel_course*rad2deg} -- Target azi: {target_azimuth*rad2deg}')
        #print(f'Self heading: {self.heading*rad2deg} -- Self azi: {self_azimuth*rad2deg} -- Self inv. azi: {self_azimuth_inv*rad2deg}')
        #theta = np.arccos((V_rel**2 + vessel.speed**2 - self.speed**2)/(2*V_rel*vessel.speed))
        #theta_t = np.arctan2(vessel.y, vessel.x)
        #theta_r = theta_t + theta

        #DCPA = abs(vessel.y - vessel.x*np.tan(theta_r))/(math.sqrt((np.tan(theta_r))**2 + 1))
        #TCPA = np.sqrt(vessel.x**2 + vessel.y**2 - DCPA**2)/V_rel


        DCPA = rel_dist*np.sin(rel_course - target_azimuth - math.pi)
        TCPA = (rel_dist/rel_vel)*np.cos(rel_course - target_azimuth - math.pi)

        return (rel_vel, DCPA, TCPA, theta_target)

    def calculate_risk(self,vessel):
        rel_vel, DCPA, TCPA, theta_target = self.calculate_TCPA_DCPA(vessel)
        #print(f'DCPA: {DCPA} -- TCPA: {TCPA} -- theta_target: {theta_target}')
        rel_dist = self.calculate_distance(vessel)
        #a = 0.02
        #b = 0.02
        #min_dist = 8
        #return max(0,-((a*DCPA)**2 + (b*TCPA)**2)+10)#Inspirert av Imazu and Koyama
        #return np.clip((min_dist/abs(DCPA))), 0, 1)
        #raw = min_dist/abs(DCPA)
        #return 1/(1 + np.exp(-10*raw + 6))

        CRI = self.calculate_CRI(DCPA,TCPA,theta_target,rel_dist,rel_vel)
        return CRI

    def calculate_CRI(self,DCPA,TCPA,theta_target,rel_dist,rel_vel):
        alpha_DCPA = 0.38
        alpha_TCPA = 0.38
        alpha_theta = 0.12
        alpha_R = 0.12
        uTheta = self.calculate_uTheta(theta_target)
        uR = self.calculate_uRelDist(rel_dist,rel_vel,theta_target)
        uDCPA = self.calculate_uDCPA(DCPA)
        uTCPA = self.calculate_uTCPA(TCPA,DCPA,rel_vel)
        #print(f'uTheta: {uTheta}, uR: {uR}, uDCPA: {uDCPA}, uTCPA: {uTCPA}')

        return (alpha_DCPA*uDCPA)+(alpha_TCPA*uTCPA)+(alpha_theta*uTheta)+(alpha_R*uR)


    def calculate_uDCPA(self,DCPA):
        d1 = 30
        d2 = 50

        if abs(DCPA) <= d1:
            return 1
        elif abs(DCPA) > d2:
            return 0
        else:
            return ((d1-abs(DCPA))/(d2-d1))**2

    def calculate_uTCPA(self,TCPA,DCPA,rel_vel):
        d1 = 30
        d2 = 50

        if abs(DCPA) <= d1:
            t1 = math.sqrt(d1**2 - DCPA**2)/rel_vel
        else:
            t1 = 0

        if abs(DCPA) <= d2:
            t2 = math.sqrt(d2**2 - DCPA**2)/rel_vel
        else:
            t2 = 0

        if abs(TCPA) <= t1:
            return 1
        elif abs(TCPA) > t2:
            return 0
        else:
            return ((t2-abs(TCPA))/(t2-t1))**2

    def calculate_uRelDist(self,rel_dist,rel_vel,theta_target):
        length = 2.5
        D1 = 6*length
        D2 = length*8*(1.7*np.cos(theta_target-19*deg2rad) + math.sqrt(4.4+2.89*(np.cos(theta_target-19*deg2rad))))

        if abs(rel_dist) < D1:
            return 1
        elif abs(rel_dist) > D2:
            return 0
        else:
            return ((D2-abs(rel_vel))/(D2-D1))**2

    def calculate_uTheta(self,theta_target):
        u_theta = 0.5*(np.cos(theta_target-19*deg2rad) + math.sqrt((440/289) + (np.cos(theta_target-19*deg2rad))**2)) - 5/17
        return u_theta




    # Calculate safe area based on a target vessel position and velocity
    # Based on Shu et al., "Composition ship collision risk based on fuzzy theory", which is
    # based on Roman Smierzchalski, "Ships' domains as collision risk at sea in the evolutionary method of trajectory planning"
    def find_ship_domain(self,vessel):
        #print(f'Ownship speed: {self.speed}')
        #print(f'Vessel speed: {vessel.speed}')
        V_rel, d_CPA, t_CPA = self.calculate_TCPA_DCPA(vessel)
        #print(f'Vrel: {V_rel} -- dCPA: {d_CPA} -- tCPA: {t_CPA}')

        V = max(self.speed, V_rel)
        L = 2*self._width # length of ownship
        # U = error in estimating location
        Db = 10*self._width # safe distance

        d1 = d_CPA/2
        d2 = t_CPA*V # + U
        d3 = max(d_CPA, self._width*math.pow(V, 0.44)) # + U
        d4 = t_CPA*V # + U
        d5 = L*pow(V, 1.26) + 30*V # + U
        d6 = max(Db/2, 0.5)

        p1 = shapely.geometry.Point(d1,0)
        p2 = shapely.geometry.Point(0, d2)
        p3 = shapely.geometry.Point(d3, 0)
        p4 = shapely.geometry.Point(d3, d4)
        p5 = shapely.geometry.Point(0, d5)
        p6 = shapely.geometry.Point(0, -d6)
    #    print(f'Distances: d1 = {d1}, d2 = {d2}, d3 = {d3}, d4 = {d4}, d5 = {d5}, d6 = {d6},')
        pointList = [p1, p2, p3, p4, p5, p6, p1]
        #print([[p.x, p.y] for p in pointList])
        ship_domain = shapely.geometry.Polygon([[p.x, p.y] for p in pointList])
        ship_domain = shapely.affinity.rotate(ship_domain, vessel.course, use_radians=True, origin=[self.x, self.y])

        return ship_domain

    def navigate(self, path:Path) -> np.ndarray:
        """
        Calculates and returns navigation states representing the vessel's attitude
        with respect to the desired path.

        Returns
        -------
        navigation_states : np.ndarray
        """

        # Calculating path arclength at reference point, i.e. the point closest to the vessel
        vessel_arclength = path.get_closest_arclength(self.position)

        # Calculating tangential path direction at reference point
        path_direction = path.get_direction(vessel_arclength)
        cross_track_error = geom.Rzyx(0, 0, -path_direction).dot(
            np.hstack([path(vessel_arclength) - self.position, 0])
        )[1]

        # Calculating tangential path direction at look-ahead point
        target_arclength = min(path.length, vessel_arclength + self.config["look_ahead_distance"])
        look_ahead_path_direction = path.get_direction(target_arclength)
        look_ahead_heading_error = float(geom.princip(look_ahead_path_direction - self.heading))

        # Calculating vector difference between look-ahead point and vessel position
        target_vector = path(target_arclength) - self.position

        # Calculating heading error
        target_heading = np.arctan2(target_vector[1], target_vector[0])
        heading_error = float(geom.princip(target_heading - self.heading))

        # Calculating path progress
        progress = vessel_arclength/path.length
        self._progress = progress

        # Concatenating states
        self._last_navi_state_dict = {
            'surge_velocity': self.velocity[0],
            'sway_velocity': self.velocity[1],
            'yaw_rate': self.yaw_rate,
            'look_ahead_heading_error': look_ahead_heading_error,
            'heading_error': heading_error,
            'cross_track_error': cross_track_error/100,
            'target_heading': target_heading,
            'look_ahead_path_direction': look_ahead_path_direction,
            'path_direction': path_direction,
            'vessel_arclength': vessel_arclength,
            'target_arclength': target_arclength
        }
        navigation_states = np.array([self._last_navi_state_dict[state] for state in Vessel.NAVIGATION_FEATURES])

        # Deciding if vessel has reached the goal
        goal_distance = linalg.norm(path.end - self.position)

        reached_goal = goal_distance <= self.config["min_goal_distance"] or progress >= self.config["min_path_progress"]
        self._reached_goal = reached_goal

        return navigation_states

    def observe(self):
        navigation_states = self.navigate(self.path)
        sector_closenesses, sector_velocities = self.perceive(self.obstacles)
        obs = np.concatenate([navigation_states, sector_closenesses, sector_velocities]) #, sector_moving_obstacles])

        return (obs)

    def req_latest_data(self) -> dict:
        """Returns dictionary containing the most recent perception and navigation
        states."""
        return {
            'distance_measurements': self._last_sensor_dist_measurements,
            'speed_measurements': self._last_sensor_speed_measurements,
            'feasible_distances': self._last_sector_feasible_dists,
            'navigation': self._last_navi_state_dict,
            'collision' : self._collision,
            'progress': self._progress,
            'reached_goal': self._reached_goal
        }

    def update(self,agent=None):

        if not self._step_counter % 10_000 or agent is None:
            directory = 'c:/users/amalih/onedrive - ntnu/github/logs/agents/MultiAgent-v0/'
            latest_subdir = max([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime)

            try:
                latest_agent = max([os.path.join(latest_subdir,d) for d in os.listdir(latest_subdir)], key=os.path.getmtime)
                #print(f'Latest agent: {latest_agent}')
                self.agent = PPO2.load(latest_agent)
                # params = self.agent.get_parameters()
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
            except ValueError:
                pass
                #print(f'No agent used')

    def update_without_agent(self, dt=1):

        index = int(np.floor(self._step_counter))
        arclength = self.path.get_closest_arclength(self.position)
        derivatives = self.path.path_derivatives(arclength)

        # Setting velocities
        self._state[3] = derivatives[0]
        self._state[4] = derivatives[1]

        dx = 0.5*dt*self.dx
        dy = 0.5*dt*self.dy
        heading = np.arctan2(dy, dx)

        self._state[0] = self.position[0] + dx
        self._state[1] = self.position[1] + dy
        self._state[2] = heading

        #print(f'New state for ship {self.index}:{self.position}')

        self._step_counter += 1
        self._boundary = self.calculate_boundary()

    def update_with_agent(self, dt=1):

        navigation_states = self.navigate(self.path)
        sector_closenesses, sector_velocities = self.perceive(self.obstacles)

        obs = np.concatenate([navigation_states, sector_closenesses, sector_velocities])

        if self.agent != None:
            action, _states = self.agent.predict(obs, deterministic=True)
            action[0] = (action[0] + 1)/2
        else:
            action = [0,0]


        self.step(action)

    def calculate_boundary(self):
        #print(f'In CB in VESSEL {self.index}')
        ship_angle = self.heading# float(geom.princip(self.heading))

        boundary_temp = shapely.geometry.Polygon(self._points)
        boundary_temp = shapely.affinity.rotate(boundary_temp, ship_angle, use_radians=True, origin='centroid')
        boundary_temp = shapely.affinity.translate(boundary_temp, xoff=self.position[0], yoff=self.position[1])

        return boundary_temp

    @property
    def nearby_vessels(self):
        return [x for x in self._nearby_obstacles if not x.static]

    @property
    def collision(self):
        return self._collision

    @property
    def position(self):
        """
        Returns an array holding the position of the AUV in cartesian
        coordinates.
        """
        return self._state[0:2]

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    @property
    def init_position(self):
        """
        Returns an array holding the path of the AUV in cartesian
        coordinates.
        """
        return self._prev_states[-1, 0:2]

    @property
    def heading_history(self):
        """
        Returns the heading of the AUV wrt true north.
        """
        return self._prev_states[:, 2]

    @property
    def heading_change(self):
        """
        Returns the change of heading of the AUV wrt true north.
        """
        return geom.princip(self._prev_states[-1, 2] - self._prev_states[-2, 2]) if len(self._prev_states) >= 2 else self.heading

    @property
    def dx(self):
        return self._state[3]

    @property
    def dy(self):
        return self._state[4]


    @property
    def width(self) -> float:
        """Width of vessel in meters."""
        return self._width

    @property
    def position(self) -> np.ndarray:
        """Returns an array holding the position of the AUV in cartesian
        coordinates."""
        return self._state[0:2]

    @property
    def path_taken(self) -> np.ndarray:
        """Returns an array holding the path of the AUV in cartesian
        coordinates."""
        return self._prev_states[:, 0:2]

    @property
    def heading(self) -> float:
        """Returns the heading of the AUV with respect to true north."""
        return self._state[2]

    @property
    def heading_taken(self) -> np.ndarray:
        """Returns an array holding the heading of the AUV for all timesteps."""
        return self._prev_states[:, 2]

    @property
    def velocity(self) -> np.ndarray:
        """Returns the surge and sway velocity of the AUV."""
        return self._state[3:5]

    @property
    def speed(self) -> float:
        """Returns the speed of the AUV."""
        return linalg.norm(self.velocity)

    @property
    def yaw_rate(self) -> float:
        """Returns the rate of rotation about the z-axis."""
        return self._state[5]

    @property
    def max_speed(self) -> float:
        """Returns the maximum speed of the AUV."""
        return const.MAX_SPEED

    @property
    def boundary(self):
        """Returns the boundary of the AUV."""
        return self._boundary

    @property
    def init_boundary(self) -> shapely.geometry.Polygon:
        """shapely.geometry.Polygon object used for simulating the
        sensors' detection of the obstacle instance."""
        return self._init_boundary

    @property
    def course(self) -> float:
        """Returns the course angle of the AUV with respect to true north."""
        crab_angle = np.arctan2(self.velocity[1], self.velocity[0])
        return self.heading + crab_angle

    @property
    def sensor_angles(self) -> np.ndarray:
        """Array containg the angles each sensor ray relative to the vessel heading."""
        return self._sensor_angles

    @property
    def sector_angles(self) -> np.ndarray:
        """Array containg the angles of the center line of each sensor sector relative to the vessel heading."""
        return self._sector_angles
    @property
    def last_sensor_situations(self) -> np.ndarray:
        return self._last_sensor_situations

    #@property
    #def sector_moving_measurements(self):
    #    return self._last_sector_moving_measurements

    @property
    def n_sensors(self):
        return self._n_sensors
