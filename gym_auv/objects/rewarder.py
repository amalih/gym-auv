import numpy as np
from abc import ABC, abstractmethod
from gym_auv.objects.vessel import Vessel
import math

deg2rad = math.pi/180

def _sample_lambda(scale):
    log = -np.random.gamma(1, scale)
    y = np.power(10, log)
    return y

def _sample_eta():
    y = np.random.gamma(shape=1.9, scale=0.6)
    return y

class BaseRewarder(ABC):
    def __init__(self, vessel) -> None:
        self.vessel = vessel
        self.params = {}

    #@property
    #def vessel(self) -> self.vessel:
    #    """self.vessel instance that the reward is calculated with respect to."""
    #    return self.vessel

    @abstractmethod
    def calculate(self) -> float:
        """
        Calculates the step reward and decides whether the episode
        should be ended.

        Returns
        -------
        reward : float
            The reward for performing action at this timestep.
        """

    def insight(self) -> np.ndarray:
        """
        Returns a numpy array with reward parameters for the agent
        to have an insight into its reward function

        Returns
        -------
        insight : np.array
            The reward insight array at this timestep.
        """
        return np.array([])
#
# class ColavRewarder(BaseRewarder):
#     def __init__(self, self.vessel):
#         super().__init__(self.vessel)
#         self.params['gamma_theta'] = 10.0
#         self.params['gamma_x'] = 0.1
#         self.params['gamma_y_speed'] = 1.0
#         self.params['gamma_y_e'] = 5.0
#         self.params['penalty_yawrate'] = 0.0
#         self.params['penalty_torque_change'] = 0.0
#         self.params['cruise_speed'] = 0.1
#         self.params['neutral_speed'] = 0.05
#         self.params['negative_multiplier'] = 2.0
#         self.params['collision'] = -10000.0
#         self.params['lambda'] = _sample_lambda(scale=0.2)
#         self.params['eta'] = 0#_sample_eta()
#
#     N_INSIGHTS = 1
#     def insight(self):
#         return np.array ([np.log10(self.params['lambda'])])
#
#     def calculate(self):
#         latest_data = self._vessel.req_latest_data()
#         nav_states = latest_data['navigation']
#         measured_distances = latest_data['distance_measurements']
#         measured_speeds = latest_data['speed_measurements']
#         collision = latest_data['collision']
#
#         if collision:
#             reward = self.params["collision"]*(1-self.params["lambda"])
#             return reward
#
#         reward = 0
#
#         # Extracting navigation states
#         cross_track_error = nav_states['cross_track_error']
#         heading_error = nav_states['heading_error']
#
#         # Calculating path following reward component
#         cross_track_performance = np.exp(-self.params['gamma_y_e']*np.abs(cross_track_error))
#         path_reward = (1 + np.cos(heading_error)*self._vessel.speed/self._vessel.max_speed)*(1 + cross_track_performance) - 1
#
#         # Calculating obstacle avoidance reward component
#         closeness_penalty_num = 0
#         closeness_penalty_den = 0
#         if self._vessel.n_sensors > 0:
#             for isensor in range(self._vessel.n_sensors):
#                 angle = self._vessel.sensor_angles[isensor]
#                 x = measured_distances[isensor]
#                 speed_vec = measured_speeds[isensor]
#                 weight = 1 / (1 + np.abs(self.params['gamma_theta']*angle))
#                 raw_penalty = self._vessel.config["sensor_range"]*np.exp(-self.params['gamma_x']*x +self.params['gamma_y_speed']*max(0, speed_vec[1]))
#                 weighted_penalty = weight*raw_penalty
#                 closeness_penalty_num += weighted_penalty
#                 closeness_penalty_den += weight
#
#             closeness_reward = -closeness_penalty_num/closeness_penalty_den
#         else:
#             closeness_reward = 0
#
#         # Calculating living penalty
#         living_penalty = self.params['lambda']*(2*self.params["neutral_speed"]+1) + self.params["eta"]*self.params["neutral_speed"]
#
#         # Calculating total reward
#         reward = self.params['lambda']*path_reward + \
#             (1-self.params['lambda'])*closeness_reward - \
#             living_penalty + \
#             self.params["eta"]*self._vessel.speed/self._vessel.max_speed - \
#             self.params["penalty_yawrate"]*abs(self._vessel.yaw_rate)
#
#         if reward < 0:
#             reward *= self.params['negative_multiplier']
#
#         return reward

class ColavRewarder(BaseRewarder):
    def __init__(self, vessel):
        super().__init__(vessel)
        self.params['gamma_theta'] = 10.0
        self.params['gamma_x_stat'] = 0.1
        self.params['gamma_x_starboard'] = 0.06
        self.params['gamma_x_port'] = 0.08
        self.params['gamma_v_y'] = 2.0
        self.params['gamma_y_e'] = 5.0
        self.params['penalty_yawrate'] = 0.0
        self.params['penalty_torque_change'] = 0.01
        self.params['cruise_speed'] = 0.1
        self.params['neutral_speed'] = 0.1
        self.params['negative_multiplier'] = 2.0
        self.params['collision'] = -10000.0
        self.params['lambda'] = 1#_sample_lambda(scale=0.2)
        self.params['eta'] = 0.1#_sample_eta()
        self.params['alpha_lambda'] = 3.5
        self.params['gamma_min_x'] = 0.04
        self.params['gamma_weight'] = 1

        self.vessel = vessel


    N_INSIGHTS = 1
    def insight(self):
        return np.array ([self.params['lambda']])

    def calculate(self):
        #print('Requesting from rewarder, self.vessel: ', self.vessel.id)
        latest_data = self.vessel.req_latest_data()
        nav_states = latest_data['navigation']
        measured_distances = latest_data['distance_measurements']
        measured_speeds = latest_data['speed_measurements']
        collision = latest_data['collision']


        #print([x[1] for x in measured_speeds])

        if collision:
            reward = self.params['collision']#*(1-self.params['lambda'])
            return reward

        reward = 0

        # Extracting navigation states
        cross_track_error = nav_states['cross_track_error']
        heading_error = nav_states['heading_error']

        # Calculating path following reward component
        cross_track_performance = np.exp(-self.params['gamma_y_e']*np.abs(cross_track_error))
        path_reward = (1 + np.cos(heading_error)*self.vessel.speed/self.vessel.max_speed)*(1 + cross_track_performance) - 1

        # Calculating obstacle avoidance reward component
        closeness_penalty_num = 0
        closeness_penalty_den = 0
        static_closeness_penalty_num = 0
        static_closeness_penalty_den = 0
        closeness_reward = 0
        static_closeness_reward = 0
        moving_distances = []
        lambdas = []

        #print(f'Min distance for vessel {self.vessel.index}: {np.amin(measured_distances)}')

        if self.vessel.n_sensors > 0:
            for isensor in range(self.vessel.n_sensors):
                angle = self.vessel.sensor_angles[isensor]
                x = measured_distances[isensor]
                speed_vec = measured_speeds[isensor]


                if speed_vec.any():
                    #print(speed_vec[1])
                    if speed_vec[1] >= 0:
                        sensor_lambda = 1/(1+np.exp(-0.03*x+4))
                    #self.params['lambda'] = 1/(1+np.exp(-0.10*x+10)) +0.1
                    if speed_vec[1] < 0:
                        sensor_lambda = 1/(1+np.exp(-0.05*x+2))

                    #if speed_vec[1] > 0:
                        #self.params['lambda'] = 1/(1+np.exp(-0.04*x+4))
                    #if speed_vec[1] < 0:
                        #self.params['lambda'] = 1/(1+np.exp(-0.08*x+3))
                    lambdas.append(sensor_lambda)

                    weight = 2 / (1 + np.exp(self.params['gamma_weight']*np.abs(angle)))
                    moving_distances.append(x)
                    #logistic_vy = 1/(1+np.exp(-5*speed_vec[1]))
                    #incoming_angle = (np.arctan2(speed_vec[0], speed_vec[1]))*rad2deg
                    #print(f'Incoming angle: {incoming_angle} -- Vy: {speed_vec[1]}')
                    #v_lim = 0.2

                    if angle < 0*deg2rad and angle > -112.5*deg2rad: #straffer høyre side
                        if speed_vec[1] > 0:
                            speed_weight = 0.02
                        else:
                            speed_weight = 0.3
                    #    if abs(incoming_angle) <= 30:
                    #        speed_weight = 0.03
                    #    elif abs(incoming_angle) > 30 and abs(incoming_angle) <= 90:
                    #        speed_weight = 0.01
                    #    else:
                    #        speed_weight = 0.5
                    #    if speed_vec[1] > v_lim:
                    #        speed_weight = 0.04
                    #    elif speed_vec[1] <= v_lim and speed_vec[1] >= -v_lim:
                    #        speed_weight = 0.02
                    #    elif speed_vec[1] < -v_lim:
                    #        speed_weight = 0.4
                        #raw_penalty = 150*np.exp(-self.params['gamma_x_starboard']*x + speed_weight*speed_vec[1])
                        raw_penalty = 75*np.exp((-self.params['gamma_x_starboard']+speed_weight*speed_vec[1])*x)
                        #print('ANGLE: ',angle, ' -- Ship on right side, y speed:', speed_vec[1])
                    if angle > 0*deg2rad and angle < 112.5*deg2rad:
                        if speed_vec[1] > 0:
                            speed_weight = 0.04
                        else:
                            speed_weight = 0.03


                    #    if abs(incoming_angle) <= 30:
                    #        speed_weight = 0.07
                    #    elif abs(incoming_angle) > 30 and abs(incoming_angle) <= 90:
                    #        speed_weight = 0.07
                    #    else:
                    #        speed_weight = 0.1
                    #    if speed_vec[1] > v_lim:
                    #        speed_weight = 0.07
                    #    elif speed_vec[1] <= v_lim and speed_vec[1] >= -v_lim:
                    #        speed_weight = 0.03
                    #    elif speed_vec[1] < -v_lim:
                    #        speed_weight = 0.05
                        raw_penalty = 75*np.exp((-self.params['gamma_x_port']+speed_weight*speed_vec[1])*x)
                        #print('ANGLE: ',angle, ' -- Ship on left side, y speed:', speed_vec[1])
                        #raw_penalty = 150*np.exp(-self.params['gamma_x_port']*x + speed_weight*speed_vec[1])
                    else:
                        if speed_vec[1] > 0:
                            speed_weight = 0.04
                        else:
                            speed_weight = 0.03
                    #    if abs(incoming_angle) <= 30:
                    #        speed_weight = 0.07
                    #    elif abs(incoming_angle) > 30 and abs(incoming_angle) <= 90:
                    #        speed_weight = 0.07
                    #    else:
                    #        speed_weight = 0.1
                    #    if speed_vec[1] > v_lim:
                    #        speed_weight = 0.07
                    #    elif speed_vec[1] <= v_lim and speed_vec[1] >= -v_lim:
                    #        speed_weight = 0.035
                    #    elif speed_vec[1] < -v_lim:
                    #        speed_weight = 0.05
                        raw_penalty = 75*np.exp((-self.params['gamma_x_stat']+speed_weight*speed_vec[1])*x)
                        #print('ANGLE: ',angle, ' -- Ship behind, y speed:', speed_vec[1])

                    weighted_penalty = (1-sensor_lambda)*weight*raw_penalty
                    closeness_penalty_num += weighted_penalty
                    closeness_penalty_den += weight

                else:
                    self.params['lambda'] = 1
                    weight = 1 / (1 + np.abs(self.params['gamma_theta']*angle))
                    raw_penalty = 75*np.exp(-self.params['gamma_x_stat']*x)
                    weighted_penalty = weight*raw_penalty
                    static_closeness_penalty_num += weighted_penalty
                    static_closeness_penalty_den += weight



            if closeness_penalty_num:
                closeness_reward = -closeness_penalty_num/closeness_penalty_den


            if static_closeness_penalty_num:
                static_closeness_reward = -static_closeness_penalty_num/static_closeness_penalty_den


        #if len(moving_distances) != 0:
        #    min_dist = np.amin(moving_distances)
        #    self.params['lambda'] = 1/(1+np.exp(-0.04*min_dist+4))
            #self.params['lambda'] = 1/(1+np.exp(-(self.params['gamma_min_x']*min_dist-self.params['alpha_lambda'])))
        #else:
        #    self.params['lambda'] = 1

        if len(lambdas):
            self.params['lambda'] = np.amin(lambdas)
        else:
            self.params['lambda'] = 1

        #if path_reward > 0:
        #    path_reward = path_lambda*path_reward
        # Calculating living penalty
        living_penalty = 1#.2*(self.params['neutral_speed']+1) + self.params['eta']*self.params['neutral_speed']

        # Calculating total reward #min((1-path_lambda)*path_reward,closeness_reward) -
        reward = self.params['lambda']*path_reward + \
            static_closeness_reward + \
            min((1-self.params['lambda'])*path_reward,closeness_reward) - \
            living_penalty
            #self.params['eta']*self.vessel.speed/self.vessel.max_speed# - \
            #self.params['penalty_yawrate']*abs(self.vessel.yaw_rate)

        if reward < 0:
            reward *= self.params['negative_multiplier']
        #print(reward)
        insight = self.insight()
        
        return reward, insight



class MultiRewarder(BaseRewarder):
    def __init__(self, vessel):
        super().__init__(vessel)
        self.params['gamma_theta'] = 10.0
        self.params['gamma_x_stat'] = 0.1
        self.params['gamma_x_starboard'] = 0.06
        self.params['gamma_x_port'] = 0.08
        self.params['gamma_v_y'] = 2.0
        self.params['gamma_y_e'] = 5.0
        self.params['penalty_yawrate'] = 0.0
        self.params['penalty_torque_change'] = 0.01
        self.params['cruise_speed'] = 0.1
        self.params['neutral_speed'] = 0.1
        self.params['negative_multiplier'] = 2.0
        self.params['collision'] = -10000.0
        self.params['lambda'] = 1#_sample_lambda(scale=0.2)
        self.params['eta'] = 0.1#_sample_eta()
        self.params['alpha_lambda'] = 3.5
        self.params['gamma_min_x'] = 0.04
        self.params['gamma_weight'] = 1

        self.vessel = vessel

    N_INSIGHTS = 1
    def insight(self):
        return np.array ([1])

    def calculate(self,vessel):
        self.vessel = vessel
        #print(self._vessel.last_sensor_situations)
        latest_data = self.vessel.req_latest_data()
        nav_states = latest_data['navigation']
        measured_distances = latest_data['distance_measurements']
        measured_speeds = latest_data['speed_measurements']
        collision = latest_data['collision']

        if collision:
            reward = self.params["collision"]
            return reward

        reward = 0
        colav_penalty = 0
        #print('Number of nearby vessels for vessel ', {self.vessel.index} ,' : ', len(self.vessel.nearby_vessels))
        for ship in self.vessel.nearby_vessels:
            rel_bearing = np.arctan2(ship.y-self.vessel.y, ship.x-self.vessel.x)
            if rel_bearing <= 5*deg2rad and rel_bearing > -112.5*deg2rad:
                weight = 2
            else:
                weight = 1
            #print('Risk: ', ship.risk, ' --- Colav penalty:', -weight*ship.risk)
            if ship.risk:
                colav_penalty += -weight*ship.risk


        static_closeness_penalty_num = 0
        static_closeness_penalty_den = 0
        static_closeness_reward = 0

        #print(f'Min distance for vessel {self.vessel.index}: {np.amin(measured_distances)}')

        if self.vessel.n_sensors > 0:
            for isensor in range(self.vessel.n_sensors):
                angle = self.vessel.sensor_angles[isensor]
                x = measured_distances[isensor]
                speed_vec = measured_speeds[isensor]


                if not speed_vec.any():
                    weight = 1 / (1 + np.abs(self.params['gamma_theta']*angle))
                    raw_penalty = 75*np.exp(-self.params['gamma_x_stat']*x)
                    weighted_penalty = weight*raw_penalty
                    static_closeness_penalty_num += weighted_penalty
                    static_closeness_penalty_den += weight


        if static_closeness_penalty_num:
            static_closeness_reward = -static_closeness_penalty_num/static_closeness_penalty_den

        # Extracting navigation states
        cross_track_error = nav_states['cross_track_error']
        heading_error = nav_states['heading_error']


        # Calculating path following reward component
        cross_track_performance = np.exp(-self.params['gamma_y_e']*np.abs(cross_track_error))
        path_reward = (1 + np.cos(heading_error)*vessel.speed/vessel.max_speed)*(1 + cross_track_performance) - 1


        living_penalty = 1

        # Calculating total reward #min((1-path_lambda)*path_reward,closeness_reward) -
        reward = path_reward + \
            static_closeness_reward + \
            colav_penalty - \
            living_penalty

        #print('Path reward: ', path_reward, ' --- Static reward: ', static_closeness_reward, ' --- Colav penalty:', colav_penalty, ' --- Living penalty: ', living_penalty)

        #if reward < 0:
        #    reward *= self.params['negative_multiplier']
    #    print('Reward: ', reward)
        return reward
