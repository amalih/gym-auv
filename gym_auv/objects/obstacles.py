import numpy as np
import shapely.geometry
import shapely.affinity
import gym_auv.utils.geomutils as geom

class Obstacle():
    def __init__(self):
        self.last_distance = 0
        self.virtual_boundary = []
        self.boundary = None
        self.static = None

    @property
    def valid(self):
        return self.boundary.is_valid

    def update(self, dt):
        pass

    def _calculate_boundary(self):
        pass

class CircularObstacle(Obstacle):
    def __init__(self, position, radius, color=(0.6, 0, 0)):
        super().__init__()
        self.color = color
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        if radius < 0:
            raise ValueError
        self.static = True
        self.radius = radius
        self.position = position.flatten()
        self._calculate_boundary()

    def _calculate_boundary(self):
        self.boundary = shapely.geometry.Point(*self.position).buffer(self.radius).boundary.simplify(0.3, preserve_topology=False)

class PolygonObstacle(Obstacle):
    def __init__(self, points, color=(0.6, 0, 0)):
        super().__init__()
        self.static = True
        self.color = color
        self.points = points
        self.boundary = shapely.geometry.Polygon(points)

class VesselObstacle(Obstacle):
    def __init__(self, width, trajectory, name=''):
        super().__init__()
        self.static = False
        self.width = width
        self.trajectory = trajectory
        self.trajectory_velocities = []
        self.name = name
        i = 0
        while i < len(trajectory)-1:
            cur_t = trajectory[i][0]
            next_t = trajectory[i+1][0]
            cur_waypoint = trajectory[i][1]
            next_waypoint = trajectory[i+1][1]

            dx = (next_waypoint[0] - cur_waypoint[0])/(next_t - cur_t)
            dy = (next_waypoint[1] - cur_waypoint[1])/(next_t - cur_t)

            for _ in range(cur_t, next_t):
                self.trajectory_velocities.append((dx, dy))

            i+= 1

        self.waypoint_counter = 0
        self.points = [
            (-self.width/2, -self.width/2),
            (-self.width/2, self.width/2),
            (self.width/2, self.width/2),
            (3/2*self.width, 0),
            (self.width/2, -self.width/2),
        ]
        self.position = np.array(self.trajectory[0][1])
        self.heading = np.pi/2

        self.update(dt=0.1)

    def update(self, dt):
        self.waypoint_counter += dt

        index = int(np.floor(self.waypoint_counter))

        if index >= len(self.trajectory_velocities) - 1:
            self.waypoint_counter = 0
            index = 0
            self.position = np.array(self.trajectory[0][1])

        dx = self.trajectory_velocities[index][0]
        dy = self.trajectory_velocities[index][1]

        self.dx = dt*dx
        self.dy = dt*dy
        self.heading = np.arctan2(self.dy, self.dx)
        self.position = self.position + np.array([self.dx, self.dy])

        #print('OBS', self.position, self.heading)

        self._calculate_boundary()

    def _calculate_boundary(self):
        ship_angle = self.heading# float(geom.princip(self.heading))

        boundary_temp = shapely.geometry.Polygon(self.points)
        boundary_temp = shapely.affinity.rotate(boundary_temp, ship_angle, use_radians=True, origin='centroid')
        boundary_temp = shapely.affinity.translate(boundary_temp, xoff=self.position[0], yoff=self.position[1])

        self.boundary = boundary_temp
