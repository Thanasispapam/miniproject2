import numpy as np

class Kalmanfilter:

    def __init__(self, dt, spoint):
        # time step
        self.dt = 1

        # initial State of X
        self.Xpre = self.spoint

        # State transition matrix
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        # random Position error in m
        self.Pxerr = 2
        self.Pyerr = 2

        # random Velocity error in m/s
        self.Vxerr = 1
        self.Vyerr = 1

        #State covariance matrix
        self.P = np.matrix([[self.Pxerr ** 2, 0, 0, 0],
                            [0, self.Pyerr ** 2, 0, 0],
                            [0, 0, self.Vxerr ** 2, 0],
                            [0, 0, 0, self.Vyerr ** 2]])

        # Process noise covriance matrix and measurement error and measurement covariance matrix
        Processnoise = 0.1
        Measerror = 0.2

        self.Q = np.eye(4) * Processnoise ** 2 * self.dt
        self.R = np.eye(2) * Measerror ** 2 / self.dt


    def predict(self):
        self.Xp = self.A * self.Xpre
        self.Pp = self.A * self.P * np.transpose(self.A) + self.Q

        return self.Xp, self.Pp

    def update(self):
        #self.Y = self.C * self.Xmeas
        self.K = (self.Pp * self.H) / (self.H * self.Pp * np.transpose(self.H) + self.R)
        self.Xf = self.Xp + self.K * (self.Y - self.H * self.Xp)
        self.Pf = (self.H - self.K * self.H) * self.Pp

        return self.Xf, self.Pf

    def getstate(self):
        return self.Xf
