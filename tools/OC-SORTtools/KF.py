import numpy as np

class KF:
    def __init__(self, x, p, q, r, a, h):
        self.x = x # initial (first) state
        self.P = p  # uncertainty covariance
        self.Q = q  # process noise covariance
        self.R = r  # measurement noise covariance
        self.A = a  # state transition matrix
        self.H = h  # measurement
    
    def predict(self):
        xk_p = np.dot(self.A, self.x1)
        pk_p = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        self.pk_p = pk_p
        self.xk_p = xk_p
        
    def kalman_gain(self):
        Kk = np.dot(np.dot(self.pk_p, self.H.T), np.linalg.inv(np.dot(np.dot(self.H, self.pk_p), self.H.T) + self.R))
        return Kk
    
    def estimate(self, zk):
        xk = self.xk_p + np.dot(self.kalman_gain(), (zk - np.dot(self.H, self.xk_p)))
        pk = self.pk_p - np.dot(np.dot(self.kalman_gain(), self.H), self.pk_p)
        return xk, pk
    
    def update(self):
        self.xk_p = self.estimate()[0]
        self.pk_p = self.estimate()[1]
        
if __name__ == "__main__":
    def main():
        # Example usage of the KF class
        position1 = np.array([[0], [0], [0], [0]])  # initial state
        position2 = np.array([[0], [0], [0], [0]])  # second state
        time = 1
        #TODO: calculate x  based on position1 and position2
        velocity = (position2 - position1) / time

    main()  
        