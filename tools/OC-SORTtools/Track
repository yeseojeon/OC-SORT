import KF

class Track:
    def __init__(self,past_observation):
        self.past_observation = past_observation
        self.has_filter = False
        self.tracked = True
        self.untracked = 0
        ##TODO: get values for matrices    
        self.p = []
        self.q = []
        self.r = []
        self.a = []

    def init_filter(self,observation):
        self.kalman_filter = KF(self.past_observation, self.p, self.q, self.r, self.a, observation)
        self.has_filter = True

    def check_filter(self):
        return self.has_filter

    def get_prediction(self):
        return self.kalman_filter.get_prediction()
    
    def get_past(self):
        return self.past_observation
    