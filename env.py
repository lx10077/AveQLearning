import numpy as np

class RandomMDP(object):
    def __init__(self, S, A, r, sparsity = 1.0):
        self.S = S
        self.A = A
        np.random.seed(3154)
        sparse = np.random.binomial(1, sparsity, size=(A*S, S))
        for i in range(A*S):
            if sum(sparse[i, :]) == 0:
                sparse[i, np.random.randint(S)] = 1
        P = np.random.uniform(0, r, size=(A*S, S))
        self.P = P / np.sum(P, 1)[:, np.newaxis]
        self.R = np.random.uniform(0, 1, size=(A*S,))

    def generator(self, T):
        trans = np.zeros(shape=(T, self.S*self.A, self.S))
        for sa in range(self.S*self.A):
            s_next = np.random.choice(self.S, T, p=self.P[sa])
            trans[[_ for _ in range(T)], sa, s_next] = 1.0
        return trans

class SimpleMDP(object):
    def __init__(self, p):
        S = 5
        A = 2
        self.R = np.zeros(shape=(S,A))
        self.P = np.zeros(shape=(S,A,S))
        self.R[1,:] = 1
        self.R[2,:] = 1
        self.P[0,0,1] = 1
        self.P[0,1,2] = 1
        self.P[1,0,1] = p
        self.P[1,0,3] = 1-p
        self.P[1,1,1] = p 
        self.P[1,1,3] = 1-p
        self.P[2,0,2] = p
        self.P[2,0,4] = 1-p
        self.P[2,1,2] = p
        self.P[2,1,4] = 1-p
        self.P[3,:,3] = 1
        self.P[4,:,4] = 1
        self.R = self.R.reshape((S*A, ))
        self.P = self.P.reshape((S*A, S))
        self.S = S
        self.A = A
    
    def generator(self, T):
        trans = np.zeros(shape=(T, self.S*self.A, self.S))
        for sa in range(self.S*self.A):
            s_next = np.random.choice(self.S, T, p=self.P[sa])
            trans[[_ for _ in range(T)], sa, s_next] = 1.0
        return trans
