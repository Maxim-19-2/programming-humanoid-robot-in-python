'''In this exercise you need to implement inverse kinematics for NAO's legs

* Tasks:
    1. solve inverse kinematics for NAO's legs by using analytical or numerical method.
       You may need documentation of NAO's leg:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    2. use the results of inverse kinematics to control NAO's legs (in InverseKinematicsAgent.set_transforms)
       and test your inverse kinematics implementation.
'''


from math import atan2
from forward_kinematics import ForwardKinematicsAgent
import numpy as np
from time import time
from numpy.matlib import identity
import random

def from_Trans(T):
    x = T[3, 0]
    y = T[3, 1] 
    z = T[3, 2]
    theta = 0
        
    if T[0, 0] == 1:
        theta = atan2(T[2, 1], T[1, 1])
    elif T[1, 1] == 1:
        theta = atan2(T[0, 2], T[0, 0])
    elif T[2, 2] == 1:
        theta = atan2(T[1, 0], T[0, 0])

    return np.array([x, y, z, theta])

class InverseKinematicsAgent(ForwardKinematicsAgent):
    def inverse_kinematics(self, effector_name, transform):
        '''solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        '''
        joint_angles = {}
        # YOUR CODE HERE
        joints = self.chains[effector_name]
        N = len(joints)-1
        theta = (np.random.random_sample(N) * 1e-5)
        lambda_ = 1
        max_step = 0.1 #variables from notebook

        for name in self.chains[effector_name]:
            joint_angles[name] = self.perception.joint[name]

        for i in range(1000):
            self.forward_kinematics(self.perception.joint)

            Ts = list()
            for joint in joints:
                Ts.append(self.transforms[joint])

            theta_e = 0
            if Ts[-1][0,0] == 1:
                theta_e = atan2(Ts[-1][2,1], Ts[-1][1,1])
            elif Ts[-1][1,1] == 1:
                theta_e = atan2(Ts[-1][0,2], Ts[-1][0,0])
            elif Ts[-1][2,2] == 1:
                theta_e = atan2(Ts[-1][0,1], Ts[-1][0,0])
            
            Te = np.matrix([(Ts[-1][3,0]), (Ts[-1][3,1]) , (Ts[-1][3,2]) , theta_e]).T
            
            target=np.matrix([(transform[3,0]),(transform[3,1]),(transform[3,2]),theta_e])
            
            #from notebook
            e = target - Te
            e[e > max_step] = max_step
            e[e < -max_step] = -max_step
            T = np.matrix([from_Trans(i) for i in Ts[1:-1]])
            J = Te - T
            dT = Te - T
            J[0, :] = -dT[1, :] # x
            J[1, :] = dT[0, :] # y
            J[-1, :] = 1  # angular
            d_theta = lambda_ * np.linalg.pinv(J) * e
            theta = np.asarray(d_theta.T)[0]

            if  np.linalg.norm(d_theta) < 1e-4:
                break

        return joint_angles

    def set_transforms(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        # YOUR CODE HERE
        joint_angles = self.inverse_kinematics(effector_name, transform)
        names = self.chains[effector_name]
        times = [[0, 3]] * len(names)
        keys = []

        for i in range (len(names)):
            name = names[i]
            keys.append([[self.perception.joint[name], [3, 0., 0.]], [joint_angles[name], [3, 0., 0.]]])
        
        self.keyframes = (names, times, keys)  # the result joint angles have to fill in

if __name__ == '__main__':
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    T = identity(4)
    T[-1, 1] = 0.05
    T[-1, 2] = -0.26
    agent.set_transforms('LLeg', T)
    agent.run()
