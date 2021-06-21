#Imports
import numpy as np
from mesa.agent import Agent

class AMSAgent(Agent):
    def __init__(self, unique_id, model, initial_state):
        """Initialises an agent in the Active Matter Synchronisation model
        
        Args :
            Initial state includes : 
                Ressources x and y, 
                Preference of the agent : 0 or 1,
                Radius of the circle of influence : R, fixed once initialised
                Position of the agent : needs to be a tuple
                Other constants for computations of rho and S
        """
        super().__init__(unique_id=unique_id, model=model)            
        self.x = initial_state['x']
        self.y = initial_state['y']
        self._next_x = None
        self._next_y = None
        
        self.preference = initial_state['preference']
        self.radius = initial_state['radius']
        self.A = 4 * np.pi * self.radius**2
        self.position = ()
        
        self.rho = self.x / self.A #needed for computing of overlapping
        self.transition_probability = None
        self.fluctuation = initial_state['fluctuation']
        self.neighbours = []
        self.overlap = []
        
         # CONSTANTS       
        self.z = initial_state['z']
        self.b = initial_state['b'] # b is a 2x2 matrix
        self.beta = initial_state['beta']        
        self.rho_0 = initial_state['rho_0']
        self.rho_1 = initial_state['rho_1']
        self.dt = initial_state['dt']
        self.p = 0.9 # probability to try to change state after characteristic time T

        self.Q = initial_state['Q']
        self.S_factor = initial_state['S_factor']
        self.a = initial_state['a']
        self.gamma_e = initial_state['gamma_e']
        self.y_thr = initial_state['y_thr']
        
        
    # Setter for radius and ajusts A
    def set_radius(self, r):
        self.radius = r
        self.A = 4 * np.pi * r**2
    
    def update_transition_probability(self):
        if self.preference == 0:
            self.transition_probability = np.exp(self.beta * (self.rho-self.rho_1)) / \
            (np.exp(self.beta * (self.rho-self.rho_1)) + np.exp(-self.beta * (self.rho-self.rho_0)))
        else:
            self.transition_probability = np.exp(-self.beta * (self.rho-self.rho_0)) / \
            (np.exp(self.beta * (self.rho-self.rho_1)) + np.exp(-self.beta * (self.rho-self.rho_0)))
                
    def compute_S(self):
        return self.S_factor / (1 + np.exp(2 * self.beta * (self.y - self.y_thr)))
    
    
    def update_state(self):
        self.update_transition_probability()
        if self.random.random() < self.transition_probability:
            self.preference = 1 - self.preference
                
    def rk4(self, f, y):
        k1 = self.dt * f(y)
        k2 = self.dt * f(y + k1 / 2)
        k3 = self.dt * f(y + k2 / 2)
        k4 = self.dt * f(y + k3)
        return k1, k2, k3, k4
    
    def dx_dt(self, x):
        # Setting z according to the preference of the agent
        if self.preference == 0:
            new_z = self.z
        elif self.preference == 1:
            new_z = 1 - self.z
        else:
            raise Exception('Preference needs to be 0 or 1')
            
        if self.fluctuation == True:
            dxdt = self.Q * new_z * x \
                     - self.b[0][0] * x ** 2 \
                     - self.b[0][1] * x * self.y \
                     + self.a * self.compute_S() * new_z * x * self.random.gauss(0,1) / self.gamma_e
        else:
             dxdt = self.Q * new_z * x \
                     - self.b[0][0] * x ** 2 \
                     - self.b[0][1] * x * self.y 
        return dxdt
    
    def dy_dt(self, y):
        # Setting z according to the preference of the agent
        if self.preference == 0:
            new_z = self.z
        elif self.preference == 1:
            new_z = 1 - self.z
        else:
            raise Exception('Preference needs to be 0 or 1')
            
        if self.fluctuation == True:        
            dydt = self.Q * (1 - new_z) * y \
                - self.b[1][1] * y ** 2 \
                - self.b[1][0] * self.x * y \
                + self.a * self.compute_S() *(1 - new_z) * y * self.random.gauss(0,1) / self.gamma_e
        else:
            dydt = self.Q * (1 - new_z) * y \
                - self.b[1][1] * y ** 2 \
                - self.b[1][0] * self.x * y 
        return dydt
    
    def step(self):
        self._next_x = self.x
        self._next_y = self.y
        """
        # Setting z according to the preference of the agent
        if self.preference == 0:
            new_z = self.z
        elif self.preference == 1:
            new_z = 1 - self.z
        else:
            raise Exception('Preference needs to be 0 or 1')
  
        # Computation of x and y using Euler's method 
        if self.fluctuation == True:
            dx = self.Q * new_z * self._next_x \
                 - self.b[0][0] * self._next_x ** 2 \
                 - self.b[0][1] * self._next_x * self._next_y \
                 + self.a * self.compute_S() * self.z * self.x * self.random.gauss(0,1) / self.gamma_e
            dy = self.Q * (1 - new_z) * self._next_y \
                 - self.b[1][1] * self._next_y ** 2 \
                 - self.b[1][0] * self._next_x * self._next_y \
                 + self.a * self.compute_S() *(1 - self.z) * self.y * self.random.gauss(0,1) / self.gamma_e
        else:
            dx = self.Q * new_z * self._next_x \
                 - self.b[0][0] * self._next_x ** 2 \
                 - self.b[0][1] * self._next_x * self._next_y 
            dy = self.Q * (1 - new_z) * self._next_y \
                 - self.b[1][1] * self._next_y ** 2 \
                 - self.b[1][0] * self._next_x * self._next_y 

        self._next_x += dx * self.dt
        self._next_y += dy * self.dt
        """
        #Computation of x and y using the classic Runge-Kutta method (rk4) 
        
        k1, k2, k3, k4 = self.rk4(self.dx_dt, self._next_x)
        delta_x = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        self._next_x += delta_x 

        q1, q2, q3, q4 = self.rk4(self.dy_dt, self._next_y)
        delta_y = (q1 + 2 * q2 + 2 * q3 + q4) / 6
        self._next_y += delta_y        

        
    def advance(self):
        self.x = self._next_x
        self.y = self._next_y
        self.model.computation_rho()
        
        # Update state 
        if self.model.schedule.steps % self.model.T == 0:
            self.update_state()
            

                
       


            
        
    