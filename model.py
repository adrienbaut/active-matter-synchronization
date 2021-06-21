# Imports
import numpy as np
from mesa.model import Model
from mesa.space import ContinuousSpace
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
from agent import AMSAgent
import random as rd

class AMSModel(Model):
    MODEL_TYPES = ('isolated', 'mean field', 'overlapping')
    AGENTS_CONFIGURATIONS = ('square', 'random', 'line') 
    
    def __init__(self, x_max, y_max, max_steps, model_type, initial_agents, initial_states, agents_parameters = {'preference' : 0, 'z' : 2/3}, seed = None):
        """
        Args:
            Space is continuous so x_max and y_max can be float
            model_type is 'isolated', 'mean field' or 'overlapping'
            initial_agents is the number of agents  
            agents_parameters includes the parameters that differ from one agent to another i.e. the position and their preference
            initial_states gives out the parameters that are common for all the agents as well as the fraction of agents in state 1, 
            the randomness on the radius, the configuration of the agents and the fluctuation for the dynamics
        """
        assert model_type in AMSModel.MODEL_TYPES,\
        "model_types should be either \"isolated\", \"mean field\" or \"overlapping\" "

        
        super().__init__()
        self.x_max = x_max
        self.y_max = y_max
        self.schedule = SimultaneousActivation(self)
        self.max_steps = max_steps
        self.model_type = model_type
        self.grid = ContinuousSpace(x_max=x_max, y_max=y_max, torus=True)
        self.configuration = initial_states[0]['configuration']
        self.frequency = initial_states[0]['frequency']
        self.N = initial_agents
        self.T = initial_states[0]['T'] # characteristic time before trying to update probabilities
        
        # Placement of agents
        for i in range(initial_agents):
            initial_states[i].update(agents_parameters)
            agent = AMSAgent(unique_id = i, model = self, initial_state = initial_states[i])
            self.schedule.add(agent)
            if self.configuration == 'line':
                agent.position = (i / initial_agents * self.x_max, self.y_max / 2)
                self.grid.place_agent(agent, agent.position)
            elif self.configuration == 'random':
                agent.position = (self.random.gauss(self.x_max/2, self.x_max/10),\
                                 self.random.gauss(self.y_max/2, self.y_max/10))
                self.grid.place_agent(agent, agent.position)
            else:
                if initial_agents != 4:
                    raise Exception('The \'square\' configuration only supports 4 agents!')
                if i <= 1:
                    agent.position = (i * self.x_max / 2 % self.x_max, i * self.x_max / 2 % self.x_max)
                    self.grid.place_agent(agent, agent.position) # diagonal terms (0,0) & (1,1) for 2x2 grid
                else:
                    agent.position = (i * self.x_max / 2 % self.x_max, (i+1) * self.x_max / 2 % self.x_max)
                    self.grid.place_agent(agent, agent.position) # anti diagonal terms (0,1) & (1,0) for 2x2 grid
            
            
        for agent in self.schedule.agents:
            if self.model_type == 'isolated':
                agent.set_radius(.1) # change position rather than radius for fixed position ? No need because isolated means no interaction with others
            elif self.model_type == 'mean field':
                agent.set_radius(np.sqrt(((self.x_max)/2)**2 + ((self.y_max)/2)**2))
            else:
                agent.neighbours = self.get_neighbours(agent) 
                self.overlap(agent)
                
        # Initialisation of rho for the agents
        self.computation_rho()
        for agent in self.schedule.agents:
            agent.update_transition_probability()
           
        # Setting the correct frequency of agents in state 1
        new_pref = self.random_frequency()
        for i in range(initial_agents):
            self.schedule.agents[i].preference = new_pref[i]            

        # To collect the data        
        self.datacollector = DataCollector(
            model_reporters = {'Time' : lambda model : model.schedule.time,
                          'x' : lambda model : model.get_x(),
                           'y' : lambda model : model.get_y(),
                          'Preferences' : lambda model : model.get_preferences(),
                          'Rho' : lambda model : model.get_rho(),
                          'Transition probabilities' : lambda model : model.get_transition_probabilities(),
                          'Positions' : lambda model : model.get_positions()}) # position not needed for final datacollector
        
    # For setting the pref randomly according to input frequency 
    def random_frequency(self):
        sample = [0]*self.N
        for i in range(self.frequency):
            sample[i]=1
        return rd.sample(sample, self.N)
    
    # Getters
    def get_x(self):
        return [agent.x for agent in self.schedule.agents]

    def get_y(self):
        return [agent.y for agent in self.schedule.agents]

    def get_preferences(self):
        return [agent.preference for agent in self.schedule.agents]

    def get_rho(self):
        return [agent.rho for agent in self.schedule.agents]

    def get_transition_probabilities(self):
        return [agent.transition_probability for agent in self.schedule.agents]
    
    def get_overlap(self):
        return [agent.overlap for agent in self.schedule.agents]
    
    def get_positions(self):
        return [agent.position for agent in self.schedule.agents]           
            
    def get_neighbours(self,agent):
        neighbours_list = []
        for agent2 in self.schedule.agents:
            if agent.position != agent2.position:
                if self.grid.get_distance(agent.position, agent2.position) <= (agent.radius + agent2.radius):
                    neighbours_list.append(agent2) 
        return neighbours_list
    
    # The value of the overlapping circles in the asymmetric lens formation was done using the formula from 
    # https://www.quora.com/How-do-you-find-out-the-area-of-intersection-of-two-circles on the 18/05/2021
    def overlap(self, agent):
        R = agent.radius
        area = []
        for neighbour in agent.neighbours: 
            d = self.grid.get_distance(agent.position, neighbour.position)
            r = neighbour.radius
            l = np.sqrt(((R+r)**2-d**2)*(d**2-(R-r)**2))/d
            alpha = np.arcsin(l/2*R)
            beta = np.arcsin(l/2*r)
            if d + r < R :  # neighbour's circle is included in the agent's one
                area.append(min(4*np.pi*(r**2), 4*np.pi*(R**2)))                    
            else: # asymmetric lens configuration
                area.append((alpha - np.sin(alpha)*np.cos(alpha))*R**2 + (beta - np.sin(beta)*np.cos(beta))* r**2)                
        agent.overlap.extend(area)
                
    # Computation of rho for the different model types
    def computation_rho(self):
        if self.model_type == 'isolated':
            for agent in self.schedule.agents:
                agent.rho = 1/agent.A * agent.x
        elif self.model_type == 'mean field':
            total_x = 0
            for agent in self.schedule.agents:
                total_x += agent.x 
            for agent in self.schedule.agents:
                agent.rho = total_x / agent.A  
        else:
            for agent in self.schedule.agents:
                overlap_density = 0
                for j in range(len(agent.neighbours)):
                    overlap_density += agent.neighbours[j].rho * agent.overlap[j] # problem for computing: rho^i depends on the other rhos that are not defined. Took isolated value as initial value for rho^i
                agent.rho = 1/agent.A * (overlap_density + agent.x)
        
    
    def step(self):
        '''Advance the model by one step.'''
        
        self.datacollector.collect(self)
        self.schedule.step()
        
        if self.schedule.steps > self.max_steps:
            self.running = False
    