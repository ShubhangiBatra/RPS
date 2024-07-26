import traci
import numpy as np
import timeit
import random
import traceback
import xml.etree.ElementTree as ET
import re

from sklearn.preprocessing import StandardScaler

class Mysimulation:
    
    def __init__(self, Model, Target_Model, Memory, gamma, TrafficGen, sumo_cmd, max_steps, num_states, num_actions, training_epochs):
        self._TrafficGen = TrafficGen
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._vehicleId = None
        self._step = 0
        self._num_states = num_states
        self._num_actions = num_actions
        self._Model = Model
        self._Target_Model = Target_Model
        self._Memory = Memory
        self._gamma = gamma
        self._training_epochs = training_epochs
       

        #data of each episode is stored here
        self._reward_store = []
        self._cumulative_chargeDelay_store = []
        self._cumulative_cost_store = []
        self._num_of_stops_store = []
        self._cumulative_chargingTime_store = []

        #Keeping track of the max_ and min_ of (chargingTime, chargeDelay, cost)
        self._maxCost = 0
        self._maxChargingTime = 0
        self._maxChargeDelay = 0

        self._vid = 'v_1'

    def run(self, episode, epsilon):
        """
        Runs an episode of simulation, then starts a training session
        """
        start_time = timeit.default_timer()

        # #Generate charging stations file for this simulation
        self._TrafficGen.generate_chargingstationfile(seed=episode)

        # first, generate the route file for this simulation and set up sumo
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0

        self._sum_reward = 0 #car will get a reward at each cs, for the whole episode this will store the tot. reward
        self._sum_charge_delay = 0.0
        self._sum_cost = 0.0
        self._num_of_stops = 0
        self._travel_time = 0
        self._sum_charging_time = 0
        self._rl_step = 0

        vid = 'v_1'

        #Adding the agent ev of v-type = 'ev', with a random route
        routeId = random.choice(self._getRouteIds())
        routeId = 'r_0'

        traci.vehicle.add(vid, routeID=routeId, typeID="ev")

        traci.vehicle.setParameter(vid, "device.battery.maximumBatteryCapacity", 1000)
        traci.vehicle.setParameter(vid, "device.battery.actualBatteryCapacity", 400)
        
        self.show_training_config(vid, routeId)

        prev_csid = "NULL"
        prev_state = np.asarray([0, 0, 0, 0, 0, 0, 0])
        prev_action = -1

        while (vid in traci.vehicle.getIDList()) or (self._step == 0):
            #generating samples
            
            try:
                
                curr_csid = self._ev_status(vid)
                reward_ = 0
                action_ = -1
                state_ = []
                if prev_csid != curr_csid and curr_csid != "NULL":
                    prev_csid = curr_csid
                    
                    state = self._get_state(vid, curr_csid)

                    state_ = state

                    #choosing the action, whether to stop or not at the current charging station
                    action = self._choose_action(state, epsilon)

                    action_ = action
                    print("At charging station {}, Action is {}".format(curr_csid, action))

                    if action == 0:
                        reward = self._calculate_reward(vid, curr_csid)
                        reward_ = reward
                        self._sum_reward += reward
                        print("Episode : ", episode, " RL_step : ", self._rl_step, " Reward : ", reward)
                    else:
                        charging_time, charging_cost = self._get_charge_duration(vid, curr_csid)
                        chargeDelay = self.getChargeDelay(curr_csid)
                        traci.vehicle.setChargingStationStop(vid, curr_csid, duration = charging_time + chargeDelay)

                        #updating the max and values of delay, charge time and cost
                        self._maxChargeDelay = max(self._maxChargeDelay, chargeDelay)
                        self._maxChargingTime = max(self._maxChargingTime, charging_time)
                        self._maxCost = max(self._maxCost, charging_cost)

                        while self._ev_status(vid) != "NULL":
                            self._simulate(1)
                            continue

                        reward = self._calculate_reward(vid, curr_csid, charging_time, chargeDelay, charging_cost, 1)
                        reward_ = reward
                        print("Episode : ", episode, " RL_step : ", self._rl_step, " Reward : ", reward)

                        self._sum_cost += charging_cost
                        self._sum_charge_delay += chargeDelay
                        self._sum_charging_time += charging_time
                        self._num_of_stops += 1
                        self._sum_reward += reward
                   
                    next_state = self._get_state(vid, curr_csid)
                    
                    #adding the sample into the memory after executing the action
                    self._Memory.add_sample((state, action, reward, next_state))
            

                    #Training after each step of rl
                    print("Training...")
                    start_time = timeit.default_timer()
                    for _ in range(self._training_epochs):
                        print("training epoch {}".format(_))
                        self._replay()
                    training_time = round(timeit.default_timer() - start_time, 1)
                    print("Training time : ", training_time)

                    #updating prev state info
                    prev_state = state
                    prev_action = action

                    self._rl_step += 1

                else:
                    self._simulate(1)
                    if self.getActualBatteryCapacity(vid)==0:
                        self._Memory.add_sample((prev_state, prev_action, -10, self._get_state(vid, curr_csid)))
                        break

                if (self.getActualBatteryCapacity(vid)==0):
                    print("State : ", state_)
                    print("Action :", action_)
                    print("Reward : ", reward_)

            except Exception:
                traceback.print_exc()
                break

        self._save_episode_stats()
        print("Total reward:", self._sum_reward, "- Epsilon:", round(epsilon, 2))    
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, 0
    
    def show_training_config(self, vid, routeId):
        print("******************** Vehicle info ********************\n")
        print("vid : ", vid)
        print("maximum_battery_capacity : ", self.getMaximumBatteryCapacity(vid))
        print("actual_battery_capacity : ", self.getActualBatteryCapacity(vid))
        print("The selected routeId is : ", routeId)
        
        print("\n******************** charging station info ********************\n")

        for csid in traci.chargingstation.getIDList():
            print("csid : ", csid)
            print("power : {}, cost : {}, chargeDelay : {}".format(self.getPower(csid), self.getCost(csid), self.getChargeDelay(csid)))


    def _get_charge_duration(self, vid, curr_csid):
        soc = self.getActualBatteryCapacity(vid)
        max_capacity = self.getMaximumBatteryCapacity(vid)
        power = self.getPower(curr_csid)

        ct = np.rint(((max_capacity-soc)/power) * 3600)
        cc = np.rint(((max_capacity-soc)/1000.0)*self.getCost(curr_csid))
        
        print("Charging time: ", ct)
        print("charging cost: ", cc)

        return ct, cc

    def _simulate(self, steps_todo):
        """
        Simulates the waiting and charging time of the EV vehichle
        Car will have Blue in wait, and Yellow during charge
        """
        traci.simulationStep()
        # if self._vid in traci.vehicle.getIDList():
        #     print("current soc is {}".format(self.getActualBatteryCapacity(self._vid)))
        self._step += 1


    def _calculate_reward(self, vid, curr_csid, charging_time=0, chargeDelay=0, cost=0, stopped=0): #charging_time=overall time spent at a cs
        
        # normalize param before next step
        n_cost = 0 if self._maxCost==0 else cost/self._maxCost
        n_chargingTime = 0 if self._maxChargingTime==0 else charging_time/self._maxChargingTime
        n_chargeDelay = 0 if self._maxChargeDelay==0 else chargeDelay/self._maxChargeDelay

        multiplier = 1
        soc = self.getActualBatteryCapacity(vid)/self.getMaximumBatteryCapacity(vid) * 100

        if soc>=50:
            multiplier = 1
        elif soc>=20:
            multiplier = 1/2
        else:
            multiplier = -2

        reward = multiplier * (1/10 * (1 - n_chargeDelay) +  1/10 * (1 - n_chargingTime) +  1/10 * (1 - n_cost) +  7/10 * (1 - stopped))
        
        return reward

    def _ev_status(self, vid):
    
        charging_Station_list = traci.chargingstation.getIDList()
        
        for csid in charging_Station_list:
            if traci.chargingstation.getLaneID(csid) != traci.vehicle.getLaneID(vid):
                continue
            cs_start_loc = traci.chargingstation.getStartPos(csid)
            cs_end_loc = traci.chargingstation.getEndPos(csid)
            veh_loc = traci.vehicle.getLanePosition(vid)
            if veh_loc >= cs_start_loc and veh_loc <= cs_end_loc:
                return csid
        
        return "NULL"

    def _get_state(self, vid, curr_csid):
        """
        Retrieve the information of charging stations and EV from SUMO
        """
        state = []
        charging_Station_list = self._getChargingStationsOnRoute('r_0')
        
        index = charging_Station_list.index(curr_csid)
        
        cd1 = self.getChargeDelay(curr_csid)
        cost1 = self.getCost(curr_csid)
        power1 = self.getPower(curr_csid)
        
        cd2 = 0
        cost2 = 0
        power2 = 0
        
        if index+1 < len(charging_Station_list):
            cd2 = self.getChargeDelay(charging_Station_list[index+1])
            cost2 = self.getCost(charging_Station_list[index+1])
            power2 = self.getPower(charging_Station_list[index+1])

        state.extend([power1, cost1, cd1, power2, cost2, cd2, self.getActualBatteryCapacity(vid)/self.getMaximumBatteryCapacity(vid)*100])
        
        return np.asarray(state)


    def _choose_action(self, state, epsilon):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action
        else:
            return np.argmax(self._Model.predict_one(state)) # the best action given the current state
   

    def _replay(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """

        batch = self._Memory.get_samples(self._Model.batch_size)

        if len(batch) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch

            # prediction
            q_s_a = self._Model.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_d = self._Model.predict_batch(next_states)  # predict Q(next_state), for every sample

            # print("This is q_s_a : ")
            # print(q_s_a)

            # setup training arrays
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value
                
            scaler = StandardScaler()
            x = scaler.fit_transform(x)
            y = scaler.fit_transform(y)
            self._Model.train_batch(x, y)  # train the NN


    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store.append(self._sum_reward)  # how much reward in this episode
        self._cumulative_chargeDelay_store.append(self._sum_charge_delay)  # total number of seconds waited by the car
        self._cumulative_cost_store.append(self._sum_cost)  # total cost of charging in each episode
        self._num_of_stops_store.append(self._num_of_stops) #
        self._cumulative_chargingTime_store.append(self._sum_charging_time)


    def _getRouteIds(self):

        regex_pattern = r'id="([^"]+)"'

        # Parse the XML file
        tree = ET.parse('./environment/grid/new_routes.rou.xml')
        root = tree.getroot()

        # Convert XML to string
        xml_string = ET.tostring(root, encoding='utf8', method='xml')

        xml_data = xml_string.decode('utf-8')

        route_ids = re.findall(regex_pattern, xml_data)

        return route_ids

    def _getChargingStationsOnRoute(self, routeId):

        csId = ['cs_1', 'cs_330', 'cs_329', 'cs_340', 'cs_51', 'cs_50', 'cs_71', 'cs_74', 'cs_127', 'cs_126', 'cs_164', 'cs_167', 'cs_202', 'cs_200', 'cs_258', 'cs_261']

        return csId


    def getPower(self, csid):
        return float(traci.chargingstation.getParameter(csid, "power"))

    def getCost(self, csid):
        return float(traci.chargingstation.getParameter(csid, "cost"))

    def getChargeDelay(self, csid):
        return float(traci.chargingstation.getParameter(csid, "chargeDelay"))

    def getActualBatteryCapacity(self, vid):
        return float(traci.vehicle.getParameter(vid, "device.battery.actualBatteryCapacity"))

    def getMaximumBatteryCapacity(self, vid):
        return float(traci.vehicle.getParameter(vid, "device.battery.maximumBatteryCapacity"))

    def getTravelledDistance(self, vid):
        return float(traci.vehicle.getDistance(vid))


    @property
    def reward_store(self):
        return self._reward_store

    @property
    def cumulative_chargingTime_store(self):
        return self._cumulative_chargingTime_store

    @property
    def cumulative_cost_store(self):
        return self._cumulative_cost_store
    
    @property
    def cumulative_chargeDelay_store(self):
        return self._cumulative_chargeDelay_store

    @property
    def num_of_stops_store(self):
        return self._num_of_stops_store

