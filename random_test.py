import traci
import timeit
import numpy as np
import random
import traceback


class RealSimulation():
    def __init__(self, Model, TrafficGen, sumo_cmd, num_states, num_actions):
        self._Model = Model
        self._TrafficGen = TrafficGen
        self._sumo_cmd = sumo_cmd
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_episode = []
        self._total_charging_time_episode = []
        self._total_cost_episode = []
        self._total_chargeDelay_epidode = []
        self._num_stops_episode = []
        self._vid = 'v_1'

        #Keeping track of the max_ of (chargingTime, chargeDelay, cost)
        self._maxCost = 0
        self._maxChargingTime = 0
        self._maxChargeDelay = 0


    def run(self, episode):
        """
        Runs the testing simulation
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        # self._TrafficGen.generate_chargingstationfile(seed=episode)
        
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._sum_reward = 0 #car will get a reward at each cs, for the whole episode this will store the tot. reward
        self._sum_charging_time = 0.0
        self._sum_cost = 0.0
        self._sum_charge_delay = 0.0
        self._num_of_stops = 0
        self._travel_time = 0
        vid = 'v_1'
        
        #Adding the vehicle of our interest into the sumo environment
        traci.vehicle.add(vid, routeID='r_0', typeID='ev')
        traci.vehicle.setColor(vid, (255, 0, 0))
        traci.vehicle.setParameter(vid,"device.battery.maximumBatteryCapacity", 1000.0)
        traci.vehicle.setParameter(vid,"device.battery.actualBatteryCapacity", 400.0)

        self.show_training_config(vid, 'r_0')
        
        prev_csid = "NULL"
        prev_state = self._get_state(vid)
        prev_action = -1

        while (vid in traci.vehicle.getIDList()) or (self._step==0):


            try:
                curr_csid = self._ev_status(vid)

                if prev_csid != curr_csid and curr_csid != "NULL":  
                    prev_csid = curr_csid

                    # get current state of the intersection
                    current_state = self._get_state(vid)
                    action = self._choose_action(current_state)
                    
                    print("At cs: ", curr_csid, " action is ", action)

                    if action == 0:
                        reward = self._calculate_reward(vid, curr_csid)
                        chargeDelay = 0
                        charging_time = 0
                        charging_cost = 0
                        self._sum_reward += reward
                    else:
                        charging_time, charging_cost = self._get_charge_duration(vid, curr_csid)
                        chargeDelay = self.getChargeDelay(curr_csid)
                        traci.vehicle.setChargingStationStop(vid, curr_csid, duration = charging_time + chargeDelay)

                        #updating the max values of delay, charge time and cost
                        self._maxChargeDelay = max(self._maxChargeDelay, chargeDelay)
                        self._maxChargingTime = max(self._maxChargingTime, charging_time)
                        self._maxCost = max(self._maxCost, charging_cost)

                        reward = self._calculate_reward(vid, curr_csid, charging_time, chargeDelay, charging_cost, 1)

                        self._sum_charge_delay += chargeDelay
                        self._sum_charging_time += charging_time
                        self._sum_cost += charging_cost
                        self._sum_reward += reward
                        self._num_of_stops += 1
                        
                    self._reward_episode.append(reward)    
                    self._total_charging_time_episode.append(charging_time)
                    self._total_cost_episode.append(charging_cost)
                    self._total_chargeDelay_epidode.append(chargeDelay)
                    self._num_stops_episode.append(action)
                    
                else:
                    if self.getActualBatteryCapacity(vid) == 0.0:
                        print("SOC = 0% Battery empty!!!!")
                        break
                    self._simulate(1)

            except Exception:
                traceback.print_exc()
                break

        

        #print("Total reward:", np.sum(self._reward_episode))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Total cost of Trip : ", self._sum_cost)
        print("Total charging time : ", self._sum_charging_time)
        print("Total charge delay : ", self._sum_charge_delay)
        print("Total reward of trip : ", self._sum_reward)
        print("Total charging stops : ", self._num_of_stops)
        print("Travel time : ", self._step)

        return simulation_time
    

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
    

    def _get_state(self, vid):
        """
        Retrieve the information of charging stations and EV from SUMO
        """
        state = []
        charging_Station_list = traci.chargingstation.getIDList()

        for charging_station_id in charging_Station_list:
            charge_delay = self.getChargeDelay(charging_station_id) #charge_delay
            cost = self.getCost(charging_station_id)
            power = self.getPower(charging_station_id)

            state.extend([charge_delay, cost, power])
        
        ev_info = [
            15885.56 - self.getTravelledDistance(vid),
            self._sum_cost,
            self._sum_charge_delay,
            self._sum_charging_time,
            self._num_of_stops
        ]

        state.extend(ev_info)
        
        return np.asarray(state)


    def _choose_action(self, state):
        """
        Pick the best action known based on the current state of the env
        """
        return np.random.choice([0, 1])
        

    def _calculate_reward(self, vid, curr_csid, charging_time=0, chargeDelay=0, cost=0, stopped=0): #charging_time=overall time spent at a cs
        
        # normalize param before next step
        n_cost = 0 if self._maxCost==0 else cost/self._maxCost
        n_chargingTime = 0 if self._maxChargingTime==0 else charging_time/self._maxChargingTime
        n_chargeDelay = 0 if self._maxChargeDelay==0 else chargeDelay/self._maxChargeDelay

        reward = 1/10 * (1 - n_chargeDelay) +  1/10 * (1 - n_chargingTime) + 1/10 * (1 - n_cost) + 7/10 * (1 - stopped)
        
        return reward
    


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

    def _simulate(self, steps_todo):
        """
        Simulates the waiting and charging time of the EV vehichle
        Car will have Blue in wait, and Yellow during charge
        """
        traci.simulationStep()
        self._step += 1

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


    @property
    def reward_episode(self):
        return self._reward_episode
    
    @property
    def cost_episode(self):
        return self._total_cost_episode
    
    @property
    def waiting_time_episode(self):
        return self._total_charging_time_episode
