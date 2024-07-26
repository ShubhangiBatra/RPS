import numpy as np
import math
import traci
import xml.etree.ElementTree as ET
import re
import random

class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated):
        self._n_cars_generated = n_cars_generated  # how many cars per episode
        self._max_steps = max_steps
        self.laneIds = self._getLaneIds()

    def generate_routefile(self, seed):
        """
        Generation of the route of every car for one episode
        """
        pass

    def generate_chargingstationfile(self, seed):

        """ 
        Generation of charging stations data 
        """

        # We are randomizing max_bat_cap, intial_actual_bat_cap, power_of_cs, cost_per_unit of cs
        power_upper_limit = 50000 #W
        power_lower_limit = 1000    #W
        chargeDelay_upper_limit = 10 #s
        chargeDelay_lower_limit = 1 #s
        cost_upper_limit = 15 #rupees/kWh
        cost_lower_limit = 5 #rupees/kWh


        with open("environment/grid/new_charging.add.xml", "w") as stations:
            
            print("""<?xml version="1.0" encoding="UTF-8"?>

<additional>""", file=stations)
            
            for counter, laneId in enumerate(self.laneIds):
                cost = np.rint(random.random()*(cost_upper_limit - cost_lower_limit) + cost_lower_limit)
                power = np.rint(random.random()*(power_upper_limit - power_lower_limit) + power_lower_limit)
                chargeDelay = np.rint(random.random()*(chargeDelay_upper_limit - chargeDelay_lower_limit) + chargeDelay_lower_limit)
                print("""   <chargingStation id="cs_{}" lane="{}" startPos="300.00" endPos="400.00" power="{}" chargeDelay="{}">
        <param key="cost" value="{}"/>
        <param key="power" value="{}"/>
        <param key="chargeDelay" value="{}"/>
    </chargingStation>""".format(counter, laneId, power, chargeDelay, cost, power, chargeDelay), file=stations)

            print("</additional>", file=stations)
            

    def _getLaneIds(self):
        # Regular expression to match lane IDs
        regex_pattern = r'lane="([^"]+)"'

        # Parse the XML file
        tree = ET.parse('./environment/grid/new_charging.add.xml')
        root = tree.getroot()

        # Convert XML to string
        xml_string = ET.tostring(root, encoding='utf8', method='xml')

        xml_data = xml_string.decode('utf-8')

        # Find all matches
        lane_ids = re.findall(regex_pattern, xml_data)

        return lane_ids
