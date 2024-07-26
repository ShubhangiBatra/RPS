from __future__ import absolute_import
from __future__ import print_function

import os
from shutil import copyfile

from real_simulation import RealSimulation
from generator import TrafficGenerator
from model import TestModel
from visualization import Visualization
from utils import import_test_configuration, set_sumo, set_test_path

# from fcfs_test import RealSimulation
# from random_test import RealSimulation

if __name__ == "__main__":

    config = import_test_configuration(config_file='testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    model_path, plot_path = set_test_path(config['models_path_name'], config['model_to_test'])

    Model = TestModel(
        input_dim=config['num_states'],
        model_path=model_path
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

    Visualization = Visualization(
        plot_path, 
        dpi=96
    )

    realSimulation = RealSimulation(
        Model,
        TrafficGen,
        sumo_cmd,
        config['num_states'],
        config['num_actions']
    )

    print('\n----- Test episode')
    simulation_time = realSimulation.run(config['episode_seed'])  # run the simulation
    print('Simulation time:', simulation_time, 's')

    print("----- Testing info saved at:", plot_path)

    copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini'))

    Visualization.save_data_and_plot(data=realSimulation._reward_episode, filename='reward', xlabel='Action step', ylabel='Reward')
    Visualization.save_data_and_plot(data=realSimulation._total_charging_time_episode , filename='waiting_time', xlabel='Step', ylabel='waiting_time')
    Visualization.save_data_and_plot(data=realSimulation._total_cost_episode, filename='cost', xlabel='step', ylabel='cost')
    Visualization.save_data_and_plot(data=realSimulation._total_chargeDelay_epidode, filename='chargeDelay', xlabel='step', ylabel='chargeDelay')
    Visualization.save_data_and_plot(data=realSimulation._num_stops_episode, filename='stops', xlabel='step', ylabel='stop')
    
