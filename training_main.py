from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
from shutil import copyfile

from practice_simulation import Mysimulation

from generator import TrafficGenerator
from memory import Memory
from model import TrainModel
from visualization import Visualization
from utils import import_train_configuration, set_sumo, set_train_path


if __name__ == "__main__":

    config = import_train_configuration(config_file='training_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])

    Model = TrainModel(
        config['num_layers'], 
        config['width_layers'], 
        config['batch_size'], 
        config['learning_rate'], 
        input_dim=config['num_states'], 
        output_dim=config['num_actions']
    )

    Target_Model = TrainModel(
        config['num_layers'], 
        config['width_layers'], 
        config['batch_size'], 
        config['learning_rate'], 
        input_dim=config['num_states'], 
        output_dim=config['num_actions']
    )

    Target_Model._model.set_weights(Model._model.get_weights())

    Memory = Memory(
        config['memory_size_max'], 
        config['memory_size_min']
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

    Visualization = Visualization(
        path, 
        dpi=96
    )

    mysim = Mysimulation(
        Model,
        Target_Model,
        Memory,
        config['gamma'],
        TrafficGen, 
        sumo_cmd, 
        config['max_steps'],
        config['num_states'],
        config['num_actions'],
        config['training_epochs'])
    
    episode = 0
    timestamp_start = datetime.datetime.now()
    
    x = 10
    while episode < config['total_episodes']:
        print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']))
        epsilon = 1.0 - (episode / config['total_episodes'])  # set the epsilon for this episode according to epsilon-greedy policy
        simulation_time, training_time = mysim.run(episode, epsilon)  # run the simulation
        print('Simulation time:', simulation_time, 's Training time:', training_time, 's')
        episode += 1
        
        if episode%x == 0:
            Target_Model._model.set_weights(Model._model.get_weights())

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    Model.save_model(path)

    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))

    Visualization.save_data_and_plot(data = mysim._reward_store, filename='reward', xlabel='Episode', ylabel='Cumulative reward')
    Visualization.save_data_and_plot(data = mysim._cumulative_cost_store, filename='cost', xlabel='Episode', ylabel='Total cost')
    Visualization.save_data_and_plot(data = mysim._cumulative_chargeDelay_store, filename='chargedelay', xlabel='Episode', ylabel='Total chargedelay time')
    Visualization.save_data_and_plot(data = mysim.cumulative_chargingTime_store, filename='chargingtime', xlabel='Episode', ylabel='Total charging time')
    Visualization.save_data_and_plot(data = mysim.num_of_stops_store, filename='stops', xlabel='Episode', ylabel='num of stops')