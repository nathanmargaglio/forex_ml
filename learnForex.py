from logger import setup_custom_logger
from training import Environment, Agent
from datastore import DataManager, DataProcessor
from logger import setup_custom_logger
import inquirer
import os
import time

if __name__=="__main__":
    pairs = ['EUR/USD', 'AUD/JPY', 'AUD/NZD', 'AUD/USD', 'CAD/JPY', 'CHF/JPY', 'EUR/GBP', 'EUR/JPY', 'EUR/USD', 'GBP/JPY', 'GBP/USD', 'NZD/USD', 'USD/CAD']
    years = ['2012', '2013', '2014', '2015', '2016', '2017']
    months = ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']

    questions = [
        inquirer.Text('name', message="Name Instance [epoch time]: ",),
        inquirer.List('pair', message="Choose Currency Pair: ", choices=pairs),
        inquirer.List('render', message="Render Images: ", choices=['no', 'yes']),
        inquirer.Text('epochs', message="Epochs[10]: ",),
    ]
    answers = inquirer.prompt(questions)

    name = answers['name'].replace(' ', '_')
    if not name:
        name = str(int(time.time()))

    epochs = answers['epochs']
    if not epochs:
        epochs = 10
    epochs = int(epochs)

    _name = name
    weights = None
    for ep in range(epochs):
        name = _name + '_' + str(ep)
        instance_dir = 'data/instances/' + name + '/'
        pair = answers['pair']
        render_figures = True if answers['render'] == 'yes' else False

        if not os.path.exists(instance_dir):
            os.makedirs(instance_dir)

        dp = DataProcessor(instance_name=name)
        dm = DataManager(dp, instance_name=name)

        env = Environment(dm.get_slice_from_pair(pair), dp.state,
            instance_name=name, render_figures = render_figures)

        agent = Agent(dp.state.columns.size, 3, instance_name=name, weights=weights)
        env.run(agent)

        if os.path.isfile('data/instances/' + name + '/brain.h5'):
            weights = 'data/instances/' + name + '/brain.h5'
