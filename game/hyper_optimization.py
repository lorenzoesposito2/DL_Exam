from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from agent import Agent
import csv

space = {
    'learning_rate': hp.loguniform('learning_rate', -6, 0),
    'gamma': hp.uniform('gamma', 0.6, 0.9),
    'epsilon_decay': hp.uniform('epsilon_decay', 0.9, 0.9999),
    'hidden_size' : hp.uniform('hidden_size', 64, 256)
}

def objective(params):
    agent = Agent(learning_rate=params['learning_rate'], gamma=params['gamma'], epsilon_decay=params['epsilon_decay'], hidden_size=params['hidden_size'])
    loss = agent.tune_hyper(agent) 
    with open('hyperopt_results.csv', 'a', newline='') as csvfile:
        fieldnames = ['learning_rate', 'gamma', 'epsilon_decay', 'hidden_size', 'loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'learning_rate': params['learning_rate'], 'gamma': params['gamma'], 'epsilon_decay': params['epsilon_decay'], 'hidden_size': params['hidden_size'], 'loss': loss}) 
    return {'loss': loss, 'status': STATUS_OK}

with open('hyperopt_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['learning_rate', 'gamma', 'epsilon_decay', 'hidden_size', 'loss']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=15, trials=trials)
space_eval(space, best)

print(best)

with open('best_values_relu.txt', 'w') as f:
    f.write('Best values from hyperopt optimization:\n')
    f.write(str(best))