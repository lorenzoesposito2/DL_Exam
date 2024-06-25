from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from agent import Agent

# research space
space = {
    'learning_rate': hp.loguniform('learning_rate', -6, 0),
    'gamma': hp.uniform('gamma', 0.6, 0.9),
    'epsilon_decay': hp.uniform('epsilon_decay', 0.9, 0.9999),
    'hidden_size' : hp.uniform('hidden_size', 64, 256)
}

# Definisci la funzione obiettivo
def objective(params):
    agent = Agent(learning_rate=params['learning_rate'], gamma=params['gamma'], epsilon_decay=params['epsilon_decay'], hidden_size=params['hidden_size'])
    loss = agent.train_hyper()  
    return {'loss': loss, 'status': STATUS_OK}

trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=5, trials=trials)
space_eval(space, best)

print(best)

with open('best_values.txt', 'w') as f:
    f.write('Best values from hyperopt optimization:\n')
    f.write(str(best))