import numpy as np
import json 

args = json.load(open('/n/home10/jfernandezdelcasti/noncanonical-olfaction/rbm/hyperparameters.json', 'r'))

args['epochs'] = 20
args['batch_size'] = 100
args['sample_number'] = 10000
args['n_hidden'] = 500

l2_regs = np.logspace(-4, -1, 7) 
gammas = np.logspace(-4, -1, 7)

k = 0
for i, reg in enumerate(l2_regs): 
    args['l2_reg'] = reg
    for j, gamma in enumerate(gammas): 
        args['learning_rate'] = gamma 
        with open(f'hyperparameters/hyperparameters_{k}.json', 'w') as out: 
            json.dump(args, out) 
            k += 1
