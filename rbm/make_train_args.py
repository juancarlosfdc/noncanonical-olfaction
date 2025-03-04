import numpy as np
import json 

args = json.load(open('train_args.json', 'r'))

args['epochs'] = 20

l2_regs = np.logspace(-4, -1, 13) 

for i, reg in enumerate(l2_regs): 
    args['l2_reg'] = reg
    with open(f'train_args_{i}.json', 'w') as out: 
        json.dump(args, out) 
