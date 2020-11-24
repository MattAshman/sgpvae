import argparse
import os
import pickle

__all__ = ['str2bool', 'save']


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save(args, metrics):
    if 'results_dir' in args.keys():
        results_dir = args['results_dir'] + args['model']
    else:
        results_dir = '_results/' + args['model']

    if os.path.isdir(results_dir):
        i = 1
        while os.path.isdir(results_dir + '_' + str(i)):
            i += 1

        results_dir += '_' + str(i)

    os.makedirs(results_dir, exist_ok=True)

    # Pickle args and metrics.
    with open(results_dir + '/args.pkl', 'wb') as f:
        pickle.dump(args, f)

    with open(results_dir + '/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    # Save args and results in text format.
    with open(results_dir + '/results.txt', 'w') as f:
        f.write('Args: \n')
        if isinstance(args, list):
            for d in args:
                f.write(str(d) + '\n')
        else:
            f.write(str(args) + '\n')

        f.write('\nMetrics: \n')
        for (key, value) in metrics.items():
            f.write('{}: {}\n'.format(key, value))
