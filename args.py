import argparse

    
    
class KeyValueAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # Split values into key-value pairs
        items = [item.split('=') for item in values]
        # Construct dictionary
        setattr(namespace, self.dest, dict(items))


def parse_args():
    parser = argparse.ArgumentParser(description='REDQ')
    
    parser.add_argument('--env', type=str, default='LunarLander-v2', help='Gym environment')
    parser.add_argument('--seed', type=int, default=0, help='Seed')
    parser.add_argument('--total_timesteps', type=int, default=200_000, help='Total time steps')
    parser.add_argument('--exp_name', type=str, default='REDQ', help='Experiment name')
    parser.add_argument('--kwargs', nargs='*', action=KeyValueAction,
                    help='Hyperparameters as key-value pairs (e.g., N=10 G=20)')
    
    return parser.parse_args()
    
 
OPT = parse_args()
