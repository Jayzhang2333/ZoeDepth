
import time
import argparse


parser = argparse.ArgumentParser(description='CompletionFormer')


# Network
parser.add_argument('--prop_time',
                    type=int,
                    default=6,
                    help='number of propagation')
parser.add_argument('--prop_kernel',
                    type=int,
                    default=3,
                    help='propagation kernel size')
parser.add_argument('--preserve_input',
                    action='store_true',
                    default=False,
                    help='preserve input points by replacement')
parser.add_argument('--affinity',
                    type=str,
                    default='TGASS',
                    choices=('AS', 'ASS', 'TC', 'TGASS'),
                    help='affinity type (dynamic pos-neg, dynamic pos, '
                         'static pos-neg, static pos, none')
parser.add_argument('--affinity_gamma',
                    type=float,
                    default=0.5,
                    help='affinity gamma initial multiplier '
                         '(gamma = affinity_gamma * number of neighbors')
parser.add_argument('--conf_prop',
                    action='store_true',
                    default=True,
                    help='confidence for propagation')
parser.add_argument('--no_conf',
                    action='store_false',
                    dest='conf_prop',
                    help='no confidence for propagation')
parser.add_argument('--legacy',
                    action='store_true',
                    default=False,
                    help='legacy code support for pre-trained models')








args = parser.parse_args()
