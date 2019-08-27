import os
import argparse

def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--root', type=str, required=True,
                        help="root path to data directory")
    parser.add_argument('-d', '--dataset', type=str, required=True, nargs='+',
                        help="deepfashion or fld")

    parser.add_argument('-b', '--batchsize', type=int, default=50,
                        help='batchsize')
    parser.add_argument('--epoch', type=int, default=30,
                        help='the number of epoch')
    parser.add_argument('-lr','--learning-rate', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--evaluate', type=bool, default=False,
                        help='evaluation only')
    parser.add_argument('-w', '--weight-file', type=str, default=None,
                        help='weight file')
    parser.add_argument('--glem', type=bool, default=True,
                        help='global-local embedding module')
    parser.add_argument('--update-weight', type=bool, default=False)

    return parser

