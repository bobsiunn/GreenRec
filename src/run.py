import simulator 
import argparse
import logging

logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)

parser = argparse.ArgumentParser()


parser.add_argument('-trace', '--trace', type=int, default=0, help='request trace mode (0:MovieLens, 1:Twitch)')
parser.add_argument('-mode', '--mode', type=int, default=0, help='request process mode (0:base, 1:approach)')
parser.add_argument('-const', '--const', type=float, default=0, help='accuracy constraint')
parser.add_argument('-delta', '--delta', type=int, default=0, help='GreenRec Planner refine parameter')

parser.add_argument('-rq_st', '--rq_start_time', type=int, default=0, help='request start time')
parser.add_argument('-rq_et', '--rq_end_time', type=int, default=0, help='request end time')
parser.add_argument('-ci_st', '--ci_start_time', type=int, default=0, help='carbon intensity start time')
parser.add_argument('-ci_et', '--ci_end_time', type=int, default=0, help='carbon intensity end time')

args = parser.parse_args()
simulator.simulation(args)

