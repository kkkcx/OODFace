import os
import sys
# current_directory = os.path.dirname(os.path.abspath(__file__))
# root_path = os.path.abspath(os.path.dirname(current_directory) + os.path.sep + ".")
# sys.path.append(root_path)
import sys
sys.path.append(".")

from RobFR.networks.get_model import getmodel
import torch
import cv2
import numpy as np
from scipy import misc
import argparse
# from RobFR.benchmark.lfw.utils import read_pair, cosdistance
from RobFR.benchmark.utils import cosdistance
from RobFR.dataset.base  import read_pair

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='White-box model', type=str, default='MobileFace')
parser.add_argument('--log', help='log file', type=str, default='log/log.txt')
args = parser.parse_args()

config = {}
def read_pairs(path):
    with open(os.path.join(path, 'pairs_og.txt')) as f:
        lines = f.readlines() 
    suffix = '.jpg'
    pairs = []
    for line in lines:
        line = line.strip().split('\t')
        if len(line) == 3:
            pair = []
            pair.append(os.path.join(line[0], line[0] + '_' + line[1].zfill(4) + suffix))
            pair.append(os.path.join(line[0], line[0] + '_' + line[2].zfill(4) + suffix))
            pair.append(0)
            pairs.append(pair)
        elif len(line) == 4:
            pair = []
            pair.append(os.path.join(line[0], line[0] + '_' + line[1].zfill(4) + suffix))
            pair.append(os.path.join(line[2], line[2] + '_' + line[3].zfill(4) + suffix))
            pair.append(1)
            pairs.append(pair)
        else:
            pass
    return pairs
def test(pairs, model, shape, model_name):
#    logfile = open('logsim/log-{}.txt'.format(model_name), 'w')
    num = len(pairs)
    result_cos = np.empty(shape=(num))
    gt = np.empty(shape=(num))
    
    zero_matrix = np.zeros(shape=(1, shape[0], shape[1], 3))
    data_path = os.path.join('data', 'lfw-{}x{}'.format(shape[0], shape[1]))

    for i in tqdm(range(num)):
        logits = []
        for filename in pairs[i][:2]:
            device = torch.device("cuda")
            # _, emb = read_pair(os.path.join(data_path, filename), model)
            _, emb = read_pair(os.path.join(data_path, filename), device, model, return_feat=True)

            logits.append(emb)
        result_cos[i] = cosdistance(logits[0], logits[1]).item()
        gt[i] = pairs[i][2]
#        logfile.write('{} {}\n'.format(result_cos[i], gt[i]))
    cos_best_value = 0
    for x in result_cos:
        acc = (result_cos < x) == gt
        acc = acc.sum() * 1.0 / num
        if cos_best_value < acc:
            cos_best_value = acc
            cos_best_threshold = x
    print('cos distance: ', cos_best_value, cos_best_threshold, model_name)
    config[model_name] = {
        'cos': cos_best_threshold,
        'cos_acc': cos_best_value,
    }

if __name__ == '__main__':
    dst = os.path.join('./data')
    pairs = read_pairs(dst)
    # print(pairs)
    print(args.model)
    model, shape = getmodel(args.model)
    test(pairs, model, shape, args.model) 
    with open(args.log, 'w') as f:
        for k, v in config.items():
            f.write('{}:{},\n'.format(k, v))

