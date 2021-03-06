import os 
import numpy as np
import torch
import model
import cv2 

np. random.seed(1000)

def save_nets(nets, model_dir):
    names = ['weights_unet']

    if not os.path.exists(model_dir):
        print('Creating model directory: {}'.format(model_dir))
        os.makedirs(model_dir)

    for net_idx, net in enumerate(nets):
        if net is not None:
            torch.save(net.state_dict(), '{}/{}.pth'.format(
                model_dir, names[net_idx])) 

def save_predictions(preds, fns, out_dir):

    if not os.path.exists(out_dir):
        print('Creating output directory: {}'.format(out_dir))
        os.makedirs(out_dir)

    for idx, pred in enumerate(preds):
        pred = torch.squeeze(pred, dim=0)
        flag = cv2.imwrite(out_dir + '/{}'.format(fns[idx]), pred.cpu().numpy())
        assert flag == True 

def load_best_weights(model, model_dir):
    model.load_state_dict(torch.load('{}/weights_unet.pth'.format(model_dir)))
    return model 


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            # self.step = lambda a: False

    def step(self, metrics):
        if self.patience == 0:
            return False, self.best, self.num_bad_epochs
            
        if self.best is None:
            self.best = metrics
            return False, self.best, 1

        if torch.isnan(metrics):
            return True, self.best, self.num_bad_epochs

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True, self.best, self.num_bad_epochs

        return False, self.best, self.num_bad_epochs

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)