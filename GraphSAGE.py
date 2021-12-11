import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

import dgl.function as fn
from sklearn.metrics import roc_auc_score,roc_curve

from DataLoader import create_ida_graphs
from Configure import parse_configs, print_configs

from os import path
from itertools import chain

import warnings
warnings.filterwarnings("ignore")

from dgl.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self,configs,input_size,hidden_size):
        super().__init__()
        self.configs = configs
        self.c1 = SAGEConv(input_size,  hidden_size, configs.aggregator)
        self.c2 = SAGEConv(hidden_size, hidden_size, configs.aggregator)
    
    def forward(self, g, features):
        h = self.c1(g, features)
        h = self.c2(g, F.relu(h))
        return h

class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return g.edata['score'][:, 0]


class MyModel(object):

    def __init__(self, configs, name=None):
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.configs = configs
        self.name = name

        self.network = GraphSAGE(configs,2+2*configs.window_size,32).to(self._device)
        self.pred = DotPredictor()

        self.train_data, self.test_data = create_ida_graphs()


    def train(self):
        # Optimizer
        if self.configs.adam: optimizer = torch.optim.Adam(chain(self.network.parameters(), self.pred.parameters()), lr=self.configs.learning_rate, weight_decay=self.configs.weight_decay)
        else:                 optimizer = torch.optim.SGD( chain(self.network.parameters(), self.pred.parameters()), lr=self.configs.learning_rate, weight_decay=self.configs.weight_decay, momentum=0.9)

        # Step Scheduler
        if self.configs.step_schedule:
            if self.configs.cosine: scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.configs.epochs)
            else:                   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

        # Criterion
        def loss(h,pos_g,neg_g):
            pos_score = self.pred(pos_g, h)
            neg_score = self.pred(neg_g, h)
            scores = torch.cat([pos_score, neg_score])
            labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
            return F.binary_cross_entropy_with_logits(scores, labels)

        print("--- Training Start ---")
        global_step = 0
        # Epoch Loop
        for i in range(self.configs.epochs):
            total_loss = 0.0
            for x,p,n in self.train_data:
                self.network.train()
                self.pred.train()

                # Compute loss
                h = self.network(x,x.ndata['feat'])
                loss_val = loss(h,p,n)

                # Grad Step
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                total_loss += loss_val.item()


            # Step Learning Rate
            if self.configs.step_schedule:
                if self.configs.cosine: scheduler.step()
                else:                   scheduler.step(total_loss)

            # Compute ROC
            ra_score = self.roc_auc(h,p,n)
            print('Epoch: {:4d} -- Training Loss: {:10.6f} -- ROC Score: {:10.6f}'.format(i, total_loss, ra_score), end='\n')
        print("--- Training Complete ---")

    def roc_auc(self,h,pos_g,neg_g,curve=False):
        self.network.eval()
        self.pred.eval()
        pscore = self.pred(pos_g, h).detach()
        nscore = self.pred(neg_g, h).detach()
        labels = torch.cat([torch.ones( pscore.shape[0]),
                            torch.zeros(nscore.shape[0])]).numpy()
        scores = torch.cat([pscore, nscore]).numpy()
        score = roc_auc_score(labels, scores)
        if curve:
            return score, roc_curve(labels, scores)
        return score

    def calc_thresh(self,fp,tp,thresh):
        import pandas as pd
        roc = pd.DataFrame({'tf' : pd.Series(tp-(1-fp), index=np.arange(len(tp))), 'threshold' : pd.Series(thresh, index=np.arange(len(tp)))})
        roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

        return list(roc_t['threshold']) 

    def evaluate(self, x=None, y=None, test_data=None):
        # TODO: quality of life upgrades
        test_data = test_data if test_data else self.test_data 
        self.network.eval()
        self.pred.eval()
        score = 0
        for x,p,n in self.test_data:
            h = self.network(x, x.ndata['feat'])
            rscore,(fp,tp,thresh) = self.roc_auc(h,p,n,curve=True)
            score += rscore
        return score/len(self.test_data), self.calc_thresh(fp,tp,thresh)
        
    def predict(self, x=None):
        if x is None:
            for g,_,_ in self.test_data:
                pass
            return self.network(g,g.ndata['feat'])
        return self.network(x, x.ndata['feat'])




if __name__ == '__main__':
    # Get training and model configurations
    configs = parse_configs()

    # If help flag raised print help message and exit
    if configs.help:
        print_configs()
    else:
        print_configs(configs)

        # Generate model folder and name
        model = MyModel(configs)
        model.train()
        roc_score, best_threshold = model.evaluate()
        print('Score:',roc_score)
        print('Threshold:',best_threshold)

