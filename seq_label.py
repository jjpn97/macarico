# Sequence labeler model
from __future__ import division, generators, print_function
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import argparse
import csv
from collections import defaultdict

import macarico.util as util
import macarico.data.synthetic as synth
from macarico.data.types import Dependencies

from macarico.lts.searn import SEARN
from macarico.lts.dagger import DAgger, Coaching
from macarico.lts.behavioral_cloning import BehavioralCloning
from macarico.lts.aggrevate import AggreVaTe
from macarico.lts.lols import LOLS, BanditLOLS
from macarico.lts.reinforce import Reinforce, LinearValueFn, A2C

from macarico.annealing import ExponentialAnnealing, NoAnnealing, Averaging, EWMA
from macarico.features.sequence import EmbeddingFeatures, BOWFeatures, RNN, DilatedCNN, AttendAt, FrontBackAttention, SoftmaxAttention, AverageAttention
from macarico.actors.rnn import RNNActor
from macarico.actors.bow import BOWActor
from macarico.policies.linear import *

import macarico.tasks.sequence_labeler as sl

from macarico.annealing import stochastic, ExponentialAnnealing, NoAnnealing, Averaging, EWMA
from macarico.features.sequence import EmbeddingFeatures, BOWFeatures, RNN, DilatedCNN, AttendAt, FrontBackAttention, SoftmaxAttention, AverageAttention
from macarico.actors.rnn import RNNActor
from macarico.actors.bow import BOWActor
from macarico.policies.linear import *

import macarico.tasks.sequence_labeler as sl
import macarico.tasks.dependency_parser as dep
import matplotlib.pyplot as plt
from functools import partial

class PolicyEvaluator:
    def __init__(self, mk_env, data, losses):
        self.res = []
        self.epoch = 0
        self.mk_env = mk_env
        self.data = data
        self.losses = losses

    def evaluate_pos(self, policy, searn=False, verbose=True):
        "Compute average `loss()` of `policy` on `data`"
        if searn:
            policy.disable_ref = True
        was_list = True
        if not isinstance(self.losses, list):
            losses = [self.losses]
            was_list = False
        else:
            losses = self.losses

        for loss in losses:
            loss.reset()
        correct, tot = 0, 0
        with torch.no_grad():
            for example in self.data:
                if searn:
                    [x.new_minibatch() for x in policy.policies]
                else:
                    policy.new_minibatch()
                y_hat = self.mk_env(example).run_episode(policy)
                y = example.Y
                correct += sum([y1 == y0 for y1, y0 in zip(y_hat, y)])
                tot += len(y)
                # if verbose:
                #     print(example)
                for loss in losses:
                    loss(example)
            scores = [loss.get() for loss in losses]
        if not was_list:
            scores = scores[0]
        
        acc = correct/tot
        if verbose:
            print(f'EPOCH: {self.epoch} || POLICY LOSS: {scores} || POLICY ACC: {acc}')
        self.res.append((self.epoch, scores, acc))
        self.epoch += 1
        if searn:
            policy.disable_ref = False
        return scores, acc
    
     # Save the results to a CSV file
    def save_res(self, save_path, beta):
        version = 1
        base_path, ext = os.path.splitext(save_path)
        while os.path.isfile(f"{base_path}_v{version}{ext}"):
            version += 1

        save_path = f"{base_path}_v{version}{ext}"

        with open(save_path, 'w', newline='') as csvfile:
            fieldnames = ['iteration', 'beta', 'loss', 'accuracy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i, (epoch, loss, acc) in enumerate(self.res):
                writer.writerow({'iteration': epoch, 'beta': beta, 'loss': loss, 'accuracy': acc})

    # Save plots of accuracy and loss with iterations
    def save_plots(self, save_path, beta):
        iterations = [res[0] for res in self.res]
        losses = [res[1] for res in self.res]
        accuracies = [res[2] for res in self.res]

        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(iterations, losses)
        plt.title('Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)
        plt.plot(iterations, accuracies)
        plt.title('Accuracy')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')

        plt.tight_layout()

        version = 1
        base_path, ext = os.path.splitext(save_path)
        while os.path.isfile(f"{base_path}_v{version}{ext}"):
            version += 1

        plot_path = f"{base_path}_v{version}{ext}"
        plt.savefig(plot_path)
        plt.close()

        print(f"Plots saved to: {plot_path}")

class ShitReference(macarico.Reference):
    def __init__(self, oracle, n_labels, oracle_threshold):
        super().__init__()
        self.oracle = oracle
        self.n_labels = n_labels
        self.oracle_threshold = oracle_threshold

    def __call__(self, state):
        if np.random.random() < self.oracle_threshold:
            return self.oracle(state) # use oracle
        else:
            return np.random.choice(self.n_labels)

    def set_min_costs_to_go(self, state, cost_vector):
        cost_vector *= 0
        cost_vector += 1
        if np.random.random() < self.oracle_threshold:
            cost_vector[state.example.Y[state.n]] = 0. # use oracle
        else:
            cost_vector[np.random.choice(self.n_labels)] = 0.
            

def policy_generator(n_actions, n_hidden=32, n_types=5):
    features = RNN(EmbeddingFeatures(n_types, d_emb=n_hidden), d_rnn=n_hidden, cell_type='RNN')
    # actor = RNNActor([SoftmaxAttention(features)], n_actions)
    actor = RNNActor([AttendAt(features)], n_actions)
    policy = CSOAAPolicy(actor, n_actions)
    return policy

def optim_generator(p, lr=0.01):
    optimizer = torch.optim.Adam(p, lr)
    return optimizer

def train_searn(n_iterations=5, n_epochs=10, beta=0.9, n_hidden=64, n_data=1000, length=5, n_types=10, n_labels=5, oracle_threshold=0, save_folder='/experiments/'):
    # create the learner
    policy = policy_generator(n_labels, n_hidden=n_hidden, n_types=n_types)
    evaluator = PolicyEvaluator(mk_env, dev_data, sl.HammingLoss())
    evaluator.evaluate_pos(policy)

    p_rollin_ref = stochastic(ExponentialAnnealing(1 - beta))
    learner = SEARN(policy, ref, p_rollin_ref, weight=stochastic(ExponentialAnnealing(1 - beta)))

    # begin training iterations
    for iteration in range(n_iterations):
        # create new classifer, set policies on Learner object
        if iteration > 0:
            policy = policy_generator(n_labels, n_hidden=n_hidden, n_types=n_types)
            learner.update_policies(policy, stochastic(ExponentialAnnealing(1 - beta)))

        optimizer = optim_generator(policy.parameters())

        # train new classifer
        train_loop = util.TrainLoop(mk_env, policy, learner, optimizer,
                    losses = [loss_fn, loss_fn, loss_fn],
                    progress_bar = False,
                    policies=learner.policies)
        
        res = train_loop.train(tr_data,
                dev_data = dev_data,
                n_epochs = n_epochs)
        
        evaluator.evaluate_pos(learner, searn=True)

        if iteration != n_iterations - 1:
            learner.rollin_ref.step()
            learner.update_weights() # step each classifer weight down 

    # Save the results to a CSV file
    beta = int(10*beta)
    oracle_threshold = int(10*oracle_threshold)
    save_path_csv = os.getcwd() + save_folder + f'SEARN_iter_{n_iterations}_epoch_{n_epochs}_beta_{beta}_oracle_{oracle_threshold}.csv'
    save_path_plot = os.getcwd() + save_folder + 'plots/' + f'SEARN_iter_{n_iterations}_epoch_{n_epochs}_beta_{beta}_oracle_{oracle_threshold}.png'
    evaluator.save_res(save_path_csv, beta)
    evaluator.save_plots(save_path_plot, beta)

    try:
        test_evaluator = PolicyEvaluator(mk_env, test_data, sl.HammingLoss())
        test_evaluator.evaluate_pos(learner, searn=True)
        test_evaluator.save_res(os.getcwd() + save_folder + f'SEARN_TEST_epoch_{n_epochs}_beta_{beta}_oracle_{oracle_threshold}.csv', beta)
    except Exception as e:
        print(e)


def train_dagger(n_epochs=10, beta=0.9, n_hidden=64, n_data=1000, length=5, n_types=10, n_labels=5, oracle_threshold=0, save_folder='/experiments/'):
    policy = policy_generator(n_labels, n_hidden=n_hidden, n_types=n_types)
    optimizer = optim_generator(policy.parameters())
    evaluator = PolicyEvaluator(mk_env, dev_data, sl.HammingLoss())
    evaluator.evaluate_pos(policy)
    
    # create the learner
    p_rollin_ref = stochastic(ExponentialAnnealing(1 - beta))
    learner = DAgger(policy, ref, p_rollin_ref)

    train_loop = util.TrainLoop(mk_env, policy, learner, optimizer,
                losses = [loss_fn, loss_fn, loss_fn],
                progress_bar = False,
                run_per_epoch=[p_rollin_ref.step, partial(evaluator.evaluate_pos, policy)])
    res = train_loop.train(tr_data,
            dev_data = dev_data,
            n_epochs = n_epochs)
 
    # Save the results to a CSV file
    beta = int(10*beta)
    oracle_threshold = int(10*oracle_threshold)
    save_path_csv = os.getcwd() + save_folder + f'DAGGER_epoch_{n_epochs}_beta_{beta}_oracle_{oracle_threshold}.csv'
    save_path_plot = os.getcwd() + save_folder + 'plots/' + f'DAGGER_epoch_{n_epochs}_beta_{beta}_oracle_{oracle_threshold}.png'
    evaluator.save_res(save_path_csv, beta)
    evaluator.save_plots(save_path_plot, beta)

    try:
        test_evaluator = PolicyEvaluator(mk_env, test_data, sl.HammingLoss())
        test_evaluator.evaluate_pos(policy)
        test_evaluator.save_res(os.getcwd() + save_folder + f'DAGGER_TEST_epoch_{n_epochs}_beta_{beta}_oracle_{oracle_threshold}.csv', beta)
    except Exception as e:
        print(e)


def train_lols(n_epochs=10, beta=0.5, n_hidden=64, n_data=1000, length=5, n_types=10, n_labels=5, oracle_threshold=1, save_folder='/experiments/'):
    # mixture=MIX_PER_ROLL,
    policy = policy_generator(n_labels, n_hidden=n_hidden, n_types=n_types)
    optimizer = optim_generator(policy.parameters())
    evaluator = PolicyEvaluator(mk_env, dev_data, sl.HammingLoss())
    evaluator.evaluate_pos(policy)
    # create the learner
    p_rollin_ref=stochastic(NoAnnealing(0))
    p_rollout_ref=stochastic(ExponentialAnnealing(1 - beta))
    learner = LOLS(policy, ref, loss_fn, p_rollin_ref, p_rollout_ref)

    train_loop = util.TrainLoop(mk_env, policy, learner, optimizer,
                losses = [loss_fn, loss_fn, loss_fn],
                progress_bar = False,
                run_per_epoch=[p_rollin_ref.step, p_rollout_ref.step, partial(evaluator.evaluate_pos, policy)])
    res = train_loop.train(tr_data,
            dev_data = dev_data,
            n_epochs = n_epochs)

    # Save the results to a CSV file
    beta = int(10*beta)
    oracle_threshold = int(10*oracle_threshold)
    save_path_csv = os.getcwd() + save_folder + f'LOLS_epoch_{n_epochs}_beta_{beta}_oracle_{oracle_threshold}.csv'
    save_path_plot = os.getcwd() + save_folder + 'plots/' + f'LOLS_epoch_{n_epochs}_beta_{beta}_oracle_{oracle_threshold}.png'
    evaluator.save_res(save_path_csv, beta)
    evaluator.save_plots(save_path_plot, beta)

    try:
        test_evaluator = PolicyEvaluator(mk_env, test_data, sl.HammingLoss())
        test_evaluator.evaluate_pos(policy)
        test_evaluator.save_res(os.getcwd() + save_folder + f'LOLS_TEST_epoch_{n_epochs}_beta_{beta}_oracle_{oracle_threshold}.csv', beta)
    except Exception as e:
        print(e)


def train_aggrevate(n_epochs=10, beta=0.5, n_hidden=64, n_data=1000, length=5, n_types=10, n_labels=5, oracle_threshold=0, save_folder='/experiments/'):
    # create the learner
    policy = policy_generator(n_labels, n_hidden=n_hidden, n_types=n_types)
    optimizer = optim_generator(policy.parameters())
    evaluator = PolicyEvaluator(mk_env, dev_data, sl.HammingLoss())
    evaluator.evaluate_pos(policy)

    p_rollin_ref=stochastic(NoAnnealing(1 - beta))
    learner = AggreVaTe(policy, ref, p_rollin_ref)

    train_loop = util.TrainLoop(mk_env, policy, learner, optimizer,
                losses = [loss_fn, loss_fn, loss_fn],
                progress_bar = False,
                run_per_epoch=[p_rollin_ref.step, partial(evaluator.evaluate_pos, policy)])
    res = train_loop.train(tr_data,
            dev_data = dev_data,
            n_epochs = n_epochs)

    # Save the results to a CSV file
    beta = int(10*beta)
    oracle_threshold = int(10*oracle_threshold)
    save_path_csv = os.getcwd() + save_folder + f'AGGREVATE_epoch_{n_epochs}_beta_{beta}_oracle_{oracle_threshold}.csv'
    save_path_plot = os.getcwd() + save_folder + 'plots/' + f'AGGREVATE_epoch_{n_epochs}_beta_{beta}_oracle_{oracle_threshold}.png'
    evaluator.save_res(save_path_csv, beta)
    evaluator.save_plots(save_path_plot, beta)

    try:
        test_evaluator = PolicyEvaluator(mk_env, test_data, sl.HammingLoss())
        test_evaluator.evaluate_pos(policy)
        test_evaluator.save_res(os.getcwd() + save_folder + f'AGGREVATE_TEST_epoch_{n_epochs}_beta_{beta}_oracle_{oracle_threshold}.csv', beta)
    except Exception as e:
        print(e)



if __name__ == '__main__':
    # main()
    parser = argparse.ArgumentParser(description='Train a sequence labeling model using SEARN, DAGGER, LOLS, or AGGREVATE.')
    parser.add_argument('--learner', type=str, choices=['searn', 'dagger', 'lols', 'aggrevate'], required=True, help='Learner to use (searn, dagger, lols, or aggrevate)')
    parser.add_argument('--task', type=str, choices=['real', 'synthetic'], required=True, help='Data task to run (real or synthetic)')
    parser.add_argument('--n_iterations', type=int, default=5, help='Number of training iterations (for SEARN)')
    parser.add_argument('--beta', type=float, default=None, help='Rollout reference annealing rate')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs per iteration')
    parser.add_argument('--n_hidden', type=int, default=64, help='Number of hidden units in the RNN')
    parser.add_argument('--n_data', type=int, default=1000, help='Number of data points')
    parser.add_argument('--length', type=int, default=5, help='Length of the sequences')
    parser.add_argument('--n_types', type=int, default=10, help='Number of input types')
    parser.add_argument('--n_labels', type=int, default=5, help='Number of labels')
    parser.add_argument('--oracle_threshold', type=float, default=1, help='Oracle threshold for ShitReference')
    parser.add_argument('--save_folder', type=str, default='/experiments/', help='Folder to save the results')

    beta_default = {
        'synthetic': {
            'searn': 0.9,
            'dagger': 0.9,
            'aggrevate': 0.9,
            'lols': 0.5
        },
        'real': {
            'searn': 0.99,
            'dagger': 0.99,
            'aggrevate': 0.99,
            'lols': 0.5
        },
    }

    args = parser.parse_args()
    beta = args.beta if args.beta else beta_default[args.task][args.learner]

    # create data
    random.seed(42)
    torch.manual_seed(42)

    data_folder = './data'
    if args.task == 'synthetic':
        data_filename = f'sequence_reversal_data_types_{args.n_types}_labels_{args.n_labels}_n_{args.n_data}.pt'

        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        data_filepath = os.path.join(data_folder, data_filename)
        if os.path.exists(data_filepath):
            data = torch.load(data_filepath)
        else:
            data = synth.make_sequence_reversal_data(args.n_data, args.n_types, args.n_labels)
            torch.save(data, data_filepath)

        random.shuffle(data)
        tr_data = data[len(data)//2:]
        dev_data = data[:len(data)//2]
        test_data = []

        n_types, n_labels = args.n_types, args.n_labels
        save_folder = '/experiments/synthetic/'

    else: # task = real
        tok_vocab = torch.load('data/ud_token_vocab.pt')
        lab_vocab = torch.load('data/ud_label_vocab.pt')
        tr_data = torch.load('data/ud_train.pt')
        dev_data = torch.load('data/ud_dev.pt')
        test_data = torch.load('data/ud_test.pt')

        n_types, n_labels = len(tok_vocab), len(lab_vocab)
        save_folder = '/experiments/real/'

    # set up envrionment and reference
    mk_env = sl.SequenceLabeler
    loss_fn = sl.HammingLoss
    oracle = sl.HammingLossReference()
    ref = ShitReference(oracle, n_labels, args.oracle_threshold)

    # evaluate reference
    ref_loss, ref_acc = PolicyEvaluator(mk_env, dev_data, sl.HammingLoss()).evaluate_pos(ref, verbose=False)
    print(f'REFERENCE LOSS: {ref_loss} || REFERENCE ACC: {ref_acc}')

    if args.learner == 'searn':
        train_searn(
            n_iterations=args.n_iterations,
            n_epochs=args.n_epochs,
            beta=beta,
            n_hidden=args.n_hidden,
            n_data=args.n_data,
            length=args.length,
            n_types=n_types,
            n_labels=n_labels,
            oracle_threshold=args.oracle_threshold,
            save_folder=save_folder
        )
    elif args.learner == 'dagger':
        train_dagger(
            n_epochs=args.n_epochs,
            beta=beta,
            n_hidden=args.n_hidden,
            n_data=args.n_data,
            length=args.length,
            n_types=n_types,
            n_labels=n_labels,
            oracle_threshold=args.oracle_threshold,
            save_folder=save_folder
        )
    elif args.learner == 'lols':
        train_lols(
            n_epochs=args.n_epochs,
            beta=beta,
            n_hidden=args.n_hidden,
            n_data=args.n_data,
            length=args.length,
            n_types=n_types,
            n_labels=n_labels,
            oracle_threshold=args.oracle_threshold,
            save_folder=save_folder
        )
    elif args.learner == 'aggrevate':
        train_aggrevate(
            n_epochs=args.n_epochs,
            beta=beta,
            n_hidden=args.n_hidden,
            n_data=args.n_data,
            length=args.length,
            n_types=n_types,
            n_labels=n_labels,
            oracle_threshold=args.oracle_threshold,
            save_folder=save_folder
        )
    
