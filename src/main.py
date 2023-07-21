import argparse
import random
import os

import pickle
from tqdm import tqdm
import numpy as np

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from pysat.solvers import Solver

from models import NeuroMUSX, NeuroSAT, NeuroMUSX_V2
from loss import Loss_NeuroMUSX
import matplotlib.pyplot as plt

def getScores(batch, pred_mus, y_mus_cpu, y_sat_cpu, id, solver_instances, mask):
    score = 0
    for i in range(np.max(batch)+1):
        if y_sat_cpu[i] == 1:
            continue
        pred_mus_now = pred_mus[batch == i]
        mask_now = mask[batch == i]
        y_mus_cpu_now = y_mus_cpu[batch == i]

        threshold_list = pred_mus_now.tolist()
        threshold_list.sort()

        pred_mus_final = None
        solver = solver_instances[id[i]]
        for threshold in threshold_list:
            if threshold == 0:
                continue
            pred_mus_temp = np.where(pred_mus_now >= threshold, 1, 0)
            if pred_mus_final is None or isUnsat(pred_mus_temp, solver):
                pred_mus_final = pred_mus_temp
            else:
                break
        score += getScore(np.sum(pred_mus_final), np.sum(y_mus_cpu_now), np.sum(mask_now))
    return score

def getScore(size_pred, size_unsat_core, n_clause):
    if n_clause-size_unsat_core == 0:
        return 0
    return max((size_pred-size_unsat_core)/(n_clause-size_unsat_core), 0)

def isUnsat(pred, solver):
    assump = np.argwhere(pred==1) + 1
    if len(assump) > 1:
        assump = np.squeeze(assump)
    else:
        assump = assump[0]
    return not solver.solve(assumptions=(assump).tolist())

def getSolver(formula, n_var):
    solver = Solver(name='m22', bootstrap_with=formula.hard, use_timer=True)
    topv = n_var
    for i, cl in enumerate(formula.soft):
        topv += 1

        solver.add_clause(cl + [-topv])
    return solver

def saveAllPlots(train_loss, train_mus_correct, train_sat_correct, train_score, test_loss, test_mus_correct, test_sat_correct, test_score, args):
    x_axis = np.arange(0, len(test_score)) * args.test_epochs

    print("Plotting...")
    plt.plot(x_axis, train_score, label="Train")
    plt.plot(x_axis, test_score, label="Test", linestyle='dashed')
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()

    plt.savefig("../plots/score.png")
    plt.clf()

    plt.plot(train_mus_correct, label="Train")
    plt.plot(x_axis, test_mus_correct, label="Test", linestyle='dashed')
    plt.xlabel("Epochs")
    plt.ylabel("MUS")
    plt.legend()

    plt.savefig("../plots/mus.png")
    plt.clf()

    plt.plot(train_sat_correct, label="Train")
    plt.plot(x_axis, test_sat_correct, label="Test", linestyle='dashed')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig("../plots/sat.png")
    plt.clf()

    plt.plot(train_loss, label="Train")
    plt.plot(x_axis, test_loss, label="Test", linestyle='dashed')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig("../plots/loss.png")
    plt.clf()

def get_args():
    parser = argparse.ArgumentParser(description='NeuroMUSX: A GNN Based Minimal Unsatisfiable Subset Extractor')
    parser.add_argument('-b', '--batch-size', default=64, type=int, help='Batch Size to use')
    parser.add_argument('-e', '--epochs', default=1000, type=int, help='Number of epochs to train for')
    parser.add_argument('-te', '--test-epochs', default=50, type=int, help='Interval of epochs for testing')
    parser.add_argument('-ns', '--neuro-sat', default=False, type=bool, help='Uses the NeuroSAT framework for testing when true')
    parser.add_argument('-dtest', '--dataset-test', default='../dataset/processed/test.pkl', type=str, help='Dataset to use for testing')
    parser.add_argument('-dtrain', '--dataset-train', default='../dataset/processed/train.pkl', type=str, help='Dataset to use for training')
    parser.add_argument('-l', '--layers', default=10, type=int, help='Number of layers to use')
    parser.add_argument('-mse', '--model-save-epoch', default=50, type=int, help='Interval of epochs for saving model')
    parser.add_argument('-r', '--random', default=True, type=bool, help='Set to True if dataset is random')
    parser.add_argument('-mp', '--model-path', default='../models/final/model_random.pt', type=str, help='Path to save model to')
    parser.add_argument('-tl', '--transfer-learning', default="", type=str, help='Path to model to use for transfer learning')
    parser.add_argument('-pse', '--plot-save-epoch', default=50, type=int, help='Interval of epochs for saving plots')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.chdir("../src")
    random.seed(1)

    with open(args.dataset_train, "rb") as f:
        train_data = pickle.load(f)
    with open(args.dataset_test, "rb") as f:
        test_data = pickle.load(f)

    if args.random:
        train_data_unsat, train_data_sat = train_data
        test_data_unsat, test_data_sat = test_data
    else:
        train_data_unsat = train_data
        test_data_unsat = test_data
        train_data_sat = []
        test_data_sat = []
    
    train_data_loaded = []
    test_data_loaded = []

    solver_instances = []
    total_positive = 0
    total_negative = 0
    
    print("Loading train data...")

    id = 0
    for cnf_data in tqdm(train_data_unsat):
        if args.neuro_sat:
            cnf_data.setSplitLiterals(True)
        features, mask = cnf_data.getFeatures()

        data = Data(x=torch.tensor(features).float(), edge_index=torch.tensor(cnf_data.edge_index), mask=torch.tensor(mask),
                        edge_attr=torch.tensor(cnf_data.edge_attr).float(), y_mus=torch.tensor(cnf_data.mus_bin).float(),
                        y_sat=torch.tensor(cnf_data.sat).float(), id=id)
        
        solver_instances.append(getSolver(cnf_data.formula, cnf_data.n_vars))
        train_data_loaded.append(data)
        total_positive += np.sum(cnf_data.mus_bin)
        total_negative += cnf_data.n_clauses - np.sum(cnf_data.mus_bin)
        id += 1

    for cnf_data in tqdm(train_data_sat):
        if args.neuro_sat:
            cnf_data.setSplitLiterals(True)
        features, mask = cnf_data.getFeatures()

        data = Data(x=torch.tensor(features).float(), edge_index=torch.tensor(cnf_data.edge_index), mask=torch.tensor(mask),
                        edge_attr=torch.tensor(cnf_data.edge_attr).float(), y_mus=torch.tensor(cnf_data.mus_bin).float(),
                        y_sat=torch.tensor(cnf_data.sat).float(), id=id)
        
        solver_instances.append(getSolver(cnf_data.formula, cnf_data.n_vars))
        train_data_loaded.append(data)
        id += 1

    train_loader = DataLoader(train_data_loaded, batch_size=args.batch_size, shuffle=True)

    print("Finished!")


    print("Loading test data...")
    for cnf_data in tqdm(test_data_unsat):
        if args.neuro_sat:
            cnf_data.setSplitLiterals(True)
        features, mask = cnf_data.getFeatures()

        data = Data(x=torch.tensor(features).float(), edge_index=torch.tensor(cnf_data.edge_index), mask=torch.tensor(mask),
                        edge_attr=torch.tensor(cnf_data.edge_attr).float(), y_mus=torch.tensor(cnf_data.mus_bin).float(),
                        y_sat=torch.tensor(cnf_data.sat).float(), id=id)
        
        solver_instances.append(getSolver(cnf_data.formula, cnf_data.n_vars))
        test_data_loaded.append(data)
        id += 1
    
    for cnf_data in tqdm(test_data_sat):
        if args.neuro_sat:
            cnf_data.setSplitLiterals(True)
        features, mask = cnf_data.getFeatures()

        data = Data(x=torch.tensor(features).float(), edge_index=torch.tensor(cnf_data.edge_index), mask=torch.tensor(mask),
                        edge_attr=torch.tensor(cnf_data.edge_attr).float(), y_mus=torch.tensor(cnf_data.mus_bin).float(),
                        y_sat=torch.tensor(cnf_data.sat).float(), id=id)
        
        solver_instances.append(getSolver(cnf_data.formula, cnf_data.n_vars))
        test_data_loaded.append(data)
        id += 1
    
    test_loader = DataLoader(test_data_loaded, batch_size=args.batch_size, shuffle=False)

    print("Finished!")

    print("Loading model...")
    if args.neuro_sat:
        model = NeuroSAT(iterations=5)
        optim = torch.optim.Adam(model.parameters(), lr=0.00002)
    else:
        if args.transfer_learning != "":
            model_loaded = torch.load(args.transfer_learning)
            model = NeuroMUSX_V2(model_loaded[1])
            model.load_state_dict(model_loaded[0])
        else:
            model = NeuroMUSX_V2(args.layers)
        optim = torch.optim.Adam(model.parameters(), lr=0.0001)
        loss_func = Loss_NeuroMUSX(torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(total_negative/total_positive)), 
                                   torch.nn.BCEWithLogitsLoss())
    
    print("Finished!")

    print("Training...")
    

    train_loss = []
    train_mus_correct = []
    train_sat_correct = []
    train_score = []

    test_loss = []
    test_mus_correct = []
    test_sat_correct = []
    test_score = []

    model.to(device)

    for epoch in range(1, args.epochs+1):
        model.train()
        sum_loss = 0
        sum_mus_correct = 0
        sum_sat_correct = 0
        sum_score = 0

        for data in tqdm(train_loader):
            optim.zero_grad()

            data = data.to(device)

            mask = data.mask.detach().cpu().numpy()
            batch = data.batch.detach().cpu().numpy()
            id_cpu = data.id.detach().cpu().numpy()

            y_mus_cpu = data.y_mus.detach().cpu().numpy()
            y_sat_cpu = data.y_sat.detach().cpu().numpy()

            out = model(data, batch)
            
            pred_mus = torch.sigmoid(out[0]).detach().cpu().numpy() * mask
            pred_sat = torch.sigmoid(out[1]).detach().cpu().numpy()
            pred_sat = np.round(pred_sat)

            if args.neuro_sat:
                continue
            else:
                loss = loss_func(out[0], data.y_mus, out[1], data.y_sat, batch, mask)
                sum_loss += float(loss)
                loss.backward()
                optim.step()

            sum_mus_correct += (np.sum(mask) - np.count_nonzero(np.round(pred_mus) - y_mus_cpu))/np.sum(mask)*len(data)
            sum_sat_correct += len(data) - np.count_nonzero(pred_sat - y_sat_cpu)
            if epoch % args.test_epochs == 0:
                sum_score += getScores(batch, pred_mus, y_mus_cpu, y_sat_cpu, id_cpu, solver_instances, mask)
        sum_sat_correct /= len(train_data_loaded)
        sum_score /= len(train_data_loaded)
        if args.random:
            sum_score *= 2
        sum_mus_correct /= len(train_data_loaded)
        sum_loss /= len(train_data_loaded)

        train_loss.append(sum_loss)
        train_mus_correct.append(sum_mus_correct)
        train_sat_correct.append(sum_sat_correct)

        if epoch % args.test_epochs == 0:
            sum_loss /= len(train_data_loaded)
            train_score.append(sum_score)
        
        print("Epoch: {} Loss: {} MUS: {} SAT: {} Score: {}".format(epoch, sum_loss, sum_mus_correct, sum_sat_correct, sum_score))
        if epoch % args.model_save_epoch == 0:
            torch.save([model.state_dict(), model.iterations], "../models/per_epoch/model_{}.pt".format(epoch))
        if epoch % args.test_epochs == 0:
            model.eval()
            with torch.no_grad():
                sum_loss = 0
                sum_mus_correct = 0
                sum_sat_correct = 0
                sum_score = 0

                for data in tqdm(test_loader):
                    data = data.to(device)

                    mask = data.mask.detach().cpu().numpy()
                    batch = data.batch.detach().cpu().numpy()
                    id_cpu = data.id.detach().cpu().numpy()

                    y_mus_cpu = data.y_mus.detach().cpu().numpy()
                    y_sat_cpu = data.y_sat.detach().cpu().numpy()

                    out = model(data, batch)
                    
                    pred_mus = torch.sigmoid(out[0]).detach().cpu().numpy() * mask
                    pred_sat = torch.sigmoid(out[1]).detach().cpu().numpy()
                    pred_sat = np.round(pred_sat)

                    if args.neuro_sat:
                        continue
                    else:
                        loss = loss_func(out[0], data.y_mus, out[1], data.y_sat, batch, mask)
                        sum_loss += float(loss)

                    sum_mus_correct += (np.sum(mask) - np.count_nonzero(np.round(pred_mus) - y_mus_cpu))/np.sum(mask)*len(data)
                    sum_sat_correct += len(data) - np.count_nonzero(pred_sat - y_sat_cpu)
                    sum_score += getScores(batch, pred_mus, y_mus_cpu, y_sat_cpu, id_cpu, solver_instances, mask)

                sum_sat_correct /= len(test_data_loaded)
                sum_score /= len(test_data_loaded)
                if args.random:
                    sum_score *= 2
                sum_loss /= len(test_data_loaded)
                sum_mus_correct /= len(test_data_loaded)

                test_loss.append(sum_loss)
                test_mus_correct.append(sum_mus_correct)
                test_sat_correct.append(sum_sat_correct)
                test_score.append(sum_score)
                if epoch == args.test_epochs:
                    test_loss.append(sum_loss)
                    test_mus_correct.append(sum_mus_correct)
                    test_sat_correct.append(sum_sat_correct)
                    test_score.append(sum_score)
                    train_score.append(train_score[0])

                print("Test Loss: {} MUS: {} SAT: {} Score: {}".format(sum_loss, sum_mus_correct, sum_sat_correct, sum_score))
        
        if epoch % args.plot_save_epoch == 0 and epoch >= args.test_epochs:
            saveAllPlots(train_loss, train_mus_correct, train_sat_correct, train_score, test_loss, test_mus_correct, test_sat_correct, test_score, args)

    print("Finished!")
    torch.save([model.state_dict(), model.iterations], args.model_path)
    saveAllPlots(train_loss, train_mus_correct, train_sat_correct, train_score, test_loss, test_mus_correct, test_sat_correct, test_score, args)
    
