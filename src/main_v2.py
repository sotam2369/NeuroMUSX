import torch
import pickle
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn
import os
import sys

import model
import random
import lossfunc as LF

def getCorrectUnsat(batch, pred, data, get_score=False):
    correct = 0
    allcorrect = 0
    score = 0
    for i in range(np.max(batch)+1):
        if data.cnf_data[i].unsat_bit == 0:
            continue
        pred_now = pred[batch == i]
        """
        if data.cnf_data[i].isUnsatCoreKissat(pred_now):
            if data.cnf_data[i].isMUS(pred_now):
                correct += 1
            allcorrect += 1
            """
        if np.sum(pred_now) == data.cnf_data[i].n_clause:
            allcorrect += 1
            score += getScore(np.sum(pred_now), np.sum(data.cnf_data[i].unsat_cores[0]), data.cnf_data[i].n_clause)
        elif data.cnf_data[i].isUnsatAssump(pred_now):
            correct += 1
            allcorrect += 1
            score += getScore(np.sum(pred_now), np.sum(data.cnf_data[i].unsat_cores[0]), data.cnf_data[i].n_clause)
        else:
            score += 1
    if get_score:
        return correct, allcorrect, score
    return correct, allcorrect

def getScores(batch, pred, data):
    score = 0
    for i in range(np.max(batch)+1):
        if data.cnf_data[i].unsat_bit == 0:
            continue
        pred_now = pred[batch == i]
        thres_list = pred_now.tolist()
        thres_list.sort()

        pred_thres = None
        for t in thres_list:
            if t == 0:
                continue
            pred_temp = np.where(pred_now >= t, 1, 0)
            if pred_thres is None or data.cnf_data[i].isUnsatAssump(pred_temp):
                pred_thres = pred_temp
            else:
                break
        score += getScore(np.sum(pred_thres), np.sum(data.cnf_data[i].unsat_cores[0]), data.cnf_data[i].n_clause)
        #print(getScore(np.sum(pred_thres), np.sum(data.cnf_data[i].unsat_cores[0]), data.cnf_data[i].n_clause))
        #print(np.sum(pred_thres), np.sum(data.cnf_data[i].unsat_cores[0]), data.cnf_data[i].n_clause)
    return score


def getScore(size_pred, size_unsat_core, n_clause):
    if n_clause-size_unsat_core == 0:
        return 0
    return max((size_pred-size_unsat_core)/(n_clause-size_unsat_core), 0)

def getCorrectUnsatBit(batch, pred, unsat_bit):
    correct = [0,0]
    for i in range(np.max(batch)+1):
        #print(pred.shape)
        pred_now = np.round(np.mean(pred[batch == i]))
        if pred_now == unsat_bit[i]:
            correct[int(pred_now.item())] = correct[int(pred_now.item())] + 1
    return correct

def divideAll(arr, div):
    for i in range(len(arr)):
        arr[i] = float(arr[i])/float(div)
    return arr

def getY(batch, pred, data):
    y = data.cnf_data[0].getClosestCore(pred[batch == 0])
    for i in range(1, np.max(batch)+1):
        y = np.concatenate((y, data.cnf_data[i].getClosestCore(pred[batch == i])))
    return y

def getCorrect(batch, pred, y_true):
    correct = 0
    for i in range(np.max(batch)+1):
        pred_now = pred[batch == i]
        y_true_now = y_true[batch == i]
        if (pred_now == y_true_now).all():
            correct += 1
    return correct

def getF1Score(conf_mat):
    if conf_mat[0][0] == 0 and conf_mat[0][1] == 0:
        precision = 0
    else:
        precision = conf_mat[0][0]/(conf_mat[0][0]+conf_mat[0][1])
    if conf_mat[0][0] == 0 and conf_mat[1][0] == 0:
        recall = 0
    else:
        recall = conf_mat[0][0]/(conf_mat[0][0]+conf_mat[1][0])
    if recall == 0 and precision == 0:
        return 0
    f1_score = 2*precision*recall/(precision+recall)
    return f1_score


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode_type = int(input("1. Normal \n2. Test \n3. Single \n4. Sat Comp\nMode: "))

    data_train = []
    data_test = []
    comp_type = None
    model_path = None
    random.seed(1)

    BATCH_SIZE = 64
    EPOCHS = 300
    TEST_EPOCHS = 50
    ITERATIONS = 5
    LOSS_WEIGHT_OFFSET = 1
    MODEL_SAVE_EPOCHS = 1000
    USE_NESY = False
    NEURO_SAT = False
    if mode_type == 4:
        BIT_SUPERVIZ = False
        USE_SCORE = False
        BYPASS_SCORE = True
    else:
        BIT_SUPERVIZ = True
        USE_SCORE = True
        BYPASS_SCORE = False

    total_positive = 0.0
    total_negative = 0.0
    total_always_negative_train = 0.0
    total_always_negative_test = 0.0
    
    if mode_type == 1 or mode_type == 2 or mode_type == 4:
        if mode_type == 1 or mode_type == 4:

            unsat_cores_train = []

            print("Loading training data...")
            if mode_type == 1:
                with open('extracted_cores/unsat_cores_train_1-40full_MMUS.pickle', 'rb') as handle:
                    unsat_cores_train = pickle.load(handle)
                with open('extracted_cores/unsat_cores_train_40_MMUS_sat.pickle', 'rb') as handle:
                    unsat_cores_train.extend(pickle.load(handle))
            else:
                comp_type = input("Test type: ")
                model_path = input("Model name: ")
                if model_path == "-1":
                    model_path = None
                else:
                    model_path = "model_" + model_path
                i = 1
                
                while os.path.exists("extracted_cores/unsat_cores_satcomp(" + comp_type + "-" + str(i) + ").pickle"):
                    #if i == 5:
                    #    i += 1
                    #    continue
                    with open("extracted_cores/unsat_cores_satcomp(" + comp_type + "-" + str(i) + ").pickle", 'rb') as handle:
                        unsat_cores_train.append(pickle.load(handle)[0])
                    i += 1
                    break
                with open('extracted_cores/unsat_cores_train_1-40full_MMUS.pickle', 'rb') as handle:
                    unsat_cores_train.extend(random.choices(pickle.load(handle), k=4))

            for cnf_data in tqdm(unsat_cores_train):
                cnf_data.edge_feature = [[[1,0],[1,0]],[[0,1],[0,1]]]
                cnf_data.loadDimacs(neuro_sat=NEURO_SAT, loadSolver=(not NEURO_SAT))
                #print(cnf_data.prob_file, np.sum(cnf_data.unsat_cores[0]))
                features, mask = cnf_data.getFeatures(neuro_sat=NEURO_SAT)
                if BIT_SUPERVIZ:
                    unsat_bit = 0
                    if np.sum(cnf_data.unsat_cores[0]) > 0:
                        unsat_bit = 1
                    cnf_data.unsat_bit = unsat_bit
                    if NEURO_SAT:
                        data = Data(x=features, edge_index=torch.tensor(cnf_data.out_graph), mask=torch.tensor(mask).float(), \
                                    cnf_data=cnf_data, vars=cnf_data.getVars(), unsat_bit = torch.tensor(unsat_bit).float())
                    else:
                        data = Data(x=features, edge_index=torch.tensor(cnf_data.out_graph), mask=torch.tensor(mask), \
                                        edge_attr = torch.tensor(cnf_data.edge_attr).float(),cnf_data=cnf_data, vars=cnf_data.getVars(), unsat_bit = torch.tensor(unsat_bit).float())
                elif USE_NESY:
                    data = Data(x=features, edge_index=torch.tensor(cnf_data.out_graph), mask=torch.tensor(mask), \
                                    edge_attr = torch.tensor(cnf_data.edge_attr).float(),cnf_data=cnf_data, vars=cnf_data.getVars())
                else:
                    data = Data(x=features, edge_index=torch.tensor(cnf_data.out_graph), mask=torch.tensor(mask), \
                                    edge_attr = torch.tensor(cnf_data.edge_attr).float(),cnf_data=cnf_data)
                data_train.append(data)

                total_always_negative_train += cnf_data.n_var

                positive = np.sum(cnf_data.unsat_cores[0]).item()
                total_positive += positive
                total_negative += cnf_data.n_clause - positive
            
            train_loader = DataLoader(
                data_train,
                batch_size=BATCH_SIZE,
                shuffle=True
            )
        else:
            comp_type = input("Test type: ")
            model_path = "model_" + input("Model name: ")

        unsat_cores_test = {}
        #model_path = "model_3-40_satv2-3"

        print("Loading testing data...")
        test_path = "unsat_cores_test_1-40full_MMUS.pickle"
        if comp_type and comp_type != "-1":
            test_path = "unsat_cores_satcomp(" + comp_type + "-1).pickle"

        with open('extracted_cores/' + test_path, 'rb') as handle:
            unsat_cores_test = pickle.load(handle)

        if not comp_type or comp_type == "-1":
            with open('extracted_cores/unsat_cores_test_40_MMUS_sat.pickle', 'rb') as handle:
                unsat_cores_test.extend(pickle.load(handle))

        for cnf_data in tqdm(unsat_cores_test):

            cnf_data.edge_feature = [[[1,0],[1,0]],[[0,1],[0,1]]]
            cnf_data.loadDimacs(neuro_sat=NEURO_SAT, loadSolver=(not NEURO_SAT))
            features, mask = cnf_data.getFeatures(neuro_sat=NEURO_SAT)
            if BIT_SUPERVIZ:
                unsat_bit = 0
                if np.sum(cnf_data.unsat_cores[0]) > 0:
                    unsat_bit = 1
                cnf_data.unsat_bit = unsat_bit
                if NEURO_SAT:
                    data = Data(x=features, edge_index=torch.tensor(cnf_data.out_graph), mask=torch.tensor(mask).float(), \
                                cnf_data=cnf_data, vars=cnf_data.getVars(), unsat_bit = torch.tensor(unsat_bit).float())
                else:
                    data = Data(x=features, edge_index=torch.tensor(cnf_data.out_graph), mask=torch.tensor(mask), \
                                    edge_attr = torch.tensor(cnf_data.edge_attr).float(), cnf_data=cnf_data, vars=cnf_data.getVars(), unsat_bit = torch.tensor(unsat_bit).float())
            elif USE_NESY:
                data = Data(x=features, edge_index=torch.tensor(cnf_data.out_graph), mask=torch.tensor(mask), \
                                edge_attr = torch.tensor(cnf_data.edge_attr).float(),cnf_data=cnf_data, vars=cnf_data.getVars())
            else:
                data = Data(x=features, edge_index=torch.tensor(cnf_data.out_graph), mask=torch.tensor(mask), \
                                edge_attr = torch.tensor(cnf_data.edge_attr).float(),cnf_data=cnf_data)
            #print(data.x.size(), data.edge_index.size(), data.edge_attr.size(), data.mask.size())
            data_test.append(data)
            
            total_always_negative_test += cnf_data.n_var
            
            positive = np.sum(cnf_data.unsat_cores[0]).item()
            total_positive += positive
            total_negative += cnf_data.n_clause - positive

        test_loader = DataLoader(
            data_test,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
    else:
        unsat_cores = {}
        with open('extracted_cores/unsat_cores_test_single.pickle', 'rb') as handle:
            unsat_cores = pickle.load(handle)


        cnf_data = unsat_cores['test.dimacs']

        features, mask = cnf_data.getFeatures()
        data = Data(x=torch.tensor(features).float(), edge_index=torch.tensor(cnf_data.out_graph),\
                    y=torch.tensor(cnf_data.unsat_core), mask=torch.tensor(mask), \
                        edge_attr = torch.tensor(cnf_data.edge_attr).float(), cnf_data=cnf_data)
        data_train.append(data)
        data_test.append(data)

        positive = torch.sum(data.y).item()
        total_positive += positive
        total_negative += cnf_data.n_clause - positive
        total_always_negative_test += cnf_data.n_var
        total_always_negative_train += cnf_data.n_var

        train_loader = DataLoader(
            [data],
            batch_size=1,
            shuffle=False
        )

        test_loader = DataLoader(
            [data],
            batch_size=1,
            shuffle=False
        )
    
    if not NEURO_SAT:
        gnn_model = model.GNNSat_V2_3(iterations=ITERATIONS, use_mask=False, bit_supervision=BIT_SUPERVIZ)
    else:
        gnn_model = model.NeuroSAT(iterations=ITERATIONS)

    if mode_type == 2:
        model_loaded = torch.load("models/" + model_path + ".pt")
        gnn_model.load_state_dict(model_loaded[0])
        gnn_model.iterations = model_loaded[1]
        gnn_model.to(device)
        print("Testing sequence...")
        gnn_model.eval()

        sum_unsat_correct = 0
        sum_unsatall_correct = 0
        sum_correct = 0
        conf_mat_total = np.asarray([[-total_always_negative_test,0],[0,0]])

        for data in tqdm(test_loader):
            data = data.to(device)

            out = gnn_model(data)

            pred = torch.sigmoid(out).detach().to('cpu').numpy()
            pred = np.where(pred == 0.5, 0, pred)
            pred = np.where(pred > 0.5, 1, 0)

            y_cpu = getY(data.batch.detach().to('cpu').numpy(), pred, data)
            y = torch.tensor(y_cpu).float().to(device)

            sum_correct += getCorrect(data.batch.detach().to('cpu').numpy(), pred ,y_cpu)
            temp = getCorrectUnsat(data.batch.detach().to('cpu').numpy(), pred, data)
            sum_unsat_correct += temp[0]
            sum_unsatall_correct += temp[1]
            conf_mat_total += confusion_matrix(y_cpu, pred)

        f1_score = getF1Score(conf_mat_total)
        print("\n\n   Test Correct: ", sum_correct)
        print("   Test Unsat Correct: ", sum_unsat_correct)
        print("   Test Unsat All Correct: ", sum_unsatall_correct)
        
        print("   Confusion Matrix:\n")
        print("    ", "%8d %8d" % (int(conf_mat_total[0][0]), int(conf_mat_total[0][1])))
        print("    ", "%8d %8d" % (int(conf_mat_total[1][0]), int(conf_mat_total[1][1])))
        print("   F1 Score:", f1_score)
        print("\n")

        gnn_model.eval()
        final_test = Batch.from_data_list([data_test[0]])

        data = final_test.to(device)

        out = gnn_model(data, print_data=False)

        pred = torch.sigmoid(out).detach().to('cpu').numpy()
        pred = np.where(pred > 0.5, 1, 0)


        #print(np.count_nonzero(pred), np.count_nonzero(pred - 1))
        #print(data_test[0].cnf_data.prob_file)

    else:
        if model_path:
            model_loaded = torch.load("models_archive/" + model_path + ".pt")
            gnn_model.load_state_dict(model_loaded[0])
            gnn_model.iterations = model_loaded[1]
            print("Loading model...")
        if NEURO_SAT:
            optim = torch.optim.Adam(gnn_model.parameters(), lr=0.00002)
        else:
            optim = torch.optim.Adam(gnn_model.parameters())
        lossfunc = LF.LossV2(False, torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(total_negative/total_positive)*LOSS_WEIGHT_OFFSET), bit_supervision=BIT_SUPERVIZ, neuro_sat=NEURO_SAT)

        test_f = []
        test_correct = []
        test_unsat_correct = []
        test_unsatall_correct = []
        test_loss = []
        test_bit_0_correct = []
        test_bit_1_correct = []
        test_score = []

        train_f = []
        train_correct = []
        train_unsat_correct = []
        train_unsatall_correct = []
        train_loss = []
        train_bit_1_correct = []
        train_bit_0_correct = []
        train_score = []

        gnn_model.to(device)
        if BIT_SUPERVIZ and not NEURO_SAT:
            gnn_model.sbs[0].to(device)

        print("Starting training sequence...")
        for epoch in range(1, EPOCHS+1):
            # Train
            gnn_model.train()
            if BIT_SUPERVIZ and not NEURO_SAT:
                gnn_model.sbs[0].train()

            sum_loss = 0
            sum_correct = 0
            sum_unsat_correct = 0
            sum_unsatall_correct = 0
            sum_bit_correct = [0,0]
            sum_score = 0
            conf_mat_total = np.asarray([[-total_always_negative_train,0],[0,0]])

            for data in tqdm(train_loader):
                optim.zero_grad()
                if BIT_SUPERVIZ and not NEURO_SAT:
                    gnn_model.sbs_opt.zero_grad()

                data = data.to(device)

                out = gnn_model(data)
                if NEURO_SAT:
                    pred = torch.sigmoid(out).detach().to('cpu').numpy()
                    temp = getCorrectUnsatBit(data.batch.detach().to('cpu').numpy(), pred, data.unsat_bit.detach().to('cpu').numpy())
                    sum_bit_correct[0] += temp[0]
                    sum_bit_correct[1] += temp[1]
                elif BIT_SUPERVIZ:
                    pred = torch.sigmoid(out[0]*data.mask).detach().to('cpu').numpy()
                    pred_bit = torch.sigmoid(out[1]).detach().to('cpu').numpy()
                    temp = getCorrectUnsatBit(data.batch.detach().to('cpu').numpy(), pred_bit, data.unsat_bit.detach().to('cpu').numpy())
                    sum_bit_correct[0] += temp[0]
                    sum_bit_correct[1] += temp[1]
                else:
                    pred = torch.sigmoid(out*data.mask).detach().to('cpu').numpy()
                pred = np.where(pred == 0.5, 0, pred)
                pred = np.where(pred > 0.5, 1, 0)

                y_cpu = getY(data.batch.detach().to('cpu').numpy(), pred, data)
                y = torch.tensor(y_cpu).float().to(device)
                if NEURO_SAT:
                    loss = lossfunc(out, y, data.batch.detach().to('cpu').numpy(), data.cnf_data, data.mask.detach().to('cpu').numpy(), bit=out, bit_true=data.unsat_bit)[0]
                elif BIT_SUPERVIZ:
                    loss_temp = lossfunc(out[0], y, data.batch.detach().to('cpu').numpy(), data.cnf_data, data.mask.detach().to('cpu').numpy(), bit=out[1], bit_true=data.unsat_bit)
                    loss = loss_temp[0]

                    if len(loss_temp) > 1:
                        loss_temp[1].backward()
                        gnn_model.sbs_opt.step()
                        
                else:
                    loss = lossfunc(out, y, data.batch.detach().to('cpu').numpy(), data.cnf_data, data.mask.detach().to('cpu').numpy())

                loss.backward()
                optim.step()

                if BIT_SUPERVIZ and not NEURO_SAT:
                    gnn_model.sbs_opt.step()

                sum_loss += float(loss)
                if not NEURO_SAT:
                    sum_correct += getCorrect(data.batch.detach().to('cpu').numpy(), pred, y_cpu)
                    if epoch % TEST_EPOCHS == 0 and not BYPASS_SCORE:
                        if USE_SCORE:
                            temp = getCorrectUnsat(data.batch.detach().to('cpu').numpy(), pred, data)
                            sum_unsat_correct += temp[0]
                            sum_unsatall_correct += temp[1]
                            sum_score += getScores(data.batch.detach().to('cpu').numpy(), (torch.sigmoid(out[0])*data.mask).detach().to('cpu').numpy(), data)
                        else:
                            temp = getCorrectUnsat(data.batch.detach().to('cpu').numpy(), pred, data)
                            sum_unsat_correct += temp[0]
                            sum_unsatall_correct += temp[1]
                    conf_mat_total += confusion_matrix(y_cpu, pred)

            sum_loss /= len(data_train)
            sum_score /= len(data_train)
            sum_score *= 2
            f1_score = getF1Score(conf_mat_total)
            print("\n\nEpoch:", epoch)
            print("   Train Loss: ", sum_loss )
            print("   Train Fully Correct: ", sum_correct)
            if BIT_SUPERVIZ:
                print("   Train Bit Correct (0): ", sum_bit_correct[0])
                print("   Train Bit Correct (1): ", sum_bit_correct[1])

            if epoch % TEST_EPOCHS == 0:
                print("   Train Unsat Correct: ", sum_unsat_correct)
                print("   Train Unsat All Correct: ", sum_unsatall_correct)
                print("   Train Score: ", sum_score)
            print("   Confusion Matrix:")
            print("    ", "%8d %8d" % (int(conf_mat_total[0][0]), int(conf_mat_total[0][1])))
            print("    ", "%8d %8d" % (int(conf_mat_total[1][0]), int(conf_mat_total[1][1])))
            print("   F1 Score:", f1_score)
            print("\n")

            train_f.append(f1_score)
            train_correct.append(sum_correct)
            train_loss.append(sum_loss)
            if BIT_SUPERVIZ:
                train_bit_0_correct.append(sum_bit_correct[0])
                train_bit_1_correct.append(sum_bit_correct[1])
            if epoch % TEST_EPOCHS == 0:
                train_unsat_correct.append(sum_unsat_correct)
                train_unsatall_correct.append(sum_unsatall_correct)
                if USE_SCORE:
                    train_score.append(sum_score)
            #if epoch >= 10 and mode_type == 4 and f1_score > 0.95:
            #    print("Training finished.")
            #    torch.save([gnn_model.state_dict(), gnn_model.iterations], "models/model_" + str(epoch) + "(final).pt")
            #    exit()


            # Testing
            if epoch % TEST_EPOCHS == 0:
                print("Testing sequence...")
                gnn_model.eval()
                if BIT_SUPERVIZ and not NEURO_SAT:
                    gnn_model.sbs[0].eval()

                sum_unsat_correct = 0
                sum_unsatall_correct = 0
                sum_correct = 0
                sum_loss = 0
                sum_bit_correct = [0,0]
                sum_score = 0
                conf_mat_total = np.asarray([[-total_always_negative_test,0],[0,0]])

                for data in tqdm(test_loader):
                    data = data.to(device)

                    out = gnn_model(data)
                    
                    if NEURO_SAT:
                        pred = torch.sigmoid(out).detach().to('cpu').numpy()
                        temp = getCorrectUnsatBit(data.batch.detach().to('cpu').numpy(), pred, data.unsat_bit.detach().to('cpu').numpy())
                        sum_bit_correct[0] += temp[0]
                        sum_bit_correct[1] += temp[1]
                    elif BIT_SUPERVIZ:
                        pred = torch.sigmoid(out[0]*data.mask).detach().to('cpu').numpy()
                        pred_bit = torch.sigmoid(out[1]).detach().to('cpu').numpy()
                        temp = getCorrectUnsatBit(data.batch.detach().to('cpu').numpy(), pred_bit, data.unsat_bit.detach().to('cpu').numpy())
                        sum_bit_correct[0] += temp[0]
                        sum_bit_correct[1] += temp[1]
                    else:
                        pred = torch.sigmoid(out*data.mask).detach().to('cpu').numpy()
                    pred = np.where(pred == 0.5, 0, pred)
                    pred = np.where(pred > 0.5, 1, 0)

                    y_cpu = getY(data.batch.detach().to('cpu').numpy(), pred, data)
                    y = torch.tensor(y_cpu).float().to(device)

                    if NEURO_SAT:
                        loss = lossfunc(out, y, data.batch.detach().to('cpu').numpy(), data.cnf_data, data.mask.detach().to('cpu').numpy(), bit=out, bit_true=data.unsat_bit)[0]
                    elif BIT_SUPERVIZ:
                        loss = lossfunc(out[0], y, data.batch.detach().to('cpu').numpy(), data.cnf_data, data.mask.detach().to('cpu').numpy(), bit=out[1], bit_true=data.unsat_bit)[0]
                    else:
                        loss = lossfunc(out, y, data.batch.detach().to('cpu').numpy(), data.cnf_data, data.mask.detach().to('cpu').numpy())

                    sum_loss += float(loss)

                    if not NEURO_SAT and not BYPASS_SCORE:
                        sum_correct += getCorrect(data.batch.detach().to('cpu').numpy(), pred ,y_cpu)
                        if USE_SCORE:
                            temp = getCorrectUnsat(data.batch.detach().to('cpu').numpy(), pred, data)
                            sum_unsat_correct += temp[0]
                            sum_unsatall_correct += temp[1]
                            sum_score += getScores(data.batch.detach().to('cpu').numpy(), (torch.sigmoid(out[0])*data.mask).detach().to('cpu').numpy(), data)
                        else:
                            temp = getCorrectUnsat(data.batch.detach().to('cpu').numpy(), pred, data)
                            sum_unsat_correct += temp[0]
                            sum_unsatall_correct += temp[1]
                        conf_mat_total += confusion_matrix(y_cpu, pred)

                sum_loss /= len(data_train)
                sum_score /= len(data_train)
                sum_score *= 2
                f1_score = getF1Score(conf_mat_total)
                print("\n\n   Test Loss: ", sum_loss )
                print("   Test Correct: ", sum_correct)
                if BIT_SUPERVIZ:
                    print("   Test Bit Correct: ", sum_bit_correct)
                print("   Test Unsat Correct: ", sum_unsat_correct)
                print("   Test Unsat All Correct: ", sum_unsatall_correct)
                print("   Test Score: ", sum_score)
                
                print("   Confusion Matrix:\n")
                print("    ", "%8d %8d" % (int(conf_mat_total[0][0]), int(conf_mat_total[0][1])))
                print("    ", "%8d %8d" % (int(conf_mat_total[1][0]), int(conf_mat_total[1][1])))
                print("   F1 Score:", f1_score)
                print("\n")
                
                test_f.append(f1_score)
                test_correct.append(sum_correct)
                test_unsat_correct.append(sum_unsat_correct)
                test_unsatall_correct.append(sum_unsatall_correct)
                test_loss.append(sum_loss)
                if BIT_SUPERVIZ:
                    test_bit_0_correct.append(sum_bit_correct[0])
                    test_bit_1_correct.append(sum_bit_correct[1])
                if USE_SCORE:
                    test_score.append(sum_score)

                #if sum_acc > 0.7 and sum_acc < 0.75:
                #    break
            if epoch % MODEL_SAVE_EPOCHS == 0:
                torch.save([gnn_model.state_dict(), gnn_model.iterations], "models/model_" + str(epoch) + ".pt")
        
        """
        gnn_model.eval()
        final_test = Batch.from_data_list([data_train[0]])

        data = final_test.to(device)

        out = gnn_model(data, print_data=False)

        pred = torch.sigmoid(out).detach().to('cpu').numpy()
        pred = np.where(pred > 0.5, 1, 0)


        print(np.count_nonzero(pred), np.count_nonzero(pred - 1))
        data.cnf_data[0].getClosestCore(pred, print_data=False)
        """


        if TEST_EPOCHS > 1:
            test_f.insert(0, test_f[0])
            test_loss.insert(0, test_loss[0])
            test_correct.insert(0, test_correct[0])
            if not BYPASS_SCORE:
                test_unsat_correct.insert(0, test_unsat_correct[0])
                train_unsat_correct.insert(0, train_unsat_correct[0])
                test_unsatall_correct.insert(0, test_unsatall_correct[0])
                train_unsatall_correct.insert(0, train_unsatall_correct[0])
                test_score.insert(0, test_score[0])
                train_score.insert(0, train_score[0])
                test_bit_0_correct.insert(0, test_bit_0_correct[0])
                test_bit_1_correct.insert(0, test_bit_1_correct[0])

        x_axis = np.arange(0, len(test_f)) * TEST_EPOCHS


        plt.plot(train_f, label="Train F1 Score")
        plt.plot(x_axis, test_f, label="Test F1 Score (" + str(TEST_EPOCHS) + " epochs)", linestyle='dashed')
        plt.xlabel("Epochs")
        plt.ylabel("F1 Score")
        plt.legend()

        # Save plot
        plt.savefig("output_images/plot_f1.png")

        plt.clf()

        plt.plot(train_correct, label="Train Fully Correct")
        plt.plot(x_axis, test_correct, label="Test Fully Correct (" + str(TEST_EPOCHS) + " epochs)", linestyle='dashed')
        plt.xlabel("Epochs")
        plt.ylabel("Correct")
        plt.legend()

        # Save plot
        plt.savefig("output_images/plot_correct.png")

        plt.clf()

        plt.plot(train_loss, label="Train Loss")
        #plt.plot(x_axis, test_loss, label="Test Loss (" + str(TEST_EPOCHS) + " epochs)", linestyle='dashed')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        # Save plot
        plt.savefig("output_images/plot_loss.png")

        plt.clf()

        plt.plot(x_axis, test_loss, label="Test Loss (" + str(TEST_EPOCHS) + " epochs)", linestyle='dashed')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        # Save plot
        plt.savefig("output_images/plot_loss_test.png")

        plt.clf()

        if not BYPASS_SCORE:

            plt.plot(x_axis, train_unsat_correct, label="Train Unsat Correct")
            plt.plot(x_axis, test_unsat_correct, label="Test Unsat Correct", linestyle='dashed')
            plt.xlabel("Epochs")
            plt.ylabel("Unsat Correct")
            plt.legend()

            # Save plot
            plt.savefig("output_images/plot_unsat_correct.png")

            plt.clf()

            plt.plot(x_axis, train_unsatall_correct, label="Train Unsat All Correct")
            plt.plot(x_axis, test_unsatall_correct, label="Test Unsat All Correct", linestyle='dashed')
            plt.xlabel("Epochs")
            plt.ylabel("Unsat All Correct")
            plt.legend()

            # Save plot
            plt.savefig("output_images/plot_unsat_all_correct.png")

            plt.clf()

        conf_mat_total = conf_mat_total.astype(np.int64)
        seaborn.heatmap(conf_mat_total, annot=True, cmap='Blues', fmt='d')
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.savefig('output_images/conf_mat.png')

        plt.clf()

        if BIT_SUPERVIZ:
            divideAll(train_bit_0_correct, 10000)
            divideAll(train_bit_1_correct, 10000)
            divideAll(test_bit_0_correct, 10000)
            divideAll(test_bit_1_correct, 10000)
            
            plt.plot(train_bit_0_correct, label="Train")
            plt.plot(x_axis, test_bit_0_correct, label="Test", linestyle='dashed')
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend()

            # Save plot
            plt.savefig("output_images/plot_bit_0_correct.png")

            plt.clf()

            plt.plot(train_bit_1_correct, label="Train")
            plt.plot(x_axis, test_bit_1_correct, label="Test", linestyle='dashed')
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend()

            # Save plot
            plt.savefig("output_images/plot_bit_1_correct.png")

            train_bit_all = []
            for i in range(len(train_bit_0_correct)):
                train_bit_all.append((train_bit_0_correct[i] + train_bit_1_correct[i])/2)
            test_bit_all = []
            for i in range(len(test_bit_0_correct)):
                test_bit_all.append((test_bit_0_correct[i] + test_bit_1_correct[i])/2)
            
            plt.clf()

            plt.plot(train_bit_all, label="Train")
            plt.plot(x_axis, test_bit_all, label="Test", linestyle='dashed')
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend()

            # Save plot
            plt.savefig("output_images/plot_bit_all_correct.png")
            plt.clf()
        
        if USE_SCORE:
            divideAll(train_score, 10000)
            divideAll(test_score, 10000)

            plt.plot(x_axis, train_score, label="Train")
            plt.plot(x_axis, test_score, label="Test", linestyle='dashed')
            plt.xlabel("Epochs")
            plt.ylabel("Score")
            plt.legend()

            # Save plot
            plt.savefig("output_images/plot_score.png")

            plt.clf()
        if not BYPASS_SCORE:
            with open('output_images/raw_data_train.pickle', 'wb') as handle:
                pickle.dump([train_f, train_correct, train_loss, train_unsat_correct, train_unsatall_correct, train_bit_0_correct, train_bit_1_correct, train_bit_all, train_score], handle)
            with open('output_images/raw_data_test.pickle', 'wb') as handle:
                pickle.dump([test_f, test_correct, test_loss, test_unsat_correct, test_unsatall_correct, test_bit_0_correct, test_bit_1_correct, test_bit_all, test_score], handle)