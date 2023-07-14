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

import model

def getCorrectUnsat(batch, pred, data):
    correct = 0
    for i in range(np.max(batch)+1):
        pred_now = pred[batch == i]
        if data.cnf_data[i].isUnsatCoreKissat(pred_now):
            correct += 1
    return correct

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
    mode_type = int(input("1. Normal \n2. Test \n3. Single \nMode: "))

    data_train = []
    data_test = []
    test_type = -1
    root_folder = "../dimacs_min/"

    BATCH_SIZE = 128
    EPOCHS = 100
    TEST_EPOCHS = 20
    ITERATIONS = 100
    LOSS_WEIGHT_OFFSET = 1
    MODEL_SAVE_EPOCHS = 10

    total_positive = 0.0
    total_negative = 0.0
    total_always_negative_train = 0.0
    total_always_negative_test = 0.0
    
    if mode_type == 1 or mode_type == 2:
        if mode_type == 1:
            unsat_cores_train = {}

            print("Loading training data...")
            with open('extracted_cores/unsat_cores_train.pickle', 'rb') as handle:
                unsat_cores_train = pickle.load(handle)
            for cnf_file in tqdm(unsat_cores_train.keys()):
                cnf_data = unsat_cores_train[cnf_file]
                #print(np.sum(cnf_data.unsat_core)/cnf_data.n_clause)

                features, mask = cnf_data.getFeatures()
                data = Data(x=torch.tensor(features).float(), edge_index=torch.tensor(cnf_data.out_graph),\
                            y=torch.tensor(cnf_data.unsat_core), mask=torch.tensor(mask), \
                                edge_attr = torch.tensor(cnf_data.edge_attr).float(),cnf_data=cnf_data)
                data_train.append(data)

                positive = torch.sum(data.y).item()
                total_positive += positive
                total_negative += cnf_data.n_clause - positive
                total_always_negative_train += cnf_data.n_var
            
            train_loader = DataLoader(
                data_train,
                batch_size=BATCH_SIZE,
                shuffle=True
            )
        else:
            test_type = input("Test type: ")
            model_path = "model_" + input("Model name: ")

        unsat_cores_test = {}

        print("Loading testing data...")
        test_path = "unsat_cores_test.pickle"
        if test_type >= 0:
            test_path = "unsat_cores_satcomp(" + test_type + ").pickle"

        with open('extracted_cores/' + test_path, 'rb') as handle:
            unsat_cores_test = pickle.load(handle)

        for cnf_file in tqdm(unsat_cores_test.keys()):
            cnf_data = unsat_cores_test[cnf_file]

            features, mask = cnf_data.getFeatures()
            data = Data(x=torch.tensor(features).float(), edge_index=torch.tensor(cnf_data.out_graph),\
                        y=torch.tensor(cnf_data.unsat_core), mask=torch.tensor(mask), \
                            edge_attr = torch.tensor(cnf_data.edge_attr).float(),cnf_data=cnf_data)
            data_test.append(data)
            
            positive = torch.sum(data.y).item()
            total_positive += positive
            total_negative += cnf_data.n_clause - positive
            total_always_negative_test += cnf_data.n_var

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
    
    gnn_model = model.GNNSat_V2(iterations=ITERATIONS)

    if mode_type == 2:
        gnn_model.load_state_dict(torch.load("models/" + model_path + ".pt"))
        gnn_model.to(device)
        print("Testing sequence...")
        gnn_model.eval()

        sum_unsat_correct = 0
        sum_correct = 0
        conf_mat_total = np.asarray([[-total_always_negative_test,0],[0,0]])

        for data in tqdm(test_loader):
            y_cpu = data.y.to('cpu').numpy()

            y = data.y.to(device).float()
            data = data.to(device)

            out = gnn_model(data)

            pred = torch.sigmoid(out).detach().to('cpu').numpy()
            pred = np.where(pred == 0.5, 0, pred)
            pred = np.where(pred > 0.5, 1, 0)

            sum_correct += getCorrect(data.batch.detach().to('cpu').numpy(), pred, y_cpu)
            sum_unsat_correct += getCorrectUnsat(data.batch.detach().to('cpu').numpy(), pred, data)
            conf_mat_total += confusion_matrix(y_cpu, pred)

        f1_score = getF1Score(conf_mat_total)
        print("   Test Correct: ", sum_correct)
        print("   Test Unsat Correct: ", sum_unsat_correct)
        
        print("   Confusion Matrix:\n")
        print("    ", "%8d %8d" % (int(conf_mat_total[0][0]), int(conf_mat_total[0][1])))
        print("    ", "%8d %8d" % (int(conf_mat_total[1][0]), int(conf_mat_total[1][1])))
        print("   F1 Score:", f1_score)

    else:
        optim = torch.optim.Adam(gnn_model.parameters())
        lossfunc = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(total_negative/total_positive)*LOSS_WEIGHT_OFFSET)

        test_f = []
        test_correct = []
        test_unsat_correct = []
        test_loss = []

        train_f = []
        train_correct = []
        train_unsat_correct = []
        train_loss = []

        gnn_model.to(device)

        print("Starting training sequence...")
        for epoch in range(1, EPOCHS+1):
            # Train
            gnn_model.train()

            sum_loss = 0
            sum_correct = 0
            sum_unsat_correct = 0
            conf_mat_total = np.asarray([[-total_always_negative_train,0],[0,0]])

            for data in tqdm(train_loader):
                optim.zero_grad()
                
                y_cpu = data.y.to('cpu').numpy()

                y = data.y.to(device).float()
                data = data.to(device)

                out = gnn_model(data)
                loss = lossfunc(out, y)

                sum_loss += float(loss)

                loss.backward()
                optim.step()

                pred = torch.sigmoid(out).detach().to('cpu').numpy()
                pred = np.where(pred == 0.5, 0, pred)
                pred = np.where(pred > 0.5, 1, 0)

                sum_correct += getCorrect(data.batch.detach().to('cpu').numpy(), pred, y_cpu)
                if epoch % TEST_EPOCHS == 0:
                    sum_unsat_correct += getCorrectUnsat(data.batch.detach().to('cpu').numpy(), pred, data)
                conf_mat_total += confusion_matrix(y_cpu, pred)

            sum_loss /= len(data_train)
            f1_score = getF1Score(conf_mat_total)
            print("\n\nEpoch:", epoch)
            print("   Train Loss: ", sum_loss )
            print("   Train Fully Correct: ", sum_correct)
            if epoch % TEST_EPOCHS == 0:
                print("   Train Unsat Correct: ", sum_unsat_correct)
            print("   Confusion Matrix:")
            print("    ", "%8d %8d" % (int(conf_mat_total[0][0]), int(conf_mat_total[0][1])))
            print("    ", "%8d %8d" % (int(conf_mat_total[1][0]), int(conf_mat_total[1][1])))
            print("   F1 Score:", f1_score)
            print("\n")

            train_f.append(f1_score)
            train_correct.append(sum_correct)
            train_loss.append(sum_loss)
            if epoch % TEST_EPOCHS == 0:
                train_unsat_correct.append(sum_unsat_correct)


            # Testing
            if epoch % TEST_EPOCHS == 0:
                print("Testing sequence...")
                gnn_model.eval()

                sum_unsat_correct = 0
                sum_correct = 0
                sum_loss = 0
                conf_mat_total = np.asarray([[-total_always_negative_test,0],[0,0]])

                for data in tqdm(test_loader):
                    y_cpu = data.y.to('cpu').numpy()

                    y = data.y.to(device).float()
                    data = data.to(device)

                    out = gnn_model(data)
                    loss = lossfunc(out, y)

                    sum_loss += float(loss)

                    pred = torch.sigmoid(out).detach().to('cpu').numpy()
                    pred = np.where(pred == 0.5, 0, pred)
                    pred = np.where(pred > 0.5, 1, 0)

                    sum_correct += getCorrect(data.batch.detach().to('cpu').numpy(), pred, y_cpu)
                    sum_unsat_correct += getCorrectUnsat(data.batch.detach().to('cpu').numpy(), pred, data)
                    conf_mat_total += confusion_matrix(y_cpu, pred)

                sum_loss /= len(data_train)
                f1_score = getF1Score(conf_mat_total)
                print("\n\n   Test Loss: ", sum_loss )
                print("   Test Correct: ", sum_correct)
                print("   Test Unsat Correct: ", sum_unsat_correct)
                
                print("   Confusion Matrix:\n")
                print("    ", "%8d %8d" % (int(conf_mat_total[0][0]), int(conf_mat_total[0][1])))
                print("    ", "%8d %8d" % (int(conf_mat_total[1][0]), int(conf_mat_total[1][1])))
                print("   F1 Score:", f1_score)
                print("\n")
                
                test_f.append(f1_score)
                test_correct.append(sum_correct)
                test_unsat_correct.append(sum_unsat_correct)
                test_loss.append(sum_loss)

                #if sum_acc > 0.7 and sum_acc < 0.75:
                #    break
            if epoch % MODEL_SAVE_EPOCHS == 0:
                torch.save(gnn_model.state_dict(), "models/model_" + str(epoch) + ".pt")
        

        gnn_model.eval()
        final_test = Batch.from_data_list([data_train[0]])

        y_cpu = final_test.y.to('cpu').numpy()
        data = final_test.to(device)

        out = gnn_model(data, print_data=True)

        pred = torch.sigmoid(out).detach().to('cpu').numpy()
        print(pred)
        pred = np.where(pred > 0.5, 1, 0)

        y_cpu = y_cpu.astype(pred.dtype)

        print(np.count_nonzero(pred), np.count_nonzero(pred - 1))
        print(pred[-100:])
        print(final_test.mask.detach().to('cpu').numpy().astype(pred.dtype)[-100:])
        print(y_cpu[-100:])


        if TEST_EPOCHS > 1:
            test_f.insert(0, test_f[0])
            test_correct.insert(0, test_correct[0])
            test_unsat_correct.insert(0, test_unsat_correct[0])
            train_unsat_correct.insert(0, train_unsat_correct[0])
            test_loss.insert(0, test_loss[0])

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
        plt.plot(x_axis, test_loss, label="Test Loss (" + str(TEST_EPOCHS) + " epochs)", linestyle='dashed')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        # Save plot
        plt.savefig("output_images/plot_loss.png")

        plt.clf()

        plt.plot(x_axis, train_unsat_correct, label="Train Unsat Correct")
        plt.plot(x_axis, test_unsat_correct, label="Test Unsat Correct", linestyle='dashed')
        plt.xlabel("Epochs")
        plt.ylabel("Unsat Correct")
        plt.legend()

        # Save plot
        plt.savefig("output_images/plot_unsat_correct.png")

        plt.clf()


        seaborn.heatmap(conf_mat_total, annot=True, cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.savefig('output_images/conf_mat.png')