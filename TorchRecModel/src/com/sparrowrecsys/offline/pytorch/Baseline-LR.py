import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")


def _get_train_test_df(training_path, test_path, columns2Keep):
    training_df = pd.read_csv(training_path, index_col=False)
    test_df = pd.read_csv(test_path, index_col=False)
    training_df.fillna(0, inplace=True)
    test_df.fillna(0, inplace=True)
    training_feature = training_df[columns2Keep]
    training_label = training_df['label']
    test_feature = test_df[columns2Keep]
    test_label = test_df['label']
    training_feature['userRatedMovie1'] = training_feature['userRatedMovie1'].astype('int64')
    test_feature['userRatedMovie1'] = test_feature['userRatedMovie1'].astype('int64')
    return training_feature, training_label, test_feature, test_label


class ModelDataSet(Dataset):
    # Retrieve an item in every call
    def __init__(self, input_DF, label_DF, sparse_col, dense_col):
        self.df = input_DF
        self.dense_df = input_DF.iloc[:, dense_col].astype(np.float32)
        self.sparse_df = input_DF.iloc[:, sparse_col].astype('int64')
        self.label = label_DF.astype(np.float32)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sparse_feature = torch.tensor(self.sparse_df.iloc[idx])
        dense_feature = torch.tensor(self.dense_df.iloc[idx])
        label = torch.tensor(self.label.iloc[idx])
        return {'Feature': (sparse_feature, dense_feature), 'Label': label}


class LR(nn.Module):
    def __init__(self, sparse_col_size, dense_col_size):
        # sparse_col_size: list[int]
        # dense_col_size: int
        super().__init__()
        self.sparse_col_size = sparse_col_size
        self.dense_col_size = dense_col_size

        # 1st order linear layer
        first_order_size = np.sum(sparse_col_size) + dense_col_size
        self.linear_firstOrder = nn.Linear(first_order_size, 1)

    def forward(self, sparse_feature, dense_feature):
        if len(sparse_feature.shape) == 1:  # 1D tensor converted to 2D tensor if batch_number == 1
            sparse_feature = sparse_feature.view(1, -1)
            dense_feature = dense_feature.view(1, -1)

        # convert sparse feature to oneHot and Embedding
        one_hot_list = []
        for i in range(len(self.sparse_col_size)):
            sparse_feature_input = sparse_feature[:, i] # batch x 1
            class_size = self.sparse_col_size[i]
            one_hot_vec = F.one_hot(sparse_feature_input, num_classes=class_size).squeeze(1)  # batch x class_number
            one_hot_list.append(one_hot_vec)
        one_hot_list.append(dense_feature)

        # Prepare input for 1st order layer, FM, deep layer
        sparse_one_hot = torch.cat(one_hot_list, dim=1)   # B x (sum(one_hot)+10), 10 is the size of dense_embedding
        # linear layer
        linear_logit = self.linear_firstOrder(sparse_one_hot)
        logit = linear_logit
        return F.sigmoid(logit).view(-1)


class TrainEval:
    def __init__(self, model, loss_fn, optim, device, train_dataloader, test_dataloader):
        self.device = device
        self.model = model.to(self.device)
        self.optim = optim
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.threshold = 0.5  # threshold for positive class

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            print("==========================================================")
            print("start training epoch: {}".format(epoch+1))
            loss_list = []
            pred_list = []
            label_list = []

            iteration = 1
            for train_data in self.train_dataloader:
                sparse_feature = train_data['Feature'][0].to(self.device)
                dense_feature = train_data['Feature'][1].to(self.device)
                label = train_data['Label'].to(self.device)
                prediction = self.model(sparse_feature, dense_feature)

                pred_list.extend(prediction.tolist())
                label_list.extend(label.tolist())

                cur_loss = self.loss_fn(prediction, label)
                loss_list.append(cur_loss.item())
                cur_loss.backward()
                self.optim.step()
                self.optim.zero_grad()

                # logging every 20 iteration
                if iteration % 20 == 0:
                    print("---------------------------------------------------------")
                    print("epoch {}/{}, cur_iteration is {}, logloss is {:.2f}"
                          .format(epoch+1, epochs, iteration, cur_loss.item()))
                iteration += 1

            # validation every epoch
            training_loss, training_accuracy, training_roc_score = self._get_metric(loss_list, pred_list, label_list)
            print("==========================================================")
            print("Result of epoch {}".format(epoch+1))
            print(f"training loss: {training_loss:.2f}, accuracy: {training_accuracy:.3f}, roc_score: {training_roc_score:.2f}")

            test_loss, test_accuracy, test_roc_score = self.eval()
            print(f"test loss: {test_loss:.2f}, accuracy: {test_accuracy:.3f}, roc_score: {test_roc_score:.2f}")

    def eval(self):
        # return logloss, accuracy, roc_score
        self.model.eval()
        loss_list = []
        pred_list = []
        label_list = []
        with torch.no_grad():
            for test_data in self.test_dataloader:
                sparse_feature = test_data['Feature'][0].to(self.device)
                dense_feature = test_data['Feature'][1].to(self.device)
                label = test_data['Label'].to(self.device)
                prediction = self.model(sparse_feature, dense_feature)
                cur_loss = self.loss_fn(prediction, label)

                loss_list.append(cur_loss.item())
                pred_list.extend(prediction.tolist())
                label_list.extend(label.tolist())
        return self._get_metric(loss_list, pred_list, label_list)

    def _get_metric(self, loss_list, pred_list, label_list):
        # return logloss, accuracy, roc_score
        # average logloss
        avg_loss = np.mean(loss_list)
        # roc_score
        roc_score = roc_auc_score(label_list, pred_list)
        # average accuracy
        pred_class_list = list(map(lambda x: 1 if x >= self.threshold else 0, pred_list))
        correct_count = 0
        for p, l in zip(pred_class_list, label_list):
            if p == l:
                correct_count += 1
        avg_accuracy = correct_count / len(label_list)

        return avg_loss, avg_accuracy, roc_score


if __name__ == "__main__":
    folder_path = "/home/leon/Documents/SparrowRecSys/src/main/resources/webroot/sampledata"
    training_path = folder_path + "/Pytorch_data/trainingSamples.csv"
    test_path = folder_path + "/Pytorch_data/testSamples.csv"
    columns2Keep = ['userId', 'userGenre1', 'userGenre2',  'userGenre3','userGenre4', 'userGenre5',
                    'scaleduserRatingCount','scaleduserAvgRating', 'scaleduserRatingStddev', 'userRatedMovie1',
                    'movieId',  'movieGenre1', 'movieGenre2', 'movieGenre3', 'scaledReleaseYear',
                    'scaledmovieRatingCount', 'scaledmovieAvgRating','scaledmovieRatingStddev']
    training_feature, training_label, test_feature, test_label = _get_train_test_df(training_path, test_path, columns2Keep)
    sparse_col = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13]  # column_index of sparse features
    sparse_col_size = [30001, 20, 20, 20, 20, 20, 1001, 1001, 20, 20, 20]  # number of classes per sparse_feature
    dense_col = [6, 7, 8, 14, 15, 16, 17]
    training_dataset = ModelDataSet(training_feature, training_label, sparse_col, dense_col)
    test_dataset = ModelDataSet(test_feature, test_label, sparse_col, dense_col)
    BATCH_SIZE = 100
    training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    model = LR(sparse_col_size, 7)
    loss_fn = nn.BCELoss()
    EPOCHS = 5
    LR = 0.01
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.001)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_eval = TrainEval(model, loss_fn, optimizer, dev, training_dataloader, test_dataloader)
    train_eval.train(EPOCHS)