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
    return training_feature, training_label, test_feature, test_label


class ModelDataSet(Dataset):
    # Retrieve an item in every call
    def __init__(self, input_DF, label_DF, sparse_col):
        self.df = input_DF
        self.sparse_df = input_DF.iloc[:, sparse_col].astype('int64')
        self.label = label_DF.astype(np.float32)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sparse_feature = torch.tensor(self.sparse_df.iloc[idx])
        label = torch.tensor(self.label.iloc[idx])
        return {'Feature': sparse_feature, 'Label': label}


class NeuralCFEarlyCross(nn.Module):
    def __init__(self, sparse_col_size):
        super().__init__()
        self.sparse_col_size = sparse_col_size
        # For categorical features, we embed the features in dense vectors of dimension of 6 * category cardinality^1/4
        embedding_size = list(map(lambda x: int(6 * pow(x, 0.25)), self.sparse_col_size))

        # Create embedding layer for all sparse features
        sparse_embedding_list = []
        for class_size, embed_size in zip(self.sparse_col_size, embedding_size):
            sparse_embedding_list.append(nn.Embedding(class_size, embed_size, scale_grad_by_freq=True))
        self.sparse_embedding_layer = nn.ModuleList(sparse_embedding_list)

        # Cal total embedding size
        total_embedding_size = np.sum(embedding_size)

        # Deep layer
        self.linear1 = nn.Linear(total_embedding_size, 10)
        self.linear2 = nn.Linear(10, 10)
        # final linear layer
        self.linear3 = nn.Linear(10, 1)

    def forward(self, sparse_feature):
        if len(sparse_feature.shape) == 1:  # 1D tensor converted to 2D tensor if batch_number == 1
            sparse_feature = sparse_feature.view(1, -1)

        # Deep side
        embedding_list = []
        sparse_feature_size = sparse_feature.shape[1]
        for i in range(sparse_feature_size):
            sparse_feature_input = sparse_feature[:, i]  # batch x 1
            embedding_layer = self.sparse_embedding_layer[i]
            embedding_output = embedding_layer(sparse_feature_input)  # batch x 1 x embedding_size
            embedding_list.append(embedding_output.squeeze(1))  # batch x embedding_size
        embedding = torch.cat(embedding_list, dim=1)  # batch x sum(embedding_size)
        deep_output = F.relu(self.linear1(embedding))  # batch x 10
        deep_output = F.relu(self.linear2(deep_output))  # batch x 10
        output = F.sigmoid(self.linear3(deep_output))  # batch x 1
        return output.view(-1)  # (batch,)


class NeuralCFLateCross(nn.Module):
    def __init__(self, sparse_col_size):
        super().__init__()
        self.sparse_col_size = sparse_col_size
        # For categorical features, we embed the features in dense vectors of dimension of 6 * category cardinality^1/4
        embedding_size = list(map(lambda x: int(6 * pow(x, 0.25)), self.sparse_col_size))

        # Create embedding layer for all sparse features
        sparse_embedding_list = []
        for class_size, embed_size in zip(self.sparse_col_size, embedding_size):
            sparse_embedding_list.append(nn.Embedding(class_size, embed_size, scale_grad_by_freq=True))
        self.sparse_embedding_layer = nn.ModuleList(sparse_embedding_list)

        # Deep layer for user side
        self.linear_u1 = nn.Linear(embedding_size[0], 10)
        self.linear_u2 = nn.Linear(10, 10)

        # Deep layer for item side
        self.linear_i1 = nn.Linear(embedding_size[1], 10)
        self.linear_i2 = nn.Linear(10, 10)

        # final linear layer
        self.linear3 = nn.Linear(20, 1)

    def forward(self, sparse_feature):
        if len(sparse_feature.shape) == 1:  # 1D tensor converted to 2D tensor if batch_number == 1
            sparse_feature = sparse_feature.view(1, -1)

        # user side deep layer
        user_feature = sparse_feature[:, 0]
        user_embedding_layer = self.sparse_embedding_layer[0]
        user_embedding = user_embedding_layer(user_feature).squeeze(1)
        user_output = self.linear_u1(user_embedding)
        user_output = self.linear_u2(user_output)

        # item side deep layer
        item_feature = sparse_feature[:, 1]
        item_embedding_layer = self.sparse_embedding_layer[1]
        item_embedding = item_embedding_layer(item_feature).squeeze(1)
        item_output = self.linear_i1(item_embedding)
        item_output = self.linear_i2(item_output)

        # concat and feed to final linear layer
        concat_output = torch.cat([user_output, item_output], dim=1)
        output = F.sigmoid(self.linear3(concat_output))  # batch x 1
        return output.view(-1)  # (batch,)


class NeuralCFDotProduct(nn.Module):
    def __init__(self, sparse_col_size):
        super().__init__()
        self.sparse_col_size = sparse_col_size
        # For categorical features, we embed the features in dense vectors of dimension of 6 * category cardinality^1/4
        embedding_size = list(map(lambda x: int(6 * pow(x, 0.25)), self.sparse_col_size))

        # Create embedding layer for all sparse features
        sparse_embedding_list = []
        for class_size, embed_size in zip(self.sparse_col_size, embedding_size):
            sparse_embedding_list.append(nn.Embedding(class_size, embed_size, scale_grad_by_freq=True))
        self.sparse_embedding_layer = nn.ModuleList(sparse_embedding_list)

        # Deep layer for user side
        self.linear_u1 = nn.Linear(embedding_size[0], 10)
        self.linear_u2 = nn.Linear(10, 10)

        # Deep layer for item side
        self.linear_i1 = nn.Linear(embedding_size[1], 10)
        self.linear_i2 = nn.Linear(10, 10)

    def forward(self, sparse_feature):
        if len(sparse_feature.shape) == 1:  # 1D tensor converted to 2D tensor if batch_number == 1
            sparse_feature = sparse_feature.view(1, -1)

        # user side deep layer
        user_feature = sparse_feature[:, 0]
        user_embedding_layer = self.sparse_embedding_layer[0]
        user_embedding = user_embedding_layer(user_feature).squeeze(1)
        user_output = self.linear_u1(user_embedding)
        user_output = self.linear_u2(user_output)

        # item side deep layer
        item_feature = sparse_feature[:, 1]
        item_embedding_layer = self.sparse_embedding_layer[1]
        item_embedding = item_embedding_layer(item_feature).squeeze(1)
        item_output = self.linear_i1(item_embedding)
        item_output = self.linear_i2(item_output)

        # dot product of user and item embedding
        return F.sigmoid(torch.bmm(user_output.unsqueeze_(1), item_output.unsqueeze_(2)).squeeze()) # (batch,)


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
                sparse_feature = train_data['Feature'].to(self.device)
                label = train_data['Label'].to(self.device)
                prediction = self.model(sparse_feature)

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
                sparse_feature = test_data['Feature'].to(self.device)
                label = test_data['Label'].to(self.device)
                prediction = self.model(sparse_feature)
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
    columns2Keep = ['userId',  'movieId']
    training_feature, training_label, test_feature, test_label = _get_train_test_df(training_path, test_path, columns2Keep)
    sparse_col = [0, 1]  # column_index of sparse features
    sparse_col_size = [30001, 1001]  # number of classes per sparse_feature
    training_dataset = ModelDataSet(training_feature, training_label, sparse_col)
    test_dataset = ModelDataSet(test_feature, test_label, sparse_col)
    BATCH_SIZE = 100
    training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    model = NeuralCFDotProduct(sparse_col_size)
    loss_fn = nn.BCELoss()
    EPOCHS = 5
    LR = 0.001
    optimizer = optim.Adam(model.parameters(), lr=LR)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_eval = TrainEval(model, loss_fn, optimizer, dev, training_dataloader, test_dataloader)
    train_eval.train(EPOCHS)
