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
    for i in range(1, 6): # convert userRatedMovie columns to longtype
        column_name = "userRatedMovie" + str(i)
        training_feature[column_name] = training_feature[column_name].astype('int64')
        test_feature[column_name] = test_feature[column_name].astype('int64')
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


class Attention_ElementWiseProduct(nn.Module):
    """
      Input:
          behavior: 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
          candidate: 3D tensor with shape: ``(batch_size,1,embedding_size)``.
      Output:
          attention_weight: 3D tensor with shape: ``(batch_size, field_size, 1)``.
    """

    def __init__(self, embedding_size):
        super().__init__()
        self.linear1 = nn.Linear(4*embedding_size, 32)
        self.linear2 = nn.Linear(32, 1)
        self.prelu = nn.PReLU()

    def forward(self, behavior, candidate):
        candidate = candidate.expand_as(behavior)
        embed_input = torch.cat([behavior, candidate, behavior-candidate, behavior*candidate], dim=2)  # (B,F,4k)
        output = self.prelu(self.linear1(embed_input))  # (B,F,32)
        output = F.sigmoid(self.linear2(output)) # (B,F,1)
        return output


class DIN(nn.Module):
    def __init__(self, sparse_col_size, dense_col_size, sparse_ratio):
        # sparse_col_size: list[int]
        # dense_col_size: int
        super().__init__()
        self.sparse_col_size = sparse_col_size
        self.dense_col_size = dense_col_size
        self.sparse_ratio = sparse_ratio

        # For categorical features, we embed the features in dense vectors of dimension of 6 * category cardinality^1/4
        embedding_size = list(map(lambda x: int(6 * pow(x, 0.25)), self.sparse_col_size))

        # Attention layer
        movieId_embed_size = embedding_size[3]
        self.attention_layer = Attention_ElementWiseProduct(movieId_embed_size)

        # Embedding layer for all sparse features
        sparse_embedding_list = []
        for class_size, embed_size in zip(self.sparse_col_size, embedding_size):
            embed_layer = nn.Embedding(class_size, embed_size, scale_grad_by_freq=True)
            # init embed_layer
            # embed_layer.weight.data.uniform_(-1/math.sqrt(class_size), 1/math.sqrt(class_size))
            sparse_embedding_list.append(embed_layer)
        self.sparse_embedding_layer = nn.ModuleList(sparse_embedding_list)

        # Deep layers
        deep_input_size = np.sum(embedding_size) + dense_col_size - 4*movieId_embed_size
        self.linear1 = nn.Linear(deep_input_size, 128)
        self.prelu1 = nn.PReLU()
        self.linear2 = nn.Linear(128, 64)
        self.prelu2 = nn.PReLU()
        self.linear3 = nn.Linear(64, 1)

    def forward(self, sparse_feature, dense_feature):
        if len(sparse_feature.shape) == 1:  # 1D tensor converted to 2D tensor if batch_number == 1
            sparse_feature = sparse_feature.view(1, -1)
            dense_feature = dense_feature.view(1, -1)

        # convert sparse feature to oneHot and Embedding
        embedding_list=[]
        for i in range(len(self.sparse_col_size)):
            sparse_feature_input = sparse_feature[:, i]  # batch x 1
            class_size = self.sparse_col_size[i]
            embedding_layer = self.sparse_embedding_layer[i]
            embedding_output = embedding_layer(sparse_feature_input).squeeze(1)  # batch x embedding_size
            embedding_list.append(embedding_output)

        # Split into "other sparse feature", behavior feature, candidate feature
        sparse_embed_list = embedding_list[:self.sparse_ratio[0]]
        behavior_embed_list = embedding_list[self.sparse_ratio[0]:self.sparse_ratio[0]+self.sparse_ratio[1]]
        candidate = embedding_list[-1].unsqueeze(1)

        sparse_embedding = torch.cat(sparse_embed_list, dim=1)
        behavior = torch.stack(behavior_embed_list, dim=1)  # B x field_number x embedding_size

        # Cal the attention weight
        attention_weight = self.attention_layer(behavior, candidate)  # B x field_number x 1

        # Apply attention weight and do sumPooling
        attention_behavior = attention_weight * behavior  # B x field_number x embedding_size
        sumPool_behavior = attention_behavior.sum(dim=1)  # B x embedding_size

        deep_input = torch.cat([sparse_embedding, dense_feature, sumPool_behavior, candidate.squeeze()], dim=1)

        # Deep layer
        deep_output = self.prelu1(self.linear1(deep_input))
        deep_output = self.prelu2(self.linear2(deep_output))
        deep_output = self.linear3(deep_output)
        return F.sigmoid(deep_output).view(-1) # (B,)


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
    columns2Keep = ['userId',
                    'userGenre1',
                    'scaleduserRatingCount',
                    'scaleduserAvgRating',
                    'scaleduserRatingStddev',
                    'userRatedMovie1',
                    'userRatedMovie2',
                    'userRatedMovie3',
                    'userRatedMovie4',
                    'userRatedMovie5',
                    'movieId',
                    'movieGenre1',
                    'scaledReleaseYear',
                    'scaledmovieRatingCount',
                    'scaledmovieAvgRating',
                    'scaledmovieRatingStddev']
    training_feature, training_label, test_feature, test_label = _get_train_test_df(training_path, test_path, columns2Keep)
    sparse_col = [0, 1, 11, 5, 6, 7, 8, 9, 10] # column_index of sparse features
    sparse_col_size = [30001, 20, 20, 1001, 1001, 1001, 1001, 1001, 1001] # number of classes per sparse_feature
    sparse_ratio = [3, 5, 1] # column_number ratio of sparse column: behavior column: candidate column
    dense_col = [2, 3, 4, 12, 13, 14, 15]
    training_dataset = ModelDataSet(training_feature, training_label, sparse_col, dense_col)
    test_dataset = ModelDataSet(test_feature, test_label, sparse_col, dense_col)
    BATCH_SIZE = 100
    training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    model = DIN(sparse_col_size, 7, sparse_ratio)
    loss_fn = nn.BCELoss()
    EPOCHS = 5
    LR = 0.001
    # Add weight decay for all parameters in the model other than those in PReLU layer
    para_list = []
    for para_name, para in model.named_parameters():
        if 'prelu' in para_name:
            para_list.append({'params': para, 'weight_decay':0})
        else:
            para_list.append({'params': para})
    optimizer = optim.Adam(para_list, lr=LR, weight_decay=0.001)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_eval = TrainEval(model, loss_fn, optimizer, dev, training_dataloader, test_dataloader)
    train_eval.train(EPOCHS)
