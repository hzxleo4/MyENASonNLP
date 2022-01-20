import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
activation_functions = {
    0: nn.Sigmoid(),
    1: nn.Tanh(),
    2: nn.ReLU(),
    3: nn.LeakyReLU(),
    4: nn.Identity()
}

def create_dataset(p_val=0.1, p_test=0.2):
    #p_val代表验证集的比例，p_test代表测试集的比例
    #创建数据集p
    import numpy as np
    import sklearn.datasets

    # Generate a dataset and plot it
    np.random.seed(0)
    num_samples = 1000

    #X, y = sklearn.datasets.make_moons(num_samples, noise=0.2)
    X,y = np.loadtxt("X.txt", dtype=int, delimiter=","),np.loadtxt("y.txt", dtype=int, delimiter=",")
    train_end = int(len(X)*(1-p_val-p_test))
    val_end = int(len(X)*(1-p_test))
    
    # define train, validation, and test sets
    X_tr = X[:train_end]
    X_val = X[train_end:val_end]
    X_te = X[val_end:]
    #print(X.shape,X_tr.shape)
    # and labels
    y_tr = y[:train_end]
    y_val = y[train_end:val_end]
    y_te = y[val_end:]

    #plt.scatter(X_tr[:,0], X_tr[:,1], s=40, c=y_tr, cmap=plt.cm.Spectral)
    return X_tr, y_tr, X_val, y_val

class Net(nn.Module):

    def __init__(self,num_features, num_classes): 
        #layers为一个列表，里面存储着一个顺序的网络结构
        #根据已经搜索出来的网络结构来创建一个实例
        super(Net, self).__init__()
        #share module
        self.node_hidden_units = [num_features,32,64,64,32,num_classes]
        self.edges = [(0,1),(0,2),(0,3),(0,4),(0,5),(1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)]
        self.share_edges_model = []
        for s,t in self.edges:
            s_units = self.node_hidden_units[s]
            t_units = self.node_hidden_units[t]
            #print(s_units,t_units)
            self.share_edges_model.append(nn.Linear(s_units,t_units))

        #self.optimizer = optim.Adam(self.parameters(), lr=1e-2)
    def add_layers(self,edges,acts):
        layers_added = []
        for i in range(len(edges)):
            layers_added.append(self.share_edges_model[edges[i]])
            layers_added.append(activation_functions[acts[i]])
        self.layers = nn.Sequential(*layers_added)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)
    def forward(self,x):
        
        #前向传播
        return self.layers(x)
    
def accuracy(ys, ts):
    #计算准确率
    # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
    correct_prediction = torch.eq(ts.long(), torch.max(ys, 1)[1])
    # averaging the one-hot encoded vector
    return torch.mean(correct_prediction.float())
    
def weight_reset(m):
    #重设m这一层的参数
    if isinstance(m, nn.Linear):
        m.reset_parameters()

class ChildNet():

    def __init__(self):
        #创建数据
        self.criterion = nn.CrossEntropyLoss()
        X_tr, y_tr, X_val, y_val = create_dataset()
        self.X_tr = X_tr.astype('float32')
        self.y_tr = y_tr.astype('float32')
        self.X_val = X_val.astype('float32')
        self.y_val = y_val.astype('float32')
        
        self.num_features = X_tr.shape[-1]
        self.num_classes = 2
        self.net = Net(X_tr.shape[-1],2)
    def compute_reward(self, edges,acts, num_epochs):
        # store loss and accuracy for information
        self.net.add_layers(edges,acts)
        #print(self.net)
        train_losses = []
        val_accuracies = []
        patience = 10
        #创建对应的网络
        #net = Net(layers, self.num_features, self.num_classes, self.layer_limit)
        #print(net)
        max_val_acc = 0
        
        # get training input and expected output as torch Variables and make sure type is correct
        tr_input = Variable(torch.from_numpy(self.X_tr))
        tr_targets = Variable(torch.from_numpy(self.y_tr))

        # get validation input and expected output as torch Variables and make sure type is correct
        val_input = Variable(torch.from_numpy(self.X_val))
        val_targets = Variable(torch.from_numpy(self.y_val))
        
        patient_count = 0
        # training loop
        #正常训练
        for e in range(num_epochs):

            # predict by running forward pass
            tr_output = self.net(tr_input)
            # compute cross entropy loss
            #tr_loss = F.cross_entropy(tr_output, tr_targets.type(torch.LongTensor)) 
            tr_loss = self.criterion(tr_output.float(), tr_targets.long())
            # zeroize accumulated gradients in parameters
            self.net.optimizer.zero_grad()
            
            # compute gradients given loss
            tr_loss.backward()
            #print(net.l_1.weight.grad)
            # update the parameters given the computed gradients
            self.net.optimizer.step()
            
            train_losses.append(tr_loss.data.numpy())
        
            #AFTER TRAINING

            # predict with validation input
            val_output = self.net(val_input)
            val_output = torch.argmax(F.softmax(val_output, dim=-1), dim=-1)
            
            # compute loss and accuracy
            #val_loss = self.criterion(val_output.float(), val_targets.long())
            val_acc = torch.mean(torch.eq(val_output, val_targets.type(torch.LongTensor)).type(torch.FloatTensor))
            
            #accuracy(val_output, val_targets)
            val_acc = float(val_acc.numpy())
            val_accuracies.append(val_acc)
            
            
            #early-stopping
            if max_val_acc > val_acc:
                patient_count += 1             
                if patient_count == patience:
                    break
            else:
                max_val_acc = val_acc
                patient_count = 0
            
            #print(e)

        #reset weights
        #net.apply(weight_reset)
            
        return val_acc#max_val_acc#**3 #-float(val_loss.detach().numpy()) 
