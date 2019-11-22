import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(2, 20)
        
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)

        self.fc4 = nn.Linear(20, 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x = self.fc4(x)
        return F.log_softmax(x)
    
class NetLayer(nn.Module):
    def __init__(self,neurons):
        super(NetLayer, self).__init__()
        
        self.fc1 = nn.Linear(2, neurons)
        
        self.fc2 = nn.Linear(neurons, neurons)
        #self.fc3 = nn.Linear(neurons, neurons)

        self.fc4 = nn.Linear(neurons, 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))

        x = self.fc4(x)
        return F.log_softmax(x)    
    
"""-----------------------------------------------------------------------------------"""
def plot_data(X, y, filename):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, s = 1)
    plt.savefig(filename)
    plt.close()
        
def plot_decision_boundary(clf, X, y, filename):
    # Set min and max values and give it some padding
    #x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    #y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    #Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    X_out = clf(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype = torch.float))
    Z = X_out.data.max(1)[1]
    # Z.shape
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    #plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, s = 1)
    plt.savefig(filename)
    plt.close()
    
def test(net,pred,X,y):
    
    net_out = net(X)
    pred = net_out.data.max(1)[1]  # get the index of the max log-probability
    correctidx = pred.eq(y.data) 
    ncorrect = correctidx.sum()
    accuracy = ncorrect.item()/len(X)
    return accuracy
    
def train(X,target,epochs,net = Net(), model_select='acc'):
    
    
    # create a stochastic gradient descent optimizer
    learning_rate = .01
    
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    
    # create a loss function
    # last layer of the Net is a softmax which when passed through NLLLoss becomes 
    # Cross entropy loss
    criterion = nn.NLLLoss()
    
    data, target = X, y
    # run the main training loop    
    
    for epoch in range(epochs):
        
              
        optimizer.zero_grad()
        out = net(X)
        
        loss = criterion(out,y)
        loss.backward()
        
        optimizer.step()
        #print('Epoch ', epoch, 'Loss ', loss.item())
        acc = test(net,out,X,y)
        #print('Accuracy ', acc)
        
        if model_select =='acc':
            
            if acc==1:
                print('Achieved accuracy 100%')                
                print('Epoch ', epoch, 'Loss ', loss.item())
                print('Accuracy ', acc)
                
                #plot_decision_boundary(net, X, y, 'XOR_decision_bound.png')
                #torch.save(net, 'XOR_best_model')
                break
        # selecting on cross entropy loss
        elif model_select == 'loss':
            if loss.item()<1e-4:
                print('Achieved loss below 1e-4')
                print('Epoch ', epoch, 'Loss ', loss.item())
                print('Accuracy ', acc)                
                break
        else:
            # selectin model when acc==1 and loss below 1e-4
            if acc==1 and loss.item()<1e-4:
                print('Achieved acc 100% and loss below 1e-4')
                print('Epoch ', epoch, 'Loss ', loss.item())
                print('Accuracy ', acc)                
                break
        
    return acc,loss.item(),net

def layer(X,target,epochs,max_neurons,model_selection='acc'):
    currentNet = None
    currentNeuron = 0
    for neuron in range(max_neurons,1,-1):
        net = NetLayer(neuron)
        acc,loss,netTrained = train(X, target, epochs,net,model_select=model_selection)
        
        if model_selection=='accloss' and acc<1 and loss>1e-4:
            print('Min Number of Neuron ',currentNeuron)            
            return currentNet,currentNeuron
        
        if model_selection=='acc' and acc <1:
            print('Min Number of Neuron ',currentNeuron)
            return currentNet,currentNeuron
        
        currentNet = netTrained
        currentNeuron= neuron
    
    return currentNet,currentNeuron
    

if __name__ == "__main__":
    
    data_list = [[0.,0.,0.],[0.,1.,1.],[1.,0.,1.],[1.,1.,0.]]
    data = pd.DataFrame(data_list)
    X = data.values[:, 0:2]  # Take only the first two features.     
    X = torch.tensor(X, dtype = torch.float)   
    y = data.values[:, 2]
    y = torch.tensor(y, dtype = torch.long)   
    
    
    #=================================================================================================
    """
    Part 1.1 a) Model selected when achieves 100% accuracy
    """
    net = Net()
    acc,loss, netTrained = train(X, y, epochs=1000, net=net, model_select='acc')
    plot_decision_boundary(netTrained, X, y, 'plots/XOR_decision_bound_100%acc.png')
    torch.save(net, 'models/XOR_best_model_acc_100%') 
    """
    Part 1.1 b) Model selected when achieves loss <1e-4
    """
    net = Net()
    acc,loss,netTrained = train(X, y, epochs=10000, net=net, model_select='loss')
    plot_decision_boundary(netTrained, X, y, 'plots/XOR_decision_bound_1e-4loss.png')
    torch.save(net, 'models/XOR_best_model_loss_1e-4') 
    
    #==================================================================================================
    
    """
    Part 1.2 a) Model selected when achieves 100% accuracy with smallest capactity
    Model is chosen when accuracy is 100%
    """    
    net, neuron = layer(X, y, 10000, 15,model_selection='acc')
    plot_decision_boundary(net, X, y, 'plots/XOR_decision_bound_100%acc_minNeuron'+str(neuron)+'.png')
    torch.save(net, 'models/XOR_best_model_acc_minNeuron'+str(neuron))    
    
    """
    Part 1.2 b) Model selected when achieves 100% accuracy and loss below 1e-4 with smallest capactity
    Model is chosen when accuracy is 100%
    """       
    net, neuron = layer(X, y, 100000, 20,model_selection='accloss')
    plot_decision_boundary(net, X, y, 'plots/XOR_decision_bound_AccLoss_minNeuron'+str(neuron)+'.png')
    torch.save(net, 'models/XOR_best_model_AccLoss_minNeuron'+str(neuron)) 
    
    #====================================================================================================
    
    """

    """