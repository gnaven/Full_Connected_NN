import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        neurons = 40
        self.fc1 = nn.Linear(2, neurons)
        
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, neurons)
        self.fc4 = nn.Linear(neurons, neurons)
        self.fc5 = nn.Linear(neurons, neurons)
        self.fc6 = nn.Linear(neurons, neurons)
        self.fc7 = nn.Linear(neurons, neurons)
        
        self.fc8 = nn.Linear(neurons, 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        x = self.fc8(x)
        return F.log_softmax(x)
    
#==================================================================================

def test(net,X,y):
    net.eval()
    net_out = net.forward(X)
    pred = net_out.data.max(1)[1]  # get the index of the max log-probability
    correctidx = pred.eq(y.data) 
    ncorrect = correctidx.sum()
    accuracy = ncorrect.item()/len(X)
    return accuracy


def train(X,target,epochs,net = Net(),shape='ellipse'):
    
    
    # create a stochastic gradient descent optimizer
    learning_rate = 0.1
    
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    
    # create a loss function
    # last layer of the Net is a softmax which when passed through NLLLoss becomes 
    # Cross entropy loss
    criterion = nn.NLLLoss()
    
    data, target = X, y
    # run the main training loop 
    min_loss = 1000000000000
    lr = learning_rate
    for epoch in range(epochs):
        
        #if epoch ==500:
            #learning_rate = learning_rate/2
            #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)             
        
        #if epoch == 1000:
            #learning_rate = learning_rate/5
            #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)             
        
        #if epoch == 2000:
            #learning_rate = learning_rate/5
            #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9) 
        if epoch %2000==0:
            lr = learning_rate*np.exp(-0.001*epoch)
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)       
        
        #if epoch == 3000:
            #learning_rate = learning_rate/5
            #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9) 
        #if epoch == 4000:
            #learning_rate = learning_rate/2
            #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)          
        
            
            
        optimizer.zero_grad()
        out = net(X)
        
        loss = criterion(out,y)
        loss.backward()
        
        optimizer.step()
        
        #if min_loss - loss.item()>=0.1:
            #learning_rate = learning_rate/10
            #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)             
            
            
        print('Epoch ', epoch, 'Loss ', loss.item(), 'Learning rate ',lr)
        if loss.item()<min_loss:
            min_loss=loss.item()
            checkpoint = {
                'state_dict': net.state_dict(),
                'minValLoss': min_loss,
            }            
            torch.save(checkpoint, 'models/'+shape+'_best_model.pt')
            
    return net

def plot_decision_boundary(clf, X, y, filename):
    # Set min and max values and give it some padding
    #x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    #y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    x_min, x_max = -1, 1
    y_min, y_max = -1, 1
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
    
    
if __name__ == "__main__":
    
    net = Net()
    shape = 'hexa'
    #loads the model and then uses that. To not load it simply change load =False
    load = True
    if load:
        
        PATH = 'models/'+shape+'_best_model.pt'
        checkpoint = torch.load(PATH)
        net.load_state_dict(checkpoint['state_dict'])
    
    data = pd.read_csv('data/FeedForward_Data_'+shape+'.csv')
    X = data.values[:, 0:2]  # Take only the first two features.     
    X = torch.tensor(X, dtype = torch.float)   
    y = data.values[:, 2]
    y = torch.tensor(y, dtype = torch.long)   
    
    netTrained = train(X,y,5000,net,shape)
    #checkpoint = torch.load(shape+'_best_model.pt')
    #netTrained = net.load_state_dict(checkpoint['state_dict'])
    acc = test(netTrained,X,y)
    print('Accuracy ', acc)
    
    filename = shape+'_best.png'
    plot_decision_boundary(netTrained, X, y, 'plots/'+filename)

