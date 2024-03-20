import numpy as np

from TinyNN.Training.loss import loss
from TinyNN.NN.Value import Value

def train(model, loss_func, X, y, epochs = 50, n_batches = 1, alpha = 1e-4, lr = 0.1):
    batch_size = len(X) // n_batches
    for k in range(epochs):
        for b in range(n_batches):
            total_loss = loss(model, loss_func,X[b*batch_size:(b+1)*batch_size],y[b*batch_size:(b+1)*batch_size], alpha = alpha)
        
            model.zero_grad()
            total_loss.backward()
            
            learning_rate = (lr/n_batches)*(1.0 - 0.9*k/100)
            for p in model.parameters():
                # print(p)
                p.data -= learning_rate * p.grad
                
        if k % 1 == 0:
            print(f"Epoch {k}, loss {np.round(total_loss.data,4)}")