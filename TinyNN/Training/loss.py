from TinyNN.NN.Value import Value

def max_margin_loss(y, yhat):
    losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(y, yhat)] # This outputs a Value
    return sum(losses) * (1.0 / len(losses))

# def accuracy_score(y, yhat):
#     return -sum((yi > 0) == (scorei.data > 0) for yi, scorei in zip(y, yhat))/len(y)

def loss(model, loss_function, Xb, yb, alpha = 1e-4):
    inputs = [list(map(Value, xrow)) for xrow in Xb] # Casts the input data to Values
    scores = model.forwards(inputs)
    
    data_loss = loss_function(yb, scores)
    
    reg_loss = alpha * sum((p*p for p in model.parameters()))
    total_loss = data_loss + alpha*reg_loss

    return total_loss