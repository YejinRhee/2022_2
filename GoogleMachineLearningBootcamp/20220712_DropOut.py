'''
확률론에서 확률 변수의 기대값 μX = E[X] = ∑ xi*p(xi)
각 사건이 벌어졌을 때의 이득과
그 사건이 벌어질 확률을 곱한 것을 
전체 사건에 대해 합한 값

Drop Out :
Can't rely on any one feature, so have to spread out weights
전체 사건은 그대로 두되, layer 들 간의 연결이 강한 부분을 느슨하게 하는 등의 효과를 통해 
overfitting 을 막아줌
Deleting Layer를 multiply하면 '각 사건'이 없어지게 되어, 전체 사건이 줄어들게 됩니다.
전체 사건이 줄어들면 keep_prop 을 적용하기 전과 후의 문맥이 달라짐 ! 동일한 조건이 아니게 됨
=> Inverted Drop Out
따라서 1/keep_prop 을 곱해줌으로써(그만큼 값을 키워줌으로써) 
문맥(기댓값)을 keep_prop을 적용하지 않았을 떄와 유사하게 만들어 줌

한편, Forward Propagation때 drop out을 했다면 
Backward Propagation 에서도 drop out을 해야 함
In general, it's important to account for anything that you're doing in the forward step 
in the backward step as well -
otherwise you're computing a gradient of a different function than you're evaluating.
'''
import numpy as np
import matplotlib.pyplot as plt

# GRADED FUNCTION: forward_propagation_with_dropout

def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
    np.random.seed(1)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    D1 = (D1 < keep_prob).astype(int)                  # D1 = D1 < keep_prob 일케해도 된다 ! 
    A1 =  A1 * D1 
    A1 =  A1 / keep_prob 
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    D2 = np.random.rand(A2.shape[0], A2.shape[1])        # Step 1: initialize matrix D2 = np.random.rand(..., ...)
    D2 = D2 < keep_prob                                  # Step 2: convert entries of D2 to 0 or 1 (using keep_prob as the threshold)
    A2 = A2 * D2                                         # Step 3: shut down some neurons of A2
    A2 = A2 / keep_prob                                 # Step 4: scale the value of neurons that haven't been shut down

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    
    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    
    return A3, cache

"""
Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.

Arguments:
X -- input dataset, of shape (2, number of examples)
parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                W1 -- weight matrix of shape (20, 2)
                b1 -- bias vector of shape (20, 1)
                W2 -- weight matrix of shape (3, 20)
                b2 -- bias vector of shape (3, 1)
                W3 -- weight matrix of shape (1, 3)
                b3 -- bias vector of shape (1, 1)
keep_prob - probability of keeping a neuron active during drop-out, scalar

Returns:
A3 -- last activation value, output of the forward propagation, of shape (1,1)
cache -- tuple, information stored for computing the backward propagation
"""


# =================================================================
# =================================================================

def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)
    dA2 = dA2 * D2                     # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
    dA2 = dA2 / keep_prob              # Step 2: Scale the value of neurons that haven't been shut down

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    dA1 = dA1 * D1                      # Step 1: Apply mask D1 to shut down the same neurons as during the forward propagation
    dA1 = dA1 / keep_prob               # Step 2: Scale the value of neurons that haven't been shut down
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients

"""
Implements the backward propagation of our baseline model to which we added dropout.

Arguments:
X -- input dataset, of shape (2, number of examples)
Y -- "true" labels vector, of shape (output size, number of examples)
cache -- cache output from forward_propagation_with_dropout()
keep_prob - probability of keeping a neuron active during drop-out, scalar

Returns:
gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
"""