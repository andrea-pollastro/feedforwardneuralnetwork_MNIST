function net = createFFNN(vectParam, vectFunction)
% function net = createNewNetwork(vectParam, vectFunction)
%
% This function is used to create a new FF NN full connected.
% vectParam and vectFunction must be respectively of dimensions L+1 and L,
% where L is the number of network's layers.
% These activation function are now supported:
% - @sigmoid
% - @relu
% - @identity
%
% RETURNS:
% - net: a FFNN full-connected
%
% PARAMETERS:
% - vectParam:  vector of dimension L+1. The first elements contains the 
%               input dimension (d). The other elements are the number of neurons 
%               for each layer (m)
% - vectFunction: vector of dimension L. It contains the activation function
%                 for each layer.
%
% Author: Andrea Pollastro, MSc student in CS at University of Naples "Federico II"

    %% Checking parameters' correctness
    L = length(vectParam)-1;
    if(L ~= size(vectFunction,2))
        error('Number of layers is different from number of activation functions');
    end
    %% Defining net's parameters
    W = cell(1,L);
    b = cell(1,L);
    inputSize = vectParam(1);
    m1 = inputSize;
    % Weights and biases initialization
    for layer=1:L
        m2 = vectParam(layer+1);
        W{layer} = 0.01*randn(m2,m1);
        b{layer} = 0.01*randn(1,m2);
        m1 = m2;
    end
    %% 'net' structure creation
    net.inputSize = inputSize;
    net.L = L;
    net.W = W;
    net.b = b;
    net.actFun = vectFunction;
    net.act = cell(1,L);
    net.output = cell(1,L);
end