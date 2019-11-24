function [net,Y]=forwardPropagation(net,X)
% function [net,Y]=forwardPropagation(net,X)
%
% Forward propagation function
%
% RETURNS:
% Y: output, matrix
% net: a FFNN full connected
%
% PARAMETERS:
% net = a FFNN full connected
% X = Nxd matrix
%
% Author: Andrea Pollastro, MSc student in CS at University of Naples "Federico II"

    N = size(X,1);
    Y = X;
    for layer=1:net.L
        a = Y*net.W{layer}' + repmat(net.b{layer},N,1);
        net.act{layer} = a;
        Y = net.actFun{layer}(a);
        net.output{layer} = Y;
    end
end