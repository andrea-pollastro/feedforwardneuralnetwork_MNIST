function [delta] = backPropagation(net,Y,T)
% function [delta] = backPropagation(net,Y,T)
%
% This function computes the back propagation algorithm
%
% RETURNS:
% delta: used for derivative calculation
%
% PARAMETERS:
% net: feed-forward neural network full connected, struct
% Y: output, matrix
% T: target, matrix
%
% Author: Andrea Pollastro, MSc student in CS at University of Naples "Federico II"

    delta = cell(1,net.L);
    der = getDerivative(net.actFun{net.L}, net.output{net.L});
    delta{net.L} =  der .* (Y-T);
    for layer = (net.L-1):-1:1
        delta{layer} = delta{layer+1}*net.W{layer+1};
        der = getDerivative(net.actFun{layer}, net.output{layer});
        delta{layer} = der .* delta{layer};
    end
end