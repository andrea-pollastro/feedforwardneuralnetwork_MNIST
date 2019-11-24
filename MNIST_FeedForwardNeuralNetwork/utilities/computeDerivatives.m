function [dW,dB] = computeDerivatives(net,delta,X)
% function [dW,dB] = computeDerivatives(net,delta,X)
%
% This functions calculates the derivatives using the delta values computed 
% from back propagation and network's output for any layer.
%
% RETURNS:
% derW: weights' derivatives
% derB: biases' derivatives
%
% PARAMETERS:
% net: NN feed-forward full connected, struct
% delta: delta, cell array
% X: input
%
% Author: Andrea Pollastro, MSc student in CS at University of Naples "Federico II"

    Z = X;
    %% Data structure creation
    dW = cell(1,net.L);
    dB = cell(1,net.L);
    %% Computing derivatives
    for layer = 1:net.L
        dW{layer} = delta{layer}'*Z;
        dB{layer} = sum(delta{layer});
        % derivatives will never be due to the machine epsilon. For this
        % reason we make the following assignment
        dW{layer}(abs(dW{layer}) < 1e-6) = 0; 
        dB{layer}(abs(dB{layer}) < 1e-6) = 0;
        Z = net.output{layer};
    end
end