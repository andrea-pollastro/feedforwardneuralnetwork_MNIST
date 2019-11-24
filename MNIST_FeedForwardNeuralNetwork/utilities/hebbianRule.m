function net = hebbianRule(net,dW,dB,ETA)
% function net = hebbianRule(net,dW,dB,ETA)
%
% This function updates net's weights and biases using the Hebbian rule
%
% Author: Andrea Pollastro, MSc student in CS at University of Naples "Federico II"

    for layer = 1:net.L
        net.W{layer} = net.W{layer} - ETA*dW{layer};
        net.b{layer} = net.b{layer} - ETA*dB{layer};
    end
end