function [net, learningParams] = simpleBackPropagation(net, learningParams, dataset)
% function [net, learningParams] = backPropagation(net, learningParams, dataset)
%
% Learning method based on simple Back Propagation algorithm
%
% - net: a FFNN full connected
% - learningParams: parameters needed for the RProp algorithm
% - dataset: struct containing the dataset divided in TS, TS_labels, VS and
%            VS_labels
%
% Author: Andrea Pollastro, MSc student in CS at University of Naples "Federico II"
    %% Starting Back propagation based algorithm
    [net,Y] = forwardPropagation(net, dataset.TS);
    delta_dp = backPropagation(net, Y, dataset.TS_labels);
    [deltaWij,deltaBij] = computeDerivatives(net, delta_dp, dataset.TS);
    net = hebbianRule(net,deltaWij,deltaBij,learningParams.ETA);
end

