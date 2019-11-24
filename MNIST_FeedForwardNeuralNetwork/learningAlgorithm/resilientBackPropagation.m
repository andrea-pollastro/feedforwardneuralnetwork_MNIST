function [net, learningParams] = resilientBackPropagation(net, learningParams, dataset)
% function [net, learningParams] = resilientBackPropagation(net, learningParams, dataset)
%
% Learning method based on Resilient Back Propagation algorithm
%
% - net: a FFNN full connected
% - learningParams: parameters needed for the RProp algorithm
% - dataset: struct containing the dataset divided in TS, TS_labels, VS and
%            VS_labels
%
% Author: Andrea Pollastro, MSc student in CS at University of Naples "Federico II"
    %% Setting function's parameters
    if isfield(learningParams,'d')
        d = learningParams.d;
    end
    delta = learningParams.delta;
    %% Starting RProp algorithm
    [net,Y] = forwardPropagation(net, dataset.TS);
    delta_bp = backPropagation(net, Y, dataset.TS_labels);
    [d.dW,d.dB] = computeDerivatives(net, delta_bp, dataset.TS);
    % During the first epoch we can't use the RProp method. For this
    % reason we must apply the simple Back Propagation method
    if(learningParams.epoch == 1)
        net = hebbianRule(net,d.dW,d.dB,learningParams.ETA);
        d.dW_prev = d.dW;
        d.dB_prev = d.dB;
    else
        [net,delta,d] = rprop(delta,d,learningParams.ETA_MINUS,learningParams.ETA_PLUS,net);
    end
    %% Storing new values
    learningParams.delta = delta;
    learningParams.d = d;
end

function [net,delta,d] = rprop(delta,d,ETA_MINUS,ETA_PLUS,net)
% function [net,delta,d] = rprop(delta,d,ETA_MINUS,ETA_PLUS,net)
%
% Resilient Back Propagation
%
% RETURNS:
% - net: a FFNN full connected
% - d: struct containing derivatives:
%    - dW_prev: weights' derivatives at step (t-1)
%    - dB_prev: biases' derivatives at step (t-1)
%    - dW: weights' derivatives at step (t)
%    - dB: biases' derivatives at step (t)
% - delta: struct containing delta values:
%    - deltaWij: deltaij steps for weights
%    - deltaBij: deltaij steps for biases
%
% PARAMETERS:
% - delta: struct containing delta values:
%    - deltaWij: deltaij steps for weights
%    - deltaBij: deltaij steps for biases
% - d: struct containing derivatives:
%    - dW_prev: weights' derivatives at step (t-1)
%    - dB_prev: biases' derivatives at step (t-1)
%    - dW: weights' derivatives at step (t)
%    - dB: biases' derivatives at step (t)
% - ETA_MINUS: eta-
% - ETA_PLUS: eta+
% - net: a FFNN full connected

    %% Setting RProp parameters
    deltaWij = delta.deltaWij;
    deltaBij = delta.deltaBij;
    dW = d.dW;
    dB = d.dB;
    dW_prev = d.dW_prev;
    dB_prev = d.dB_prev;
    % Max step
    updateMAX = 50;
    % Min step
    updateMIN = 1e-6;
    %% Computing RProp
    for layer=1:net.L
        % Matrix multiplication between derivatives at step (t) and step (t-1)
        derWp = dW{layer}.*dW_prev{layer};
        derBp = dB{layer}.*dB_prev{layer};
        % Updating of deltaij steps
        deltaWij{layer}(derWp > 0) = min(deltaWij{layer}(derWp > 0)*ETA_PLUS, updateMAX);
        deltaBij{layer}(derBp > 0) = min(deltaBij{layer}(derBp > 0)*ETA_PLUS, updateMAX);
        deltaWij{layer}(derWp < 0) = max(deltaWij{layer}(derWp < 0)*ETA_MINUS, updateMIN);
        deltaBij{layer}(derBp < 0) = max(deltaBij{layer}(derBp < 0)*ETA_MINUS, updateMIN);
        % Setting deltaW (deltaB) values    
        deltaW = -sign(dW{layer}).*deltaWij{layer};
        deltaB = -sign(dB{layer}).*deltaBij{layer};
        % Updating weights and biases
        net.W{layer} = net.W{layer} + deltaW;
        net.b{layer} = net.b{layer} + deltaB;
        % Setting new dW_prev
        dW_prev{layer} = dW{layer};
        dB_prev{layer} = dB{layer};
    end
    %% Updating output's structures
    delta.deltaWij = deltaWij;
    delta.deltaBij = deltaBij;
    d.derW = dW;
    d.derB = dB;
    d.dW_prev = dW_prev;
    d.dB_prev = dB_prev;
end