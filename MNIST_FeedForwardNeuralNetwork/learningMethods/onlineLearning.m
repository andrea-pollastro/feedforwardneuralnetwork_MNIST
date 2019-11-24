function [bestNet,res] = onlineLearning(net, learningParams, dataset)
% function [bestNet,res] = onlineLearning(net, learningParams, dataset)
%
% This function returns a FFNN full-connected made trained using online
% learning.
%
% RETURNS:
% - bestNet: NN feed-forward full connected
% - res: struct containing output's result:
%   - error: training set errors
%   - VS_error: validation set errors
%   - VS_error_min: minimum error on validation set
%   - epoch: total number of epochs
%
% PARAMETERS:
% - net: a FFNN full-connected, struct
% - learningParams: a struct containing params needed for the learning
%                   algorithm used
% - dataset: a struct containing dataset elements:
%   - TS: Training set, MUST BE Nxd
%   - TS_labels: Training set's labels, MUST BE Nxc
%   - VS: Validation set, MUST BE Nxd
%   - VS_labels: Validation set's labels, MUST BE Nxc
%
% Author: Andrea Pollastro, MSc student in CS at University of Naples "Federico II"
    
    %% Setting script's parameters
    % Hyperparameters
    errorFunction = learningParams.errorFunction;
    threshold = learningParams.threshold;
    maxEpoch = learningParams.maxEpoch;
    % Dataset parameters
    TS = dataset.TS;
    TS_labels = dataset.TS_labels;
    VS = dataset.VS;
    VS_labels = dataset.VS_labels;
    %% Data structures definition
    % Errors' storage
    error = zeros(1,maxEpoch);
    VS_error = zeros(1,maxEpoch);
    VS_error_min = intmax;
    
    %% Learning Phase
    bestNet = net;
    stop = false;
    epoch = 1;
    while((epoch <= maxEpoch) && (stop == false))
        for n=1:size(TS,1)
            dataset.TS = TS(n,:);
            dataset.TS_labels = TS_labels(n,:);
            learningParams.epoch = epoch;
            [net, learningParams] = learningParams.learningAlgorithm(net, learningParams, dataset);
%             [net,Y] = forwardPropagation(net,TS(n,:));
%             [delta] = backPropagation(net,Y,TS_labels(n,:));
%             [deltaWij,deltaBij] = computeDerivatives(net,delta,TS(n,:));
%             net = hebbianRule(net,deltaWij,deltaBij,ETA);
        end
        %% Computing error on training set
        [net,Y] = forwardPropagation(net,TS);
        error(epoch) = evaluateError(errorFunction,Y,TS_labels,size(TS,1));
        %% Computing error on evaluation set
        [net,Y_val] = forwardPropagation(net,VS);
        VS_error(epoch) = evaluateError(errorFunction,Y_val,VS_labels,size(VS,1));
        %% Storing best net
        if(VS_error(epoch) < VS_error_min)
            VS_error_min = VS_error(epoch);
            bestNet = net;
        end
        %% Stop criteria
        if(abs(100*((VS_error_min/VS_error(epoch))-1)) > threshold)
            stop = true;
        end
        %% Displaying current data
        temporaryRes = sprintf('epoch: %d, err_val_min: %e, err_val: %f', epoch, VS_error_min, VS_error(epoch));
        disp(temporaryRes)
        %% Updating epoch value
        epoch = epoch +1;
    end
    
    %% Creating output's structure
    res.error = error;
    res.VS_error = VS_error;
    res.VS_error_min = VS_error_min;
    res.epoch = epoch - 1;
end