clear all
close all
clc
%% Loading path
loadPath;

%% Importing MNIST dataset
%%%% PLEASE DOWNLOAD train-images.idx3-ubyte from http://yann.lecun.com/exdb/mnist/ %%%%
data = loadMNISTImages('train-images.idx3-ubyte')';
labels = loadMNISTLabels('train-labels.idx1-ubyte');
labels(labels == 0) = 10;

%% dataSet structure definition
% Training Set
dataset.TS = data(1:5000,:);
dataset.TS_labels = dummyvar(labels(1:5000));
% Validation set
dataset.VS = data(50000:51000,:);
dataset.VS_labels = dummyvar(labels(50000:51000));

%% FFNN params definition
vectParam = [size(dataset.TS,2),50,10];
vectFunction = {@sigmoid,@identity};

%% FFNN creation
net = createFFNN(vectParam, vectFunction);

%% Learning params definition
learningParams.errorFunction = @crossEntropy;
learningParams.ETA = 0.0000001;
learningParams.threshold = 5;
learningParams.maxEpoch = 50;
learningParams.learningAlgorithm = @simpleBackPropagation;

%% START LEARNING
[net,res] = onlineLearning(net, learningParams, dataset);

%% Plotting error
hold on
    plot(res.error,'-');  
    plot(res.VS_error, 'r-');
    xlabel('epoch');    ylabel('error');
    legend('Training set error','Validation set error')
hold off

%% Test set definition
TS = loadMNISTImages('t10k-images.idx3-ubyte')';
TS_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
%% Testing net using test set
[net,Y] = forwardPropagation(net,TS);
Y = convertResultsFromDummyvar(Y);
Y(Y == 10) = 0;

%% Computing accuracy
acc = accuracy(Y, TS_labels);
%% Displaying results
temporaryRes = sprintf('accuracy: %.4f\tVS min error: %.12f\ttot epoch: %d', acc, res.VS_error_min , res.epoch);
disp(temporaryRes)

%% Visualizing preditions
visualizeMNIST(TS,Y,1);

%% RESTORING PATH
restoredefaultpath
