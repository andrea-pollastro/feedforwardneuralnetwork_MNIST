function visualizeMNIST(images,labels,start)
% function visualizeMNIST(images,labels)
%
% Function of MNIST visualization predictions 
%
% Author: Andrea Pollastro, MSc student in CS at University of Naples "Federico II"
    %% Setting parameters
    totFigures = 36;
    if nargin < 3
       start = 0;
    end
    if start < 0 || start > (size(images,1)-totFigures)
       start = 0; 
    end
    % We've a 6x6 grid, for this reason we can visualize 36 images
    figure
    colormap(gray)
    %% Displaying figures and predictions
    for i = 1 : totFigures
        subplot(6,6,i)
        digit=reshape(images(start+i,:), [28,28]);
        imagesc(digit)
        title(num2str(labels(start+i,1)))
    end
end