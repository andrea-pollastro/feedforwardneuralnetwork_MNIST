function [results] = convertResultsFromDummyvar(Y)
% function [results] = convertResultsFromDummyvar(Y)
%
% This functions converts arrays made with dummyvar function of MATLAB in a
% number.
% For example:
%      x=[0,0,0,0,1;
%         0,1,0,0,0;
%         0,0,0,1,0]
% is translated in
%      x=[5;
%         2;
%         4]
%
% Author: Andrea Pollastro, MSc student in CS at University of Naples "Federico II"

    length = size(Y,1);
    results = zeros(length,1);
    for i=1:size(Y,1)
        [~,index] = max(Y(i,:));
        results(i) = index;
    end
end