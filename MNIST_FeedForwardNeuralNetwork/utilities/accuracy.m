function acc = accuracy(Y,T)
%function acc = accuracy(Y,T)
%
% Given the output and the targets, this functions calculates the accuracy
% of the prediction
%
% Author: Andrea Pollastro, MSc student in CS at University of Naples "Federico II"

    if(size(Y,1) ~= size(T,1))
        error('cannot calculate accuracy: different sizes')
    end
    acc = (sum((Y==T)))/size(T,1);
end