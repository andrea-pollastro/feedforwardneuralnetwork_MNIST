function error = crossEntropy(Y,T)
% function error = crossEntropy(Y,T)
%
% Cross entropy error function
%
% RETURNS:
% error: floating point number
%
% PARAMETERS:
% Y: output, matrix
% T: targets, matrix
%
% Author: Andrea Pollastro, MSc student in CS at University of Naples "Federico II"
    error = -sum(sum(T .* log(Y),2));
end