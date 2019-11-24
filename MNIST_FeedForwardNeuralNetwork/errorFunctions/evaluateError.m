function err = evaluateError(fun,Y,T,sizeX)
% function err = evaluateError(fun,Y,T,sizeX)
%
% This function returns the error given the function error, Y, T and the size of X. 
% The current supported functions are:
% - @meanSquares
% - @crossEntropy
%
% RETURNS:
% err: error
%
% PARAMETERS:
% fun: function handler
% Y: output, matrix
% T: target, matrix
% sizeX: size of X, integer
%
% Author: Andrea Pollastro, MSc student in CS at University of Naples "Federico II"
    if(isequal(fun,@meanSquares))
        err = meanSquares(Y,T)/sizeX;
    elseif(isequal(fun,@crossEntropy))
        err = crossEntropy(softmax(Y')',T)/sizeX;
    else
        error('error function not supported yet')
    end
end