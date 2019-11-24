function error = meanSquares(Y,T)
% function error = meanSquares(Y,T)
%
% Mean squares error function
%
% RETURNS:
% error: floating point number
%
% PARAMETERS:
% Y: output, matrix
% T: targets, matrix
%
% Author: Andrea Pollastro, MSc student in CS at University of Naples "Federico II"

    Y = Y-T;
    Y = Y .* Y;
    RES = sum(Y,2);
    RES = RES/2;
    error = sum(RES);
end