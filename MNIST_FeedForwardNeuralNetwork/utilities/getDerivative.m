function der = getDerivative(fun,output)
% function der = getDerivative(fun,output)
%
% This function returns the derivative given the output and the activation
% function. If the derivative is not implemented, the function raises an
% error. The current supported functions are:
% - @sigmoid
% - @relu
% - @identity
%
% RETURNS:
% der: derivative
%
% PARAMETERS:
% fun: function handler
% output: output, matrix
%
% Author: Andrea Pollastro, MSc student in CS at University of Naples "Federico II"

    if(isequal(fun,@sigmoid))
        der = output .* (1-output);
    elseif(isequal(fun,@identity))
        der = 1;
    elseif(isequal(fun,@relu))
        der = double(output > 0);
    else
        error('activation function derivative not supported yet')
    end
end