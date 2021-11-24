function y = sto(a,k)
%STO Summary of this function goes here
%   Detailed explanation goes here
y = sign(a).*max(0,abs(a)-k);
end

