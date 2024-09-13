%% auxiliary function 
function out = nbc_loss(y,model,A,b)
[m,~] = size(A);
Ay = A*y;
Atran = A';
bAy = b.*Ay;
if model==1
    out = sum(1-tanh(bAy))/m;
else
    out = Atran*(b.*((tanh(bAy)).^2-1))/m;
end
end