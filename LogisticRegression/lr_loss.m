%% auxiliary function 
function out = lr_loss(y,model,A,b)
[m,~] = size(A);
Ay = A*y;
Atran = A';
expba = exp(- b.*Ay);
if model==1
    out = sum(log(1 + expba))/m;
else
    out = Atran*(b./(1+expba) - b)/m;
end
end