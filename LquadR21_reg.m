function [X, obj]=LquadR21_reg1(SX,AA, BB,CC,d,dump, cump, lamba3)
NIter = 5;
[m n] = size(AA);
if nargin < 10
    d1 = ones(dump,1);
    d2 = ones(cump,1);
else
    Xi = sqrt(sum(X0.*X0,2)+eps);
    d = 0.5./(Xi);
end;

for iter = 1:NIter
    D1 = diag(d1);
    D2 = diag(d2);
    M = AA+2*blkdiag(lamba3(1)*D1,lamba3(2)*D2);
    X = sylvester(M,BB,CC); 
    Xi1 = sqrt(sum(X(1:dump,:).*X(1:dump,:),2)+eps);
    Xi2 = sqrt(sum(X(dump+1:end,:).*X(dump+1:end,:),2)+eps);
    d1 = 0.5./(Xi1);
    d2 = 0.5./(Xi2);
obj=1;
end;

