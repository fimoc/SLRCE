% Solve the TraceRatio problems: 
% max_{W'*W=I}   tr(W'*A*W)/tr(W'*B*W)
% min_{W'*W=I}   tr(W'*A*W)/tr(W'*B*W)
% The matrix A should be symmetric and B must be positive semi-definite.
% The cross set of null spaces of A and B should be null.
% If A and B are both positive semi-definite, the matrix A+B should be positive definite.

% VERY IMPORTANT:
% In trace ratio LDA, max_{W'*W=I}   tr(W'*Sb*W)/tr(W'*Sw*W), two important issues should be noted:
% 1, St(=Sb+Sw) should be nonsingular, i.e., PCA should be performed to remove the null space of the 
%    training data (corresponding to the zero eigenvalue of St) before call the function.
% 2, The projected dimension ‘dim’ could be set to 1~size(Sw,1), don't just set it to c-1, where c is 
%    the class number. 2*c or 3*c is recommended.
function [W, obj] = TraceRatio_fast(A, B, dim, isMax)
% Ref: Yangqing Jia, Feiping Nie, Changshui Zhang. Trace Ratio Problem Revisited. 
%      IEEE Transactions on Neural Networks (TNN), Volume 20, Issue 4, Pages 729-735, 2009.
%
% We give a faster algorithm in this paper to solve the Trace Ratio problem.

if nargin < 4
    isMax = 1;
end;

n = size(A,1);
W = eye(n,dim);
ob = trace(W'*A*W)/trace(W'*B*W);

counter = 1;
obd = 1;
while obd > 10^-6 && counter < 20
    M = A - ob*B;  M = max(M,M');
    [v, d] = eig(M);
    d = diag(d);
    if isMax == 1
        [temp, idx] = sort(d, 'descend');
    else
        [temp, idx] = sort(d);
    end;
    [W, idx] = vector_selection(A, B, v, dim, isMax);
    
    obd = abs(sum(temp(1:dim)));
    ob = trace(W'*A*W)/abs(trace(W'*B*W));
    obj(counter) = ob;
    counter = counter + 1;
end;

if counter == 20
    disp('Warnning: the trace ratio does not converge!');
end;



%% Select eigenvectors to maximize the trace ratio objective
function [W, idx] = vector_selection(A, B, v, dim, isMax)

n = size(A,2);
a = diag(v'*A*v);
b = abs(diag(v'*B*v));
idx = n-dim+1:n;
obj = sum(a(idx))/sum(b(idx));

err = 10;
counter = 0;
while err > 0 && counter < 20
    obj0 = obj;
    m = a - obj*b;
    if isMax == 1
        [temp idx] = sort(m, 'descend');
    else
        [temp idx] = sort(m);
    end;
    idx = idx(1:dim);

    obj = sum(a(idx))/sum(b(idx));
    err = abs(obj - obj0);
    counter = counter + 1;
end;

if counter == 20
    disp('error! trace ratio can not converge');
end;

W = v(:,idx);