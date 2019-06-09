function [P, Z, J,E] = SLRCE(X,Y,gnd,options)
 
[m,n] = size(X); %% m is sum of multi-view data original dimension and n is the size.
Label = unique(gnd);
nLabel = length(Label);

%%initialize projection P
  dump=size(Y{1},1);
  cump=size(Y{2},1);
  Cxx = X(1:dump,:)*(X(1:dump,:))';
  Cyy = X(dump+1:end,:)*(X(dump+1:end,:))';
  Cxy = X(1:dump,:)*(X(dump+1:end,:))';                                                                                                                                                                                                                                            
  Cxx=Cxx+eye(size(Cxx,1)) * 10^(-7);
  Cyy=Cyy+eye(size(Cyy,1)) * 10^(-7);
  SX=blkdiag(Cxx,Cyy);
  SXY=[Cxx Cxy;Cxy' Cyy ];
  SXY=max(SXY,SXY');
  SX =max(SX,SX);
  tol = 1.e-6;
  [P, rho1] = TraceRatio_fast(SXY-SX,eye(size(SX)),200);

%% build the laplacian regulation
   options1 = [];
   options1.Metric = 'Euclidean';
   options1.NeighborMode = 'KNN';
   options1.k = 3;
   options1.WeightMode = 'HeatKernel';
   options1.t = 100000;
   W1 = constructW(Y{1}',options1);
   W2 = constructW(Y{2}',options1);
   L1=diag(sum(W1,2))-W1;
   L2=diag(sum(W2,2))-W2;
   L=L1+L2;
%%
if (~exist('options','var'))
   options = [];
end

lambda1 = 1e-2;
if isfield(options,'lambda1')
    lambda1 = options.lambda1;
end

lambda2 = 0.1;
if isfield(options,'lambda2')
    lambda2 = options.lambda2;
end

lambda3=0.001;
if isfield(options,'lambda3')
    lambda3 = options.lambda3;
end

alpha = 0.001;
if isfield(options,'alpha')
    alpha = options.alpha;
end

maxIter = 50;
if isfield(options,'maxIter')
    maxIter = options.maxIter;
end
tol=1e-4;
%% initilize other parameters
max_mu = 10^6;
mu = 0.1;
rho = 1.5;
%rho = 1.8;
d=size(P,2);
Z = zeros(n,n);
E = zeros(d,n);
J = zeros(n,n);
Y1 = zeros(d,n); %%<P'X - YZ - E>
Y2 = zeros(n,n);  %%<Z-J>
Y3 = zeros(d,d);
for iter = 1: maxIter
    %% update P
    if(iter > 1)
    Ap = (-2*lambda3*(SXY-SX)+mu*(X-X*Z)*(X-X*Z)');
    Bp = (Y3+Y3');
    Cp = (mu*(X-X*Z)*E'-(X-X*Z)*Y1');
    [P, obj]=LquadR21_reg(SX,Ap,Bp,Cp, size(P,2), dump, cump,alpha);
    P=Gram_Schmidt_Orthogonalization(P);
    end
    
    %% update J
        tmp_J = Z + Y2/mu; 
        [U,sigma,V] = svd(tmp_J,'econ');
        sigma = diag(sigma);
        svp = length(find(sigma>1/(mu)));
        if svp>=1
            sigma = sigma(1:svp)-1/(mu);
        else
            svp = 1;
            sigma = 0;
        end
        J = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
        clear U V sigma
    
    %% for Z
    Z1 = 2*lambda2/mu*L+ X'*P*P'*X+eye(n);
    Z2 = J+X'*P*(P'*X-E)+(X'*P*Y1-Y2)/mu;
    Z = Z1\Z2;
    
    %% for E
    tmp_E = P'*(X-X*Z) + Y1/mu;
    E = max(0,tmp_E - lambda1/mu)+min(0,tmp_E + lambda1/mu);  
    
    %% for Y1~Y2, mu
    leq1 = P'*(X - X*Z) - E;
    leq2 = Z-J;
    leq3 = P'*P - eye(size(P'*P,1));

    Y1 = Y1 + mu*leq1;
    Y2 = Y2 + mu*leq2;
    Y3 = Y3 + mu*leq3;
    mu = min(rho*mu, max_mu);   
    
    stopC = max(max(max(abs(leq1))),max(max(abs(leq2))));
%     stopC= max(stopC1,max(max(abs(leq3))));
    if iter==1 || mod(iter,10)==0 || stopC<tol
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',stopC=' num2str(stopC,'%2.3e')]);
    end

    if stopC < tol
        break;
    end
end


