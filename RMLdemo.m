load RML.mat
LabelIndex=gnd;
for i = 1:479
     X_train(i,:) = X(i,:) ./ max(1e-12,norm(X(i,:)));
     Y_train(i,:) = Y(i,:) ./ max(1e-12,norm(Y(i,:)));
end 
%reduce the dimension
PCAoptions.ReducedDim = 200;
[eigvector_PCA, eigvalue_PCA, meanData, new_X] = PCA(X_train,PCAoptions);
[eigvector_PCA, eigvalue_PCA, meanData, new_Y] = PCA(Y_train,PCAoptions);
X=new_X;
Y=new_Y;
% randomly select the train set and test set
for iter=1:10
str=strcat('.\RML\',int2str(iter),'.mat');
load (str)
X_train_v1 = [];
X_train_v2 = [];
X_train_total=[];
X_train_v1 = X(trainIdx,:);
X_train_v2 = Y(trainIdx,:);
X_test_v1 = X(testIdx,:);
X_test_v2 = Y(testIdx,:);

gnd=[];
gnd_test=[];
gnd=LabelIndex(trainIdx);
gnd_test=LabelIndex(testIdx);
X_train_vali{1,1} = X_train_v1';
X_train_vali{2,1} = X_train_v2';
X_train_total = [X_train_v1'; X_train_v2'];
X_test_vali{1,1} = X_test_v1;
X_test_vali{2,1} = X_test_v2;
X_test_vali{3,1} = [X_test_v1 X_test_v2];

options=[];
options.lambda1=0.01; 
options.lambda2=10;
options.lambda3=0.01;
options.alpha=[1e-5,1e-5];

[P, Z, J ,E] = SLRCE(X_train_total,X_train_vali,gnd,options);

X_train_vali_total = [X_train_v1  X_train_v2];
X_train_vali_M = cell(1,1);
X_test_vali_M = cell(1,1);
Wx=P(1:size(X_train_vali{1},1),:);
Wy=P(size(X_train_vali{1},1)+1:end,:);

n=1;
  for ii=2:2:size(P,2)
    W1=Wx(:,1:ii);
    W2=Wy(:,1:ii);
    W_I = [W1 zeros(size(W1,1),size(W2,2));zeros(size(W2,1),size(W1,2)) W2]; 
    X_train_vali_M{1,1} = X_train_vali_total * W_I;
    X_test_vali_M{1,1} = X_test_vali{3} * W_I;
    results_M = knn_classify(X_train_vali_M,X_test_vali_M,1,gnd,1);
    accurate_M(iter,n) = length(find(gnd_test' == results_M{1,1}))/length(gnd_test);
    n=n+1;
  end
end
fprintf('The top recognition accury on RML is %f\n\n',max(mean(accurate_M)));



