function results = knn_classify(train_data, test_data, num_neighbors, train_labels,nV)
% classify test_data using KNN classifier established from train_data
% Input:
%       train_data : multiview train data, each view corresponds to one cell
%       test_data  : multiview test data, cell, each row in an instance
%       num_neighbors  : number of nearest neighbors
%       train_labels  : training labels, TRAIN_NUM * 1 matrix
% Output:
%       results  : labels for testing data, 1 * TEST_NUM  array

n_classify = length(train_data);
class_num = length(unique(train_labels));
%label_vote = zeros(size(test_data{1,1},1),class_num); 
cnt_label = cell(n_classify,1);
maxv = cell(n_classify,1);
results = cell(n_classify,1);

for i = 1:nV
    [train_num(i) D] = size(train_data{i,1});
    [test_num(i) D2] = size(test_data{i,1});
    if D ~= D2
        error('Invalid Data');
    end
    cnt_label{i,1} = zeros(test_num(i),class_num);
    for j = 1:test_num(i)
        [dists{i,1}, neighbors{i,1}] = find_top_K_neighbors(train_data{i,1}, test_data{i,1}(j,:), num_neighbors);
        cnt_label{i,1}(j,:) = recog(train_labels(neighbors{i,1}),class_num);
    end
    [maxv{i,1} results{i,1}] = max(cnt_label{i,1}');
    %label_vote = label_vote + cnt_label{i,1}; %if weight is needed in this multiview classification, it can be added before cnt_labels{i,1}
end
for i = nV+1:n_classify
    [train_num(i) D] = size(train_data{i,1});
    [test_num(i) D2] = size(test_data{i,1});
    if D ~= D2
        error('Invalid Data');
    end
    cnt_label{i,1} = zeros(test_num(i),class_num);
    for j = 1:test_num(i)
        [dists{i,1}, neighbors{i,1}] = find_top_K_neighbors(train_data{i,1}, test_data{i,1}(j,:), num_neighbors);
        cnt_label{i,1}(j,:) = recog(train_labels(neighbors{i,1}),class_num);
    end
    [maxv{i,1} results{i,1}] = max(cnt_label{i,1}');
end

%[maxv{nV+1} results{nV+1,1}] = max(label_vote');





%------------------------------subfunction--------------------------------%
function [dists, neighbors] = find_top_K_neighbors(train_data, test_sample, K)
% find the top K nearest neighbors in the train_data for test_sample
% Input:
%       train_data      -       training data, N x D matrix
%       test_sample     -       test_sample, 1 x D array
%       K               -       number of neighbors
% Output:
%       dist            -       least K distance
%       neighbors       -       K nearest neighbors, 1 x K array

[N, D] = size(train_data);
[dummy, D2] = size(test_sample);

if D ~= D2
    error('Invalid Data...');
end

test_matrix = repmat(test_sample, N, 1);        % N by D
dist_mat = (train_data - test_matrix) .^ 2;     % N by D
dist_array = sum(dist_mat');                    % 1 by N
[dists, neighbors] = sort(dist_array);
dists = dists(1:K);
neighbors = neighbors(1:K);


%--------------------------------subfunction------------------------------%
function cnt_label = recog(neighbor_labels, num_class)
% find the label for the current test sample with neighbor_labels
% Input:
%       neighbor_labels         -       labels for K neighors of the current 
%                                       test sample
%       num_class               -       number of class
% Output:
%       label                   -       label for the current test sample

num_neighbors = length(neighbor_labels);
cnt_label = zeros(1, num_class);

for i = 1:num_neighbors
    cnt_label(neighbor_labels(i)) = cnt_label(neighbor_labels(i)) + 1;
end

%[dummy, label] = max(cnt_labels);