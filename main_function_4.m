

load('data3.mat');

cvp = cvpartition(y, 'Holdout', 0.2);
training = cvp.training;
test = cvp.test;

train_X = X(training,:); % form the training data
train_y = y(training); 
    
test_X = X(test,:); % form the testing data
test_y = y(test);    
%% 1.a
scatter(X(:,1), X(:,2),2,y);
xlabel('X1');
ylabel('X2');
title('Scatterplot between X1 and X2 colored by class');
%% 1.b
gamma = 2;
model = fitcsvm(train_X, train_y, 'KernelFunction','RBF', 'KernelScale', 1/sqrt(gamma)); 

train_pred_y = predict(model, train_X); 
train_acc = sum(train_y == train_pred_y)/length(train_y);

test_pred_y = predict(model, test_X);
test_acc = sum(test_y == test_pred_y)/length(test_y);

svm_plot_classes(X,y,model); 

clc
sprintf('Train accuracy: %0.4f', train_acc)
sprintf('Test accuracy: %0.4f', test_acc)

%% 1.c
gamma = 1;
model = fitcsvm(train_X, train_y, 'KernelFunction','RBF', 'KernelScale', 1/sqrt(gamma)); 

train_pred_y = predict(model, train_X); 
train_acc = sum(train_y == train_pred_y)/length(train_y);

test_pred_y = predict(model, test_X);
test_acc = sum(test_y == test_pred_y)/length(test_y);

svm_plot_classes(X,y,model); 

clc
sprintf('Train accuracy: %0.4f', train_acc)
sprintf('Test accuracy: %0.4f', test_acc)

%% 1.d
%C=1 as default
gamma = 10;
model = fitcsvm(train_X, train_y, 'KernelFunction','RBF', 'KernelScale', 1/sqrt(gamma)); 

train_pred_y = predict(model, train_X); 
train_acc = sum(train_y == train_pred_y)/length(train_y);

test_pred_y = predict(model, test_X);
test_acc = sum(test_y == test_pred_y)/length(test_y);

svm_plot_classes(X,y,model); 

clc
sprintf('Train accuracy: %0.4f', train_acc)
sprintf('Test accuracy: %0.4f', test_acc)


%% 1.e
clc
C=0.1;
gamma=1;
while C<=1000
    
    model = fitcsvm(train_X, train_y, 'KernelFunction','RBF', 'BoxConstraint', C, 'KernelScale', 1/sqrt(gamma)); 
    
    train_pred_y = predict(model, train_X); 
    train_acc = sum(train_y == train_pred_y)/length(train_y);
    
    test_pred_y = predict(model, test_X);
    test_acc = sum(test_y == test_pred_y)/length(test_y);
    
    sprintf('C=%d Train accuracy: %0.4f; Test accuracy: %0.4f',C, train_acc,test_acc)

    C=C*10;
end


%% 1.f

model = fitcsvm(train_X, train_y, 'KernelFunction', 'polynomial', 'PolynomialOrder', 3); 

train_pred_y = predict(model, train_X); 
train_acc = sum(train_y == train_pred_y)/length(train_y);

test_pred_y = predict(model, test_X);
test_acc = sum(test_y == test_pred_y)/length(test_y);

svm_plot_classes(X,y,model);

clc
sprintf('Train accuracy: %0.4f', train_acc)
sprintf('Test accuracy: %0.4f', test_acc)



%% 2.a
cvp_val = cvpartition(train_y, 'Holdout', 0.2);
training_val = cvp_val.training;
validation = cvp_val.test;

train_X_val = train_X(training_val,:); 
train_y_val = train_y(training_val); 
    
val_X = train_X(validation,:); 
val_y = train_y(validation);   

kernel_scales = [0.01 0.1 1 5 10 50 100 500 1000];
Cs = [0.01 0.1 1 5 10 50 100 500 1000];
N = length(kernel_scales);
M = length(Cs);
acc_s = zeros(N,M);

for sigma_idx = 1:N
    for C_idx = 1:M
        model = fitcsvm(train_X_val, train_y_val, 'KernelFunction','RBF', 'BoxConstraint', Cs(C_idx), 'KernelScale', 1/sqrt(kernel_scales(sigma_idx)));
        val_pred_y = predict(model, val_X);
        val_acc = sum(val_y == val_pred_y)/length(val_y);
        acc_s(sigma_idx, C_idx) = val_acc;
    end
end
%get indices of best accuracy
[max_value,idx]=max(acc_s(:));
[idx_c, idx_s]=ind2sub(size(acc_s),idx);
%get bet parameters
kernel_scale_best = kernel_scales(idx_s);
C_best = Cs(idx_c);
% Use the best parameters to train a classifier with all the data
best_model = fitcsvm(train_X, train_y, 'KernelFunction', 'rbf', 'KernelScale', kernel_scale_best, 'BoxConstraint', C_best);
%predictions
train_pred_y = predict(best_model, train_X_val); 
best_train_acc = sum(train_y_val == train_pred_y)/length(train_y_val);
val_pred_y = predict(best_model, val_X); 
best_val_acc = sum(val_y == val_pred_y)/length(val_y);
test_pred_y = predict(best_model, test_X); 
best_test_acc = sum(test_y == test_pred_y)/length(test_y);
clc
sprintf('Best Train accuracy: %0.4f', best_train_acc)
sprintf('Best Validation accuracy: %0.4f', best_val_acc)
sprintf('Best Test accuracy: %0.4f', best_test_acc)


%% 2.b
svm_plot_classes(train_X_val,train_y_val,best_model); 
svm_plot_classes(val_X,val_y,best_model);
svm_plot_classes(test_X,test_y,best_model);


%% 2.c
clc
%-----------FIRST EXECUTE SECTION WITH THE FUNCTION BELOW-------------
calculateF1Score(train_y_val, train_pred_y)
calculateF1Score(val_y, val_pred_y)
calculateF1Score(test_y, test_pred_y)

%% 3.a
clc
model = fitcsvm(train_X, train_y, 'KernelFunction', 'rbf', 'KernelScale', ...
    5, 'BoxConstraint', 5);
svm_plot_classes(X,y,model);

cvp = cvpartition(y,'KFold',10);
AUC_all = zeros(cvp.NumTestSets,1);
for i = 1:cvp.NumTestSets
    training_cv = cvp.training(i);
    test_cv = cvp.test(i);
    
    train_X_cv = X(training_cv,:); % form the training data
    train_y_cv = y(training_cv); 
    
    test_X_cv = X(test_cv,:); % form the testing data
    test_y_cv = y(test_cv);
    %train
    model = fitcsvm(train_X_cv, train_y_cv, 'KernelFunction', 'rbf', 'KernelScale', ...
    5, 'BoxConstraint', 5);
    test_pred_y = predict(model, test_X_cv);
    
    [~,~,~,AUC_all(i)] = perfcurve(test_y_cv,test_pred_y,2);
end
mean(AUC_all)
figure; boxplot(AUC_all)
title('SVM - AUC')
%% 3.b
clc
model = TreeBagger(20, train_X, train_y,'MinLeafSize', 10);
tree_plot_classes(X,y,model);


cvp = cvpartition(y,'KFold',10);
AUC_all = zeros(cvp.NumTestSets,1);
for i = 1:cvp.NumTestSets
    training_cv = cvp.training(i);
    test_cv = cvp.test(i);
    
    train_X_cv = X(training_cv,:); % form the training data
    train_y_cv = y(training_cv); 
    
    test_X_cv = X(test_cv,:); % form the testing data
    test_y_cv = y(test_cv);
    %train
    model = TreeBagger(20, train_X_cv, train_y_cv,'MinLeafSize', 10);
    test_pred_y = predict(model, test_X_cv);
    
    [~,~,~,AUC_all(i)] = perfcurve(test_y_cv,test_pred_y,2);
end
mean(AUC_all)
figure; boxplot(AUC_all)
title('Random Forest - AUC')

%% 3.c
%%
function f1Score = calculateF1Score(y_true, y_pred)
    % Calculate True Positives (TP)
    TP = sum((y_true == 2) & (y_pred == 2));

    % Calculate False Positives (FP)
    FP = sum((y_true == 1) & (y_pred == 2));

    % Calculate False Negatives (FN)
    FN = sum((y_true == 2) & (y_pred == 1));

    % Calculate Precision
    Precision = TP / (TP + FP);

    % Calculate Recall
    Recall = TP / (TP + FN);

    % Calculate F1 Score
    f1Score = 2 * (Precision * Recall) / (Precision + Recall);

    % Handle the case where the denominator is 0
    if isnan(f1Score)
        f1Score = 0;
    end
end




