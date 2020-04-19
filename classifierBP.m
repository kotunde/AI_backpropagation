%classification of handwritten digits

function classifier()
    clear all; 
    close all; 
    clc;

    % number of images
    N = 3000;

    % defining activation function & its derivative
    theta = @tansig;
    dtheta = @(x) 1-theta(x).*theta(x);
    
    [xTrain, yTrain, xTest, yTest] = loadData();
    
    
     %imagesc(reshape(xTrain(1,:),28,28)');
     %yTrain(1)
    
    % injecting bias
    xTrain = [ones(N,1) xTrain];
    
    % number of neurons in the consecutive layers
    layers = [785 12 12 1];

    W = genbackprop(xTrain, yTrain, layers, theta, dtheta, 1000, 0.01, 0.00003);
    
    yPred = {xTest};
    V = {xTest};

    for j = 1:size(X,1) 
        [yPred, V] = forwardprop(xTest(j,:)', W, theta); 
    end
    
    C = confusionmat(yTest, yPred)
    
end

%%
function [xTrain, yTrain, xTest, yTest] = loadData()
    %load test data
    train_data = load('mnist_train.csv');
    train_images = train_data(:,2:785);
    train_labels = train_data(:,1);
    [rows, cols] = size(train_data);
    
    % store images matrix-like format
%     xTrain = [];
%     
%     for i=1:rows
%         xTrain = [xTrain; reshape(train_images(i,:),28,28)];
%     end

    %store images in a row
    xTrain = train_images;

    %store labels as a column vector
    yTrain = train_labels;
    
    %load train data
    test_data = load('mnist_test.csv');
    test_images = test_data(:,2:785);
    test_labels = test_data(:,1);
    
    [rows, cols] = size(test_data);
    
    % store images matrix-like format
%     xTest = [];
%     
%     for i=1:rows
%         xTest = [xTest; reshape(test_images(i,:),28,28)];
%     end
    
    %store images in a row
    xTest = test_images;
    
    %store labels as a column vector
    yTest = test_labels;
    
    %imageTest = images(p(n+1:end),:);
end
