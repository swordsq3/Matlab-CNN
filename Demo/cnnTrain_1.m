clc; clear;
addpath(genpath('../'));
%% Convolution Neural Network Exercise

% diary on;
% diary('run_log');

% 모델 파일 이름
modelFile = 'trained_model.mat';

%% STEP 1: Initialize Parameters and Load Data
%  complete the config.m to config the network structure;
cnnConfig = config();

%  calling cnnInitParams() to initialize parameters

% cnnInitParams 호출 전 파라미터 초기화 조건 추가
if isfile(modelFile)
    % 저장된 모델 로드
    fprintf('Loading existing model from %s...\n', modelFile);
    load(modelFile, 'opttheta', 'cnnConfig', 'meta');
    theta = opttheta; % 저장된 모델의 최적화된 파라미터를 초기화 변수에 사용
else
    % 새로 파라미터 초기화
    fprintf('No pre-trained model found. Initializing new parameters...\n');
    [theta, meta] = cnnInitParams(cnnConfig);
end

% Load MNIST Data
images = loadMNISTImages('train-images-idx3-ubyte');
d = cnnConfig.layer{1}.dimension;
images = reshape(images,d(1),d(2),d(3),[]);
labels = loadMNISTLabels('train-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

%% STEP 2: Learn Parameters
% Train the model using minFuncSGD
options.epochs = double(1); % 학습 반복 횟수
options.minibatch = double(128); % 미니배치 크기
options.alpha = 1e-1; % 학습률
options.momentum = .95; % 모멘텀

opttheta = minFuncSGD(@(x, y, z) cnnCost(x, y, z, cnnConfig, meta), theta, images, labels, options);

% 학습된 모델 저장
save(modelFile, 'opttheta', 'cnnConfig', 'meta');
fprintf('Model saved to %s\n', modelFile);


%% STEP 3: Validation
% Load MNIST Test Data
d = double(cnnConfig.layer{1}.dimension); % dimension을 double로 변환
testImages = loadMNISTImages('../Dataset/MNIST/t10k-images-idx3-ubyte');
testImages = reshape(testImages, d(1), d(2), d(3), []);
testLabels = loadMNISTLabels('../Dataset/MNIST/t10k-labels-idx1-ubyte');
testLabels(testLabels == 0) = 10; % Remap 0 to 10

% 모델 테스트
[cost, grad, preds] = cnnCost(opttheta, testImages, testLabels, cnnConfig, meta, true);

% 정확도 계산
acc = sum(preds == testLabels) / double(length(preds)); % length(preds)를 double로 변환
fprintf('Accuracy is %f\n', acc);

% diary off;
