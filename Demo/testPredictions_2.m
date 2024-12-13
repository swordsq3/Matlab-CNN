% testPredictions.m
clc; clear;
addpath(genpath('../'));

% 저장된 모델 불러오기
load('trained_model.mat');

% 테스트 데이터 로드
testImages = loadMNISTImages('../Dataset/MNIST/t10k-images-idx3-ubyte');
d = double(cnnConfig.layer{1}.dimension);
testImages = reshape(testImages,d(1),d(2),d(3),[]);
testLabels = loadMNISTLabels('../Dataset/MNIST/t10k-labels-idx1-ubyte');
testLabels(testLabels==0) = 10;
size(testImages)
% 예측
[~,~,preds] = cnnCost(opttheta,testImages,testLabels,cnnConfig,meta,true);

% 랜덤 이미지 10개 선택하여 예측 결과 시각화
figure;
for i = 1:10
    idx = randi(length(testLabels));
    subplot(2,5,i);
    imshow(reshape(testImages(:,:,1,idx), [28 28]), []);
    title(sprintf('Pred: %d, True: %d', preds(idx), testLabels(idx)));
end