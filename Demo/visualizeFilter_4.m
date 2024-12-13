% visualizeFiltersWithImageLarge.m
clc; clear;
addpath(genpath('../'));

% 저장된 모델 불러오기
load('trained_model.mat');

% 필터 시각화를 위한 초기화
theta_stack = thetaChange(opttheta, meta, 'vec2stack', cnnConfig);
W = theta_stack{2}.W;  % 첫 번째 conv 레이어의 가중치

% 샘플 입력 이미지 로드 (MNIST 데이터에서 첫 번째 이미지 사용)
images = loadMNISTImages('../Dataset/MNIST/train-images-idx3-ubyte');
sampleImage = reshape(images(:,1), cnnConfig.layer{1}.dimension(1:2)); % 28x28로 변환

% 시각화를 위한 figure 생성
numFilters = size(W, 4); % 필터 개수
figure('Position', [100, 100, 1200, 800]); % 창 크기 확대

% 4열로 배치
numRows = ceil((numFilters + 1) / 4); % 전체 이미지를 4열로 정렬
subplotIndex = 1;

% 원본 이미지 표시
subplot(numRows, 4, subplotIndex);
imshow(sampleImage, []);
title('Original Image');
subplotIndex = subplotIndex + 1;

% 필터 적용 및 시각화
for i = 1:numFilters
    % i번째 필터 적용
    filter = W(:, :, 1, i); % 첫 번째 채널 필터 추출
    filteredImage = conv2(sampleImage, filter, 'valid'); % 유효한 영역만 컨볼루션

    % 필터 적용 결과 시각화
    subplot(numRows, 4, subplotIndex);
    imshow(filteredImage, []);
    title(sprintf('Filter %d', i));
    subplotIndex = subplotIndex + 1;
end

sgtitle('Step-by-Step Filter Visualization');
