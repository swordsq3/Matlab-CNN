clc; clear;
addpath(genpath('../'));

% MNIST 데이터 로드
images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');
labels(labels == 0) = 10; % 0을 10으로 재매핑

% 라벨 히스토그램
figure;
histogram(labels, 1:11, 'FaceColor', 'b', 'EdgeColor', 'k');
xticks(1:10);
xlabel('Labels');
ylabel('Frequency');
title('Histogram of MNIST Labels');

% 랜덤으로 이미지 10개 선택
numImages = 10;
randomIndices = randperm(size(images, 2), numImages); % 10개의 랜덤 인덱스
selectedImages = images(:, randomIndices); % 선택된 이미지
selectedLabels = labels(randomIndices); % 해당 라벨

% 이미지 시각화
figure;
for i = 1:numImages
    subplot(2, 5, i); % 2행 5열로 배치
    img = reshape(selectedImages(:, i), 28, 28); % 28x28로 변환
    imshow(img, []);
    title(sprintf('Label: %d', selectedLabels(i)));
end
sgtitle('Randomly Selected MNIST Images');

