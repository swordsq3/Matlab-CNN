clc; clear;
addpath(genpath('../'));

% 저장된 모델 불러오기
load('trained_model.mat'); % trained_model.mat 파일 필요

% 테스트 데이터 디렉토리
testDir = './test/';
imageFiles = dir(fullfile(testDir, '*.png'));
imageFiles = [imageFiles; dir(fullfile(testDir, '*.jpg'))];

% 테스트할 이미지가 없는 경우 경고 표시
if isempty(imageFiles)
    error('테스트 데이터 폴더(%s)에 PNG 또는 JPG 파일이 없습니다.', testDir);
end

% 네트워크 입력 크기 확인
expectedInputSize = cnnConfig.layer{1}.dimension; % [28, 28, 1]
disp(['Expected Input Size: ', mat2str(expectedInputSize)]);

% 결과 표시를 위한 figure 생성
figure('Position', [100 100 1200 800]); % 큰 창 설정
a = zeros(28,28,1,length(imageFiles)+1);
actualLabel = zeros(1,length(imageFiles)+1);

% 이미지 처리 및 예측
for i = 1:length(imageFiles)
    % 원본 이미지 경로 및 실제 레이블 추출
    imgPath = fullfile(testDir, imageFiles(i).name);
    actualLabel(i) = str2double(regexprep(imageFiles(i).name, '\D', '')); % 파일 이름에서 숫자 추출
    
    % 이미지 읽기

    originalImg = imread(imgPath);
    
    % 이미지 전처리
    if size(originalImg, 3) == 3
        img = rgb2gray(originalImg); % RGB -> 그레이스케일 변환
    else
        img = originalImg;           % 흑백 이미지 그대로 사용
    end

    img = imresize(img, [expectedInputSize(1), expectedInputSize(2)]); % 크기 맞춤
    if mean(img(:)) > 128             % 평균 밝기 기준 반전
        img = 255 - img;
    end
   
    processedImg = double(img) / 255; % 정규화
    processedImg = reshape(processedImg, size(processedImg, 1), size(processedImg, 2), 1);
    a(:,:,1,i) = processedImg;
    size(a)

    % % 결과 시각화
    subplot(length(imageFiles),2,2*i-1);
    imshow(originalImg, []); % 원본 이미지 표시
    subplot(length(imageFiles),2,2*i);
     
    imshow(reshape(processedImg, [28 28]), []);
    
    [~,~,predictedLabel] = cnnCost(opttheta, a, [], cnnConfig, meta, true);
    % 파일 이름에서 숫자만 추출

    title(sprintf('True: %d, Pred: %d', actualLabel(i), predictedLabel(i)), 'FontSize', 40);
    % % 
end
