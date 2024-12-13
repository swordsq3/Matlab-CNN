function [pooledFeatures, weights] = cnnPool(poolDim, convolvedFeatures, pooltypes)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region, is a 1 * 2 vector(poolDimRow poolDimCol);
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%  weights        - how much the input contributes to the output
%     

if nargin < 3
    pooltypes = 'meanpool';
end

numImages = double(size(convolvedFeatures, 4));
numFilters = double(size(convolvedFeatures, 3));
convolvedDimRow = double(size(convolvedFeatures, 1));
convolvedDimCol = double(size(convolvedFeatures, 2));
poolDim = double([poolDim(1) poolDim(2)]);
pooledDimRow = double(floor(convolvedDimRow / poolDim(1)));
pooledDimCol = double(floor(convolvedDimCol / poolDim(2)));

weights = zeros(size(convolvedFeatures));
featuresTrim = convolvedFeatures(1:pooledDimRow*poolDim(1),1:pooledDimCol*poolDim(2),:,:);

if strcmp(pooltypes, 'meanpool')
    weights(1:pooledDimRow*poolDim(1), 1:pooledDimCol*poolDim(2),:,:) = ones(size(featuresTrim)) / poolDim(1) / poolDim(2);
end

pooledFeatures = zeros(pooledDimRow, pooledDimCol, numFilters, numImages);

poolFilter = ones(poolDim) * 1/poolDim(1)/poolDim(2);

for imageNum = 1:numImages
    for filterNum = 1:numFilters
        features = featuresTrim(:,:,filterNum, imageNum);
        switch pooltypes
            case 'meanpool'
                poolConvolvedFeatures = conv2(features,poolFilter,'valid');
                startRow = 1:double(poolDim(1)):size(poolConvolvedFeatures,1);
                startCol = 1:double(poolDim(2)):size(poolConvolvedFeatures,2);
                pooledFeatures(:,:,filterNum,imageNum) = poolConvolvedFeatures(startRow,startCol);
            case 'maxpool'
                temp = im2col(features, poolDim, 'distinct');
                [m, i] = max(temp);
                temp = zeros(size(temp));
                temp(sub2ind(size(temp),i,1:size(i,2))) = 1;
                weights(1:pooledDimRow*poolDim(1),1:pooledDimCol*poolDim(2),filterNum,imageNum) = col2im(temp, poolDim,[pooledDimRow*poolDim(1) pooledDimCol*poolDim(2)], 'distinct');
                pooledFeatures(:,:,filterNum,imageNum) = reshape(m, size(pooledFeatures,1), size(pooledFeatures,2));         
            otherwise
                error('wrongLayertype: %s',pooltypes);
        end
    end
end
end
 
function b = maxblock(a)
    b = max(max(a.data));
end