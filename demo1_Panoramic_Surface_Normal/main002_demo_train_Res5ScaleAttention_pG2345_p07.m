%% add path and setup configuration
clc; clear; 
addpath(genpath('./scriptFolder'));
addpath(genpath('../libs'));
path_to_matconvnet = '../matconvnet-1.0-beta23_modifiedDagnn';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));
%% read matconvnet model
% set GPU
gpuId = 1;
gpuDevice(gpuId);

saveFolder = 'main006normal_Res5ScaleAttention_pG2345_p07';
modelName = 'softmax_net-epoch-140.mat';

netbasemodel = load( fullfile('./model', saveFolder, modelName) );
netbasemodel = netbasemodel.netbasemodel;
netbasemodel = dagnn.DagNN.loadobj(netbasemodel);
%% load imdb and dataset file
load('imdb.mat');
imdb.val_normal = imdb.val_localNormal;
imdb.train_normal = imdb.train_localNormal;
%% adjust the model
scalingFactor = 1;
trainAugmentation = true;
netbasemodel.meta.normalization.averageImage = reshape([123.68, 116.779,  103.939],[1,1,3]); % imagenet mean values
netbasemodel.meta.normalization.imageSize = [imdb.meta.height, imdb.meta.width, 3, 1];
netbasemodel.meta.normalization.border = [0, 768]; % 704x1280
netbasemodel.meta.normalization.stepSize = [1, 384];
%% modify the architecture
lossCellList = {'obj_normalEst', 1, 'obj_normalEst_acosLoss', 1};
% resX = 'res5';
% inDim = 512;
% outDim = 512;
% netbasemodel = insertPyramidPool(netbasemodel, resX, inDim, outDim);

sparsityLayerList = {};
lambda = 0.7;
withLossFlag = true;
sparsityLossWeight = 0.00001;
%% insert pixel-gate on resBlock2
%{
rootLayerName = 'pool1';
GateLayerNames = {'res2_1_1relu', 'res2_1_2relu'};
topLayerName = 'res2_1_3conv';
baseName = 'res2_1';
inputDim = 64;
channelIndex = 2;
stride = 1;
[netbasemodel, lossName]= insertPixelGate_withMildLoss(netbasemodel, ...
    rootLayerName, GateLayerNames, baseName, topLayerName, inputDim, lambda, channelIndex, withLossFlag, stride);
sparsityLayerList{end+1} = lossName;
if withLossFlag
    lossCellList{end+1} = lossName;
    lossCellList{end+1} = sparsityLossWeight;
end

rootLayerName = 'res2_1_relu';
GateLayerNames = {'res2_2_1relu', 'res2_2_2relu'};
topLayerName = 'res2_2_3conv';
baseName = 'res2_2';
inputDim = 256;
channelIndex = 2;
[netbasemodel, lossName]= insertPixelGate_withMildLoss(netbasemodel, ...
    rootLayerName, GateLayerNames, baseName, topLayerName, inputDim, lambda, channelIndex, withLossFlag);
sparsityLayerList{end+1} = lossName;
if withLossFlag
    lossCellList{end+1} = lossName;
    lossCellList{end+1} = sparsityLossWeight;
end


rootLayerName = 'res2_2_relu';
GateLayerNames = {'res2_3_1relu', 'res2_3_2relu'};
topLayerName = 'res2_3_3conv';
baseName = 'res2_3';
inputDim = 256;
channelIndex = 2;
[netbasemodel, lossName]= insertPixelGate_withMildLoss(netbasemodel, ...
    rootLayerName, GateLayerNames, baseName, topLayerName, inputDim, lambda, channelIndex, withLossFlag);
sparsityLayerList{end+1} = lossName;
if withLossFlag
    lossCellList{end+1} = lossName;
    lossCellList{end+1} = sparsityLossWeight*100;
end
%}
%% insert pixel-gate on resBlock3
%{
rootLayerName = 'res2_3_relu';
GateLayerNames = {'res3_1_1relu', 'res3_1_2relu'};
topLayerName = 'res3_1_3conv';
baseName = 'res3_1';
inputDim = 256;
channelIndex = 2;
stride = 2;
[netbasemodel, lossName]= insertPixelGate_withMildLoss(netbasemodel, ...
    rootLayerName, GateLayerNames, baseName, topLayerName, inputDim, lambda, channelIndex, withLossFlag, stride);
sparsityLayerList{end+1} = lossName;
if withLossFlag
    lossCellList{end+1} = lossName;
    lossCellList{end+1} = sparsityLossWeight;
end

rootLayerName = 'res3_1_relu';
GateLayerNames = {'res3_2_1relu', 'res3_2_2relu'};
topLayerName = 'res3_2_3conv';
baseName = 'res3_2';
inputDim = 512;
channelIndex = 2;
[netbasemodel, lossName]= insertPixelGate_withMildLoss(netbasemodel, ...
    rootLayerName, GateLayerNames, baseName, topLayerName, inputDim, lambda, channelIndex, withLossFlag);
sparsityLayerList{end+1} = lossName;
if withLossFlag
    lossCellList{end+1} = lossName;
    lossCellList{end+1} = sparsityLossWeight;
end

rootLayerName = 'res3_2_relu';
GateLayerNames = {'res3_3_1relu', 'res3_3_2relu'};
topLayerName = 'res3_3_3conv';
baseName = 'res3_3';
inputDim = 512;
channelIndex = 2;
[netbasemodel, lossName]= insertPixelGate_withMildLoss(netbasemodel, ...
    rootLayerName, GateLayerNames, baseName, topLayerName, inputDim, lambda, channelIndex, withLossFlag);
sparsityLayerList{end+1} = lossName;
if withLossFlag
    lossCellList{end+1} = lossName;
    lossCellList{end+1} = sparsityLossWeight;
end

rootLayerName = 'res3_3_relu';
GateLayerNames = {'res3_4_1relu', 'res3_4_2relu'};
topLayerName = 'res3_4_3conv';
baseName = 'res3_4';
inputDim = 512;
channelIndex = 2;
[netbasemodel, lossName]= insertPixelGate_withMildLoss(netbasemodel, ...
    rootLayerName, GateLayerNames, baseName, topLayerName, inputDim, lambda, channelIndex, withLossFlag);
sparsityLayerList{end+1} = lossName;
if withLossFlag
    lossCellList{end+1} = lossName;
    lossCellList{end+1} = sparsityLossWeight;
end
%}
%% insert pixel-gate on resBlock4
%{
rootLayerName = 'res3_4_relu';
GateLayerNames = {'res4_1_1relu', 'res4_1_2relu'};
topLayerName = 'res4_1_3conv';
baseName = 'res4_1';
inputDim = 512;
channelIndex = 2;
[netbasemodel, lossName]= insertPixelGate_withMildLoss(netbasemodel, ...
    rootLayerName, GateLayerNames, baseName, topLayerName, inputDim, lambda, channelIndex, withLossFlag);
sparsityLayerList{end+1} = lossName;
if withLossFlag
    lossCellList{end+1} = lossName;
    lossCellList{end+1} = sparsityLossWeight;
end

rootLayerName = 'res4_1_relu';
GateLayerNames = {'res4_2_1relu', 'res4_2_2relu'};
topLayerName = 'res4_2_3conv';
baseName = 'res4_2';
inputDim = 1024;
channelIndex = 2;
[netbasemodel, lossName]= insertPixelGate_withMildLoss(netbasemodel, ...
    rootLayerName, GateLayerNames, baseName, topLayerName, inputDim, lambda, channelIndex, withLossFlag);
sparsityLayerList{end+1} = lossName;
if withLossFlag
    lossCellList{end+1} = lossName;
    lossCellList{end+1} = sparsityLossWeight;
end

rootLayerName = 'res4_2_relu';
GateLayerNames = {'res4_3_1relu', 'res4_3_2relu'};
topLayerName = 'res4_3_3conv';
baseName = 'res4_3';
inputDim = 1024;
channelIndex = 2;
[netbasemodel, lossName]= insertPixelGate_withMildLoss(netbasemodel, ...
    rootLayerName, GateLayerNames, baseName, topLayerName, inputDim, lambda, channelIndex, withLossFlag);
sparsityLayerList{end+1} = lossName;
if withLossFlag
    lossCellList{end+1} = lossName;
    lossCellList{end+1} = sparsityLossWeight;
end

rootLayerName = 'res4_3_relu';
GateLayerNames = {'res4_4_1relu', 'res4_4_2relu'};
topLayerName = 'res4_4_3conv';
baseName = 'res4_4';
inputDim = 1024;
channelIndex = 2;
[netbasemodel, lossName]= insertPixelGate_withMildLoss(netbasemodel, ...
    rootLayerName, GateLayerNames, baseName, topLayerName, inputDim, lambda, channelIndex, withLossFlag);
sparsityLayerList{end+1} = lossName;
if withLossFlag
    lossCellList{end+1} = lossName;
    lossCellList{end+1} = sparsityLossWeight;
end

rootLayerName = 'res4_4_relu';
GateLayerNames = {'res4_5_1relu', 'res4_5_2relu'};
topLayerName = 'res4_5_3conv';
baseName = 'res4_5';
inputDim = 1024;
channelIndex = 2;
[netbasemodel, lossName]= insertPixelGate_withMildLoss(netbasemodel, ...
    rootLayerName, GateLayerNames, baseName, topLayerName, inputDim, lambda, channelIndex, withLossFlag);
sparsityLayerList{end+1} = lossName;
if withLossFlag
    lossCellList{end+1} = lossName;
    lossCellList{end+1} = sparsityLossWeight;
end

rootLayerName = 'res4_5_relu';
GateLayerNames = {'res4_6_1relu', 'res4_6_2relu'};
topLayerName = 'res4_6_3conv';
baseName = 'res4_6';
inputDim = 1024;
channelIndex = 2;
[netbasemodel, lossName]= insertPixelGate_withMildLoss(netbasemodel, ...
    rootLayerName, GateLayerNames, baseName, topLayerName, inputDim, lambda, channelIndex, withLossFlag);
sparsityLayerList{end+1} = lossName;
if withLossFlag
    lossCellList{end+1} = lossName;
    lossCellList{end+1} = sparsityLossWeight;
end
%}
%% insert pixel-gate on resBlock5
%{
rootLayerName = 'res4_6_relu';
GateLayerNames = {'res5_1_1relu', 'res5_1_2relu'};
topLayerName = 'res5_1_3conv';
baseName = 'res5_1';
inputDim = 1024;
channelIndex = 2;
[netbasemodel, lossName]= insertPixelGate_withMildLoss(netbasemodel, ...
    rootLayerName, GateLayerNames, baseName, topLayerName, inputDim, lambda, channelIndex, withLossFlag);
sparsityLayerList{end+1} = lossName;
if withLossFlag
    lossCellList{end+1} = lossName;
    lossCellList{end+1} = sparsityLossWeight;
end


% rootLayerName = 'res5_1_relu';
% GateLayerNames = {'res5_2_1relu', 'res5_2_2relu'};
% topLayerName = 'res5_2_3conv';
% baseName = 'res5_2';
% inputDim = 2048;
% channelIndex = 2;
% [netbasemodel, lossName]= insertPixelGate_withMildLoss(netbasemodel, ...
%     rootLayerName, GateLayerNames, baseName, topLayerName, inputDim, lambda, channelIndex, withLossFlag);
% sparsityLayerList{end+1} = lossName;
% if withLossFlag
%     lossCellList{end+1} = lossName;
%     lossCellList{end+1} = sparsityLossWeight;
% end


rootLayerName = 'res5_2_relu';
GateLayerNames = {'res5_3_1relu', 'res5_3_2relu'};
topLayerName = 'res5_3_3conv';
baseName = 'res5_3';
inputDim = 2048;
channelIndex = 2;
[netbasemodel, lossName]= insertPixelGate_withMildLoss(netbasemodel, ...
    rootLayerName, GateLayerNames, baseName, topLayerName, inputDim, lambda, channelIndex, withLossFlag);
sparsityLayerList{end+1} = lossName;
if withLossFlag
    lossCellList{end+1} = lossName;
    lossCellList{end+1} = sparsityLossWeight;
end
%}
%% config learning rate
baselr = 0.0000;
lossCellList{end+1} = 'res2_1_attentionProbSparse_Loss';
lossCellList{end+1} = sparsityLossWeight*baselr;
lossCellList{end+1} = 'res2_2_attentionProbSparse_Loss';
lossCellList{end+1} = sparsityLossWeight*baselr;
lossCellList{end+1} = 'res2_3_attentionProbSparse_Loss';
lossCellList{end+1} = sparsityLossWeight*baselr;
lossCellList{end+1} = 'res3_1_attentionProbSparse_Loss';
lossCellList{end+1} = sparsityLossWeight*baselr;
lossCellList{end+1} = 'res3_2_attentionProbSparse_Loss';
lossCellList{end+1} = sparsityLossWeight*baselr;
lossCellList{end+1} = 'res3_3_attentionProbSparse_Loss';
lossCellList{end+1} = sparsityLossWeight*baselr;
lossCellList{end+1} = 'res3_4_attentionProbSparse_Loss';
lossCellList{end+1} = sparsityLossWeight*baselr;
lossCellList{end+1} = 'res4_1_attentionProbSparse_Loss';
lossCellList{end+1} = sparsityLossWeight*baselr;
lossCellList{end+1} = 'res4_2_attentionProbSparse_Loss';
lossCellList{end+1} = sparsityLossWeight*baselr;
lossCellList{end+1} = 'res4_3_attentionProbSparse_Loss';
lossCellList{end+1} = sparsityLossWeight*baselr;
lossCellList{end+1} = 'res4_4_attentionProbSparse_Loss';
lossCellList{end+1} = sparsityLossWeight*baselr;
lossCellList{end+1} = 'res4_5_attentionProbSparse_Loss';
lossCellList{end+1} = sparsityLossWeight*baselr;
lossCellList{end+1} = 'res4_6_attentionProbSparse_Loss';
lossCellList{end+1} = sparsityLossWeight*baselr;
lossCellList{end+1} = 'res5_1_attentionProbSparse_Loss';
lossCellList{end+1} = sparsityLossWeight*baselr;
% lossCellList{end+1} = 'res5_2_attentionProbSparse_Loss';
% lossCellList{end+1} = sparsityLossWeight*baselr;
lossCellList{end+1} = 'res5_3_attentionProbSparse_Loss';
lossCellList{end+1} = sparsityLossWeight*baselr;
%{
for i = 1:length(netbasemodel.layers)
    if strcmpi( class(netbasemodel.layers(i).block), 'attentionChannel' )
        fprintf('''%s'', \n', netbasemodel.layers(i).name);        
        netbasemodel.params(netbasemodel.layers(i).paramIndexes(1)).learningRate = 1;
        netbasemodel.params(netbasemodel.layers(i).paramIndexes(2)).learningRate = 1;
    end
    
    if strcmpi( class(netbasemodel.layers(i).block), 'dagnn.Conv' )
        fprintf('''%s'', \n', netbasemodel.layers(i).name);        
        idxList = netbasemodel.layers(i).paramIndexes;
        for idx=idxList
            netbasemodel.params(idx).learningRate = 1;
        end
    end
end

for i = 1:length(netbasemodel.layers)
    if contains(netbasemodel.layers(i).name, 'ScaleAttention_') ...
            && contains(netbasemodel.layers(i).name, '_pyramid_')
        fprintf('''%s'', \n', netbasemodel.layers(i).name);        
        paramIdxList = netbasemodel.layers(i).paramIndexes;
        for ii = 1:length(paramIdxList)
            netbasemodel.params(netbasemodel.layers(i).paramIndexes(ii)).learningRate = 1;        
        end
    end
    
    if contains(netbasemodel.layers(i).name, 'ScaleAttention_') ...
            && contains(netbasemodel.layers(i).name, 'AttentionLayer') ...
            && contains(netbasemodel.layers(i).name, 'conv')
        fprintf('''%s'', \n', netbasemodel.layers(i).name);        
        paramIdxList = netbasemodel.layers(i).paramIndexes;        
        for ii = 1:length(paramIdxList)
            netbasemodel.params(netbasemodel.layers(i).paramIndexes(ii)).learningRate = 1;        
        end
    elseif contains(netbasemodel.layers(i).name, 'ScaleAttention_') ...
            && contains(netbasemodel.layers(i).name, 'AttentionLayer') ...
            && contains(netbasemodel.layers(i).name, 'bn')
        fprintf('''%s'', \n', netbasemodel.layers(i).name);        
        paramIdxList = netbasemodel.layers(i).paramIndexes;        
        netbasemodel.params(netbasemodel.layers(i).paramIndexes(1)).learningRate = 1;        
        netbasemodel.params(netbasemodel.layers(i).paramIndexes(2)).learningRate = 1;        
        netbasemodel.params(netbasemodel.layers(i).paramIndexes(3)).learningRate = 0.01;
    end
end
%}

% netbasemodel.params(netbasemodel.getParamIndex('normalEstLayerOne_conv_f')).learningRate = 5;
% netbasemodel.params(netbasemodel.getParamIndex('normalEstLayerTwo_conv_f')).learningRate = 5;
% 
% netbasemodel.params(netbasemodel.getParamIndex('normalOutput_f')).learningRate = 5;
% netbasemodel.params(netbasemodel.getParamIndex('normalOutput_b')).learningRate = 5;
% for i = 180:204
%     netbasemodel.params(i).learningRate = netbasemodel.params(i).learningRate/5;
% end
% for i = 205:230
%     netbasemodel.params(i).learningRate = 0;
% end
% for i = 231:234
%     netbasemodel.params(i).learningRate=5;
% end

for i = 1:numel(netbasemodel.params)
    fprintf('%d \t%s, \t\t%.2f\n',i, netbasemodel.params(i).name, netbasemodel.params(i).learningRate);
end
%% configure training environment
totalEpoch = 150;
learningRate = 1:totalEpoch;
learningRate = (1e-5) * (1-learningRate/totalEpoch).^0.9; % epoch 1~?

t_list = ones(2,1) * [1:-0.03:0.001];
t_list = t_list(:)';
t_list = [t_list, 0.001*ones(1,length(learningRate))];
t_list = t_list(1:length(learningRate));

%% setup for training
weightDecay=0.0005; % weightDecay: usually use the default value
batchSize = 1;
opts.batchSize = batchSize;
opts.learningRate = learningRate;
opts.weightDecay = weightDecay;
opts.momentum = 0.9 ;

opts.scalingFactor = scalingFactor;

opts.expDir = fullfile('./exp', 'main006normal_Res5ScaleAttention_pG2345_p07');
if ~isdir(opts.expDir)
    mkdir(opts.expDir);
end

opts.withBoundary = false ;
opts.withDepth = false ;
opts.withNormal = true ;
opts.withSemanticSeg = false;
opts.withInstanceSeg = false ;
opts.withWeights = false ;
opts.trainAugmentation = trainAugmentation;

opts.numSubBatches = 1 ;
opts.continue = true ;
opts.gpus = gpuId ;
%gpuDevice(opts.train.gpus); % don't want clear the memory
opts.prefetch = false ;
opts.sync = false ; % for speed
opts.cudnn = true ; % for speed
opts.numEpochs = numel(opts.learningRate) ;
opts.learningRate = learningRate;

opts.train = 1:numel(imdb.train_image);
opts.val = 1:numel(imdb.val_image);

opts.checkpointFn = [];
mopts.classifyType = 'softmax';

rng(777);
bopts.numThreads = 12;
bopts.border = netbasemodel.meta.normalization.border; 
bopts.stepSize = netbasemodel.meta.normalization.stepSize;
bopts.imageSize = netbasemodel.meta.normalization.imageSize ; %704x1280
bopts.imdb = imdb;
%% train
fn = getBatchWrapper_Stanford2D3D_multiTask_flipAugOnly(bopts);

opts.backPropDepth = inf; % could limit the backprop
prefixStr = [mopts.classifyType, '_'];
% opts.backPropAboveLayerName = 'res6_conv';
opts.backPropAboveLayerName = 'conv1_1';
% opts.backPropAboveLayerName = 'res3_1_projBranch';
% opts.backPropAboveLayerName = 'res4_1_projBranch';
% opts.backPropAboveLayerName = 'res5_1_projBranch';

global backPropLayerAbove backPropParamAbove
backPropLayerAbove = netbasemodel.getLayerIndex(opts.backPropAboveLayerName);
backPropParamAbove = netbasemodel.layers(backPropLayerAbove).paramIndexes;

trainfn = @cnnTrainStanford2D3D_multiTask_gradual;
[netbasemodel, info] = trainfn(netbasemodel, prefixStr, imdb, fn, ...
    'derOutputs', lossCellList, opts);
%% leaving blank
%{
flip-aug 
Saving the benchmarking results in ./matResult/val_main006normal_Res5ScaleAttention_pG2345_p07_softmax_net-epoch-40_visualization_softmax_net-epoch-40.
within 11.25-deg: 0.5650  within 22.50-deg: 0.7308  within 30.00-deg: 0.7969   Angle Dist Mean: 17.0422

flip-aug 
Saving the benchmarking results in ./matResult/val_main006normal_Res5ScaleAttention_pG2345_p07_softmax_net-epoch-67_visualization_softmax_net-epoch-67.
within 11.25-deg: 0.5746  within 22.50-deg: 0.7372  within 30.00-deg: 0.8011   Angle Dist Mean: 16.7248

Saving the benchmarking results in ./matResult/val_main006normal_Res5ScaleAttention_pG2345_p07_softmax_net-epoch-100_visualization_softmax_net-epoch-100.
within 11.25-deg: 0.5829  within 22.50-deg: 0.7424  within 30.00-deg: 0.8042   Angle Dist Mean: 16.4872

flip-aug 
Saving the benchmarking results in ./matResult/val_main006normal_Res5ScaleAttention_pG2345_p07_softmax_net-epoch-140_visualization_softmax_net-epoch-140.
within 11.25-deg: 0.5793  within 22.50-deg: 0.7408  within 30.00-deg: 0.8039   Angle Dist Mean: 16.5423

Saving the benchmarking results in ./matResult/val_main006normal_Res5ScaleAttention_pG2345_p07_softmax_net-epoch-150_visualization_softmax_net-epoch-150.
within 11.25-deg: 0.5806  within 22.50-deg: 0.7411  within 30.00-deg: 0.8040   Angle Dist Mean: 16.5043

%}
