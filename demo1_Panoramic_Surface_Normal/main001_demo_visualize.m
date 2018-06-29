%% add path and setup configuration
clc; clear; close all;

addpath(genpath('./scriptFolder'));
addpath(genpath('../libs'));
path_to_matconvnet = '../matconvnet-1.0-beta23_modifiedDagnn';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));

mean_r = 123.68; %means to be subtracted and the given values are used in our training stage
mean_g = 116.779;
mean_b = 103.939;
mean_bgr = reshape([mean_b, mean_g, mean_r], [1,1,3]);
mean_rgb = reshape([mean_r, mean_g, mean_b], [1,1,3]);
%% read matconvnet model
load('imdb.mat');
imdb.val_normal = imdb.val_localNormal;
imdb.train_normal = imdb.train_localNormal;
flagSaveFig = true; % {true false} whether to store the result

% set GPU 
gpuId = 1; %[1, 2];
gpuDevice(gpuId);
%% setup network
saveFolder = 'main006normal_Res5ScaleAttention_pG2345_p07';
modelName = 'softmax_net-epoch-140.mat';

netbasemodel = load( fullfile('./model/', saveFolder, modelName) );
netbasemodel = netbasemodel.netbasemodel;
netbasemodel = dagnn.DagNN.loadobj(netbasemodel);
layerTop = 'l2norm_normal';
netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).precious = 1;

[~, modelName] = fileparts(modelName);
saveFolder = fullfile('figFolder', [strrep(saveFolder, '/', ''), '_' modelName, '_visualization']);

netbasemodel.move('gpu');
netbasemodel.mode = 'test' ;
% netMat.mode = 'normal' ;
netbasemodel.conserveMemory = 1;
%% test res2
imlist = dir('./valimages/val*rgb.png');
setName = 'val'; % train val
for testImgIdx = 1:length(imlist)     
    %% read the image and annotation 
    cur_path_to_image = sprintf('./valimages/val%03d_rgb.png',testImgIdx);
    cur_path_to_annot = sprintf('./valimages/val%03d_semantic.png',testImgIdx);
    cur_path_to_depth = sprintf('./valimages/val%03d_depth.png',testImgIdx);
    cur_path_to_normal = sprintf('./valimages/val%03d_normals.png',testImgIdx);
    
    cur_image = imread(cur_path_to_image);
    cur_annot = imread(cur_path_to_annot);
    cur_depth = imread(cur_path_to_depth);
    cur_normal = imread(cur_path_to_normal);
    
    cur_image = single(cur_image);
    cur_annot = single(cur_annot);
    cur_depthLog10 = log10(single(cur_depth));
    cur_normal = single(cur_normal);
    cur_normal = (single(cur_normal)-127.5)/127.5; 
    a = sqrt(sum(cur_normal.^2,3));
    normalMask = (a<1.1 & a>0.8);
    cur_normal_fliplr = fliplr(cur_normal); cur_normal_fliplr(:,:,1) = -1*cur_normal_fliplr(:,:,1);
    
    
    fprintf('image-%03d %s ... \n', testImgIdx, cur_path_to_image);
    imFeed = bsxfun(@minus, cur_image, mean_rgb);        
    
    netbasemodel.eval({'data', gpuArray(imFeed)}) ;
    pred_normal = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).value);
        
    netbasemodel.eval({'data', gpuArray(fliplr(imFeed))}) ;
    pred_normal_fliplr = gather(netbasemodel.vars(netbasemodel.layers(netbasemodel.getLayerIndex(layerTop)).outputIndexes).value);
        
    
    fprintf(' done!\n');
    %% visualization    
    cur_annot_color = imdb.meta.mapping_id2color(cur_annot(:)+1, :);
    cur_annot_color = reshape(cur_annot_color, [size(cur_annot,1), size(cur_annot,2), 3]);
    
    %pred_annot_color = imdb.meta.mapping_id2color(predSeg(:)+1, :);
    %pred_annot_color = reshape(pred_annot_color, [size(predSeg,1), size(predSeg,2), 3]);

    imgFig = figure(1);
    subWindowH = 4; 
    subWindowW = 2;
    windowID = 1;    
    set(imgFig, 'Position', [100 100 900 900]) % [1 1 width height]
    
    subplot(subWindowH, subWindowW, windowID); windowID = windowID + 1;
    imagesc(uint8(cur_image)); title(sprintf('image-%04d', testImgIdx)); axis off image;
    
    subplot(subWindowH, subWindowW, windowID); windowID = windowID + 1;
    imagesc(cur_annot_color); title(sprintf('gtSeg-color'));  axis off image; 
    
    %subplot(subWindowH, subWindowW, windowID); windowID = windowID + 1;
    %imagesc(cur_depth); title(sprintf('cur_depth')); axis off image;
    
    subplot(subWindowH, subWindowW, windowID); windowID = windowID + 1;
    imagesc(cur_depthLog10); title(sprintf('gtDepth(log10)')); axis off image;
   
    subplot(subWindowH, subWindowW, windowID); windowID = windowID + 1;
    imagesc(normalMask); title(sprintf('normalMask')); axis off image;
    
    subplot(subWindowH, subWindowW, windowID); windowID = windowID + 1;
    imagesc(cur_normal/2+0.5); title(sprintf('gtNormal')); axis off image;
    
    subplot(subWindowH, subWindowW, windowID); windowID = windowID + 1;
    imagesc(pred_normal/2+0.5); title(sprintf('estNormal')); axis off image;
    
    subplot(subWindowH, subWindowW, windowID); windowID = windowID + 1;
    imagesc(cur_normal_fliplr/2+0.5); title(sprintf('gtNormalFlipLR')); axis off image;
    
    subplot(subWindowH, subWindowW, windowID); windowID = windowID + 1;
    imagesc(pred_normal_fliplr/2+0.5); title(sprintf('estNormalFlipLR')); axis off image;
    
    
    if flagSaveFig && ~isdir(saveFolder)
        mkdir(saveFolder);
    end
    if flagSaveFig
        export_fig( sprintf('%s/valImgId%04d.jpg', saveFolder, testImgIdx) );
    end
end
%% leaving blank

