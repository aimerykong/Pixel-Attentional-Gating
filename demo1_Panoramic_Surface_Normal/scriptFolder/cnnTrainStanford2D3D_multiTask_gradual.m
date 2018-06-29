function [net,stats] = cnnTrainStanford2D3D_multiTask_gradual(net, prefixStr, imdb, getBatch, varargin)
%CNN_TRAIN_DAG Demonstrates training a CNN using the DagNN wrapper
%    CNN_TRAIN_DAG() is similar to CNN_TRAIN(), but works with
%    the DagNN wrapper instead of the SimpleNN wrapper.
%
% Copyright (C) 2014-16 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
%
% modified by Shu Kong @ UCI

%% parse opts
opts.scalingFactor = 1;
opts.imdb = [];
opts.expDir = fullfile('exp') ;
opts.continue = true ;
opts.batchSize = 1 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.test = [] ;
opts.gpus = [] ;
opts.prefetch = false ;
opts.numEpochs = 200 ;
opts.learningRate = 0.001 ; 
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.randomSeed = 0 ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.profile = false ;

opts.trainAugmentation = false;
opts.withBoundary = false ;
opts.withDepth = false ;
opts.withNormal = false ;
opts.withSemanticSeg = false ;
opts.withInstanceSeg = false ;
opts.withWeights = false ;


opts.sync = false ; % for speed
opts.cudnn = true ; % for speed
opts.backPropDepth = inf; % could limit the backprop
opts.backPropAboveLayerName = 'conv1_1';

opts.derOutputs = {'objective', 1} ;
opts.extractStatsFn = @extractStats ;
opts.checkpointFn = []; % will be called after every epoch
opts.plotStatistics = true;
opts = vl_argparse(opts, varargin) ;

% if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
% if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
% if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
% if isnan(opts.train), opts.train = [] ; end
%% Initialization
evaluateMode = isempty(opts.train) ;
if ~evaluateMode
    if isempty(opts.derOutputs)
        error('DEROUTPUTS must be specified when training.\n') ;
    end
end

state.getBatch = getBatch ;
stats = [] ;
%% Train and validate
modelPath = @(ep) fullfile(opts.expDir, sprintf('%snet-epoch-%d.mat', prefixStr, ep));
modelFigPath = fullfile(opts.expDir, [prefixStr 'net-train.pdf']) ;
start = opts.continue * myfindLastCheckpoint(opts.expDir, prefixStr) ;
if start >= 1
    fprintf('%s: resuming by loading epoch %d\n', mfilename, start) ;
    [net, stats] = loadState(modelPath(start)) ;
end

for i = 1:numel(net.params)
    fprintf('%d \t%s, \t\t%.2f\n',i, net.params(i).name, net.params(i).learningRate);
end 
%% train
plotLearningCurves(stats);
for epoch=start+1:opts.numEpochs    
    % Set the random seed based on the epoch and opts.randomSeed.
    % This is important for reproducibility, including when training
    % is restarted from a checkpoint.    
    rng(epoch + opts.randomSeed) ;
    prepareGPUs(opts, epoch == start+1) ;
    
    % Train for one epoch.    
    state.epoch = epoch ;
    state.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
    state.train = opts.train(randperm(numel(opts.train))) ; % shuffle
    state.val = opts.val(1:numel(opts.val)) ; % do not shuffle
    state.imdb = imdb ;
        
    if numel(opts.gpus) <= 1
        for ii = 1:length(net.layers)
            if strcmpi(class(net.layers(ii).block), 'GumbelNoisySoftMax')
                net.layers(ii).block.t = opts.t_list(min(epoch, numel(opts.t_list)));                
            end
        end
        
        [stats.train(epoch),prof] = stanford2D3D_processEpoch_multiTask(net, state, opts.scalingFactor, opts, 'train', opts.numEpochs) ;
        stats.val(epoch) = stanford2D3D_processEpoch_multiTask(net, state, opts.scalingFactor, opts, 'val', opts.numEpochs) ;

        if opts.profile
            profview(0,prof) ;
            keyboard ;
        end
    else
        savedNet = net.saveobj() ;
        spmd
            net_ = dagnn.DagNN.loadobj(savedNet) ;
            [stats_.train, prof_] = stanford2D3D_processEpoch_multiTask(net_, state, opts.scalingFactor, opts, 'train') ;
            stats_.val = stanford2D3D_processEpoch_multiTask(net_, state, opts.scalingFactor, opts, 'val') ;
            if labindex == 1, savedNet_ = net_.saveobj() ; end
        end
        net = dagnn.DagNN.loadobj(savedNet_{1}) ;
        stats__ = accumulateStats(stats_) ;
        stats.train(epoch) = stats__.train ;
        stats.val(epoch) = stats__.val ;
        if opts.profile
            mpiprofile('viewer', [prof_{:,1}]) ;
            keyboard ;
        end
        clear net_ stats_ stats__ savedNet savedNet_ ;
    end
    
    % save
    if ~evaluateMode
        saveState(modelPath(epoch), net, stats) ;
    end
    
    if opts.plotStatistics
%         switchFigure(1) ; clf ;
%         imgFig = figure(1);
%         set(imgFig, 'Position', [100 100 1500 800]) % [1 1 width height]

        plots = setdiff(...
            cat(2,...
            fieldnames(stats.train)', ...
            fieldnames(stats.val)'), {'num', 'time'}) ;
        for p = plots
            p = char(p) ;
            values = zeros(0, length(stats.train)) ;
            leg = {} ;
            for f = {'train', 'val'}
                f = char(f) ;
                if isfield(stats.(f), p)
                    tmp = [stats.(f).(p)] ;
                    values(end+1,:) = tmp(1,:)' ;
                    leg{end+1} = f ;
                end
            end
            if numel(plots)<=3
                subplot(1,numel(plots),find(strcmp(p,plots))) ;
            elseif numel(plots) <= 4
                subplot(2, 2, find(strcmp(p,plots))) ;
            elseif numel(plots) <= 6
                subplot(2, 3, find(strcmp(p,plots))) ;
            elseif numel(plots) <= 9                
                subplot(3, 3, find(strcmp(p,plots))) ;
            elseif numel(plots) <= 12
                subplot(3, 4, find(strcmp(p,plots))) ;
            elseif numel(plots) <= 16
                subplot(4, 4, find(strcmp(p,plots))) ;
            elseif numel(plots) <= 20
                subplot(4, 5, find(strcmp(p,plots))) ;
            elseif numel(plots) <= 25
                subplot(5, 5, find(strcmp(p,plots))) ;
            else
                subplot(5, 6, find(strcmp(p,plots))) ;
            end
            plot(1:size(values,2), values(:, 1:end)','o-') ; % don't plot the first epoch            

            xlabel('epoch') ;
            [minVal,minIdx] = min(values(2,:));
            [minValTr,minIdxTr] = min(values(1,:));
            title(sprintf('%s tsErr%.4f (%d) trErr%.4f (%d) ', p(1:min(10,length(p))), min(values(2,:)), minIdx, min(values(1,:)),minIdxTr), 'Interpreter', 'none') ;
            legend(leg{:},'location', 'SouthOutside') ;
            grid on ;            
        end
        drawnow ;
        [curpath, curname, curext] = fileparts(modelFigPath);        
        export_fig(fullfile(curpath, [curname, '.png']), '-png');
%         print(1, modelFigPath, '-dpdf') ;
    end
    if ~isempty(opts.checkpointFn),
        opts.checkpointFn();
    end
    
end


