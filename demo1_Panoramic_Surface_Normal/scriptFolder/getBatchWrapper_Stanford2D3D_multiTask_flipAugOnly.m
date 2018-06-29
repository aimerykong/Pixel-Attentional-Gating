% return a get batch function
% -------------------------------------------------------------------------
function fn = getBatchWrapper_Stanford2D3D_multiTask_flipAugOnly(opts)
% -------------------------------------------------------------------------
    fn = @(images, mode) getBatch_dict(images, mode, opts) ;
end

% -------------------------------------------------------------------------
function [imBatch, segBatch, depthBatch, normalBatch, maskBatch] = getBatch_dict(images, mode, opts)
% -------------------------------------------------------------------------
    %images = strcat([imdb.path_to_dataset filesep], imdb.(mode).(batch) ) ; 
    [imBatch, segBatch, depthBatch, normalBatch, maskBatch] = getImgBatch_Stanford2D3D_multiTask_flipAugOnly(images, mode, opts) ;
end
