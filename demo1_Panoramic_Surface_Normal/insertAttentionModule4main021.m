function [netbasemodel, lName] = insertAttentionModule4main021(netbasemodel, inDim, poolScaleList, sName, namePrefix)
% Attentional Scale Pyramid Pooling Module
%
%
% Shu Kong @ UCI
% 20171130

outDim = inDim;
root_sName = sName;
%% add the pyramid pooling block
for poolIdx = poolScaleList  % poolScaleList = [1 2 4 6 8 10] % [1 2 4 8 16]
    sName = root_sName;
    baseName = sprintf('%s_pyramid_pool%d', namePrefix, poolIdx);
    lName = [baseName, '_conv'];
    paramName = [lName '_f'];
    block = dagnn.Conv('size', [3 3 inDim outDim], 'hasBias', false, 'stride', 1, 'pad', poolIdx, 'dilate', poolIdx);
    netbasemodel.addLayer(lName, block, sName, lName, {paramName});
    ind = netbasemodel.getParamIndex(paramName);
    netbasemodel.params(ind).value = randn([3 3 inDim outDim], 'single')*sqrt(2/outDim);
    netbasemodel.params(ind).learningRate = 1;
    sName = lName;
    
    lName = [baseName, '_ChaAtten'];
    block = attentionChannel('numChannels', outDim);
    block.globalMean = zeros(outDim, 1, 'single');
    block.globalVariance = ones(outDim, 1, 'single');
    netbasemodel.addLayer(lName, block, sName, lName, {[lName '_multiplier'], [lName '_bias']});
    pidx = netbasemodel.getParamIndex({[lName '_multiplier'], [lName '_bias']});
    netbasemodel.params(pidx(1)).weightDecay = 1;
    netbasemodel.params(pidx(2)).weightDecay = 1;
    netbasemodel.params(pidx(1)).learningRate = 1;
    netbasemodel.params(pidx(2)).learningRate = 1;
    netbasemodel.params(pidx(1)).value = ones(outDim, 1, 'single'); %ones(512, 1, 'single'); % slope
    netbasemodel.params(pidx(2)).value = zeros(outDim, 1, 'single'); %zeros(512, 1, 'single');  % bias
    
    sName = lName;
    lName = [baseName, '_relu'];
    block = dagnn.ReLU('leak', 0);
    netbasemodel.addLayer(lName, block, sName, lName);
end

%% add the attention block
baseName = sprintf('%s_AttentionLayerOne', namePrefix);
lName = [baseName, '_conv'];
block = dagnn.Conv('size', [1 1 inDim 128], 'hasBias', false, 'stride', 1, 'pad', 0, 'dilate', 1);
netbasemodel.addLayer(lName, block, root_sName, lName, {[lName '_f']});
ind = netbasemodel.getParamIndex([lName '_f']);
netbasemodel.params(ind).value = randn([1 1 inDim 128], 'single')*sqrt(2/128);
netbasemodel.params(ind).learningRate = 10;
sName = lName;


lName = [baseName, '_bn'];
block = dagnn.BatchNorm('numChannels', 128);
netbasemodel.addLayer(lName, block, sName, lName, {[lName '_g'], [lName '_b'], [lName '_m']});
pidx = netbasemodel.getParamIndex({[lName '_g'], [lName '_b'], [lName '_m']});
netbasemodel.params(pidx(1)).weightDecay = 1;
netbasemodel.params(pidx(1)).learningRate = 10;
netbasemodel.params(pidx(2)).weightDecay = 1;
netbasemodel.params(pidx(2)).learningRate = 10;
netbasemodel.params(pidx(3)).learningRate = 0.1;
netbasemodel.params(pidx(3)).trainMethod = 'average';

netbasemodel.params(pidx(1)).value = ones([128,1], 'single'); % slope
netbasemodel.params(pidx(2)).value = zeros([128,1], 'single');  % bias
netbasemodel.params(pidx(3)).value = [zeros([128,1], 'single'), ones([128,1], 'single')]; % moments
netbasemodel.layers(netbasemodel.getLayerIndex(lName)).block.usingGlobal = false;
sName = lName;

lName = [baseName, '_relu'];
block = dagnn.ReLU('leak', 0);
netbasemodel.addLayer(lName, block, sName, lName);
sName = lName;


baseName = sprintf('%s_AttentionLayerTwo', namePrefix);
lName = [baseName, '_conv'];
block = dagnn.Conv('size', [3 3 128 32], 'hasBias', false, 'stride', 1, 'pad', 1, 'dilate', 1);
netbasemodel.addLayer(lName, block, sName, lName, {[lName '_f']});
ind = netbasemodel.getParamIndex([lName '_f']);
netbasemodel.params(ind).value = randn([3 3 128 32], 'single')*sqrt(2/32);
netbasemodel.params(ind).learningRate = 10;
sName = lName;

lName = [baseName, '_bn'];
block = dagnn.BatchNorm('numChannels', 32);
netbasemodel.addLayer(lName, block, sName, lName, {[lName '_g'], [lName '_b'], [lName '_m']});
pidx = netbasemodel.getParamIndex({[lName '_g'], [lName '_b'], [lName '_m']});
netbasemodel.params(pidx(1)).weightDecay = 1;
netbasemodel.params(pidx(1)).learningRate = 10;
netbasemodel.params(pidx(2)).weightDecay = 1;
netbasemodel.params(pidx(2)).learningRate = 10;
netbasemodel.params(pidx(3)).learningRate = 0.1;
netbasemodel.params(pidx(3)).trainMethod = 'average';

netbasemodel.params(pidx(1)).value = ones([32,1], 'single'); % slope
netbasemodel.params(pidx(2)).value = zeros([32,1], 'single');  % bias
netbasemodel.params(pidx(3)).value = [zeros([32,1], 'single'), ones([32,1], 'single')]; % moments
netbasemodel.layers(netbasemodel.getLayerIndex(lName)).block.usingGlobal = false;
sName = lName;

lName = [baseName, '_relu'];
block = dagnn.ReLU('leak', 0);
netbasemodel.addLayer(lName, block, sName, lName);
sName = lName;


lName = sprintf('%s_AttentionLayer_conv', namePrefix);
block = dagnn.Conv('size', [1 1 32 length(poolScaleList)+1], 'hasBias', true, 'stride', 1, 'pad', 0, 'dilate', 1);
netbasemodel.addLayer(lName, block, sName, lName, {[lName '_f'], [lName '_b']});
ind = netbasemodel.getParamIndex([lName '_f']);
netbasemodel.params(ind).value = randn([1 1 32 length(poolScaleList)+1], 'single')*sqrt(2/length(poolScaleList));
netbasemodel.params(ind).learningRate = 10;
ind = netbasemodel.getParamIndex([lName '_b']);
netbasemodel.params(ind).value = zeros([length(poolScaleList)+1, 1], 'single');
netbasemodel.params(ind).learningRate = 10;
sName = lName;


lName = sprintf('%s_AttentionGumbelNoisySoftMax', namePrefix);
netbasemodel.addLayer(lName, GumbelNoisySoftMax(), sName, lName);

%% add multiplicative gating layer
lName = sprintf('%s_AttentionGatingLayer', namePrefix);
block = dagnn.MaskGating();
sNameList = {root_sName};

for poolIdx = poolScaleList
    sNameList{end+1} = sprintf('%s_pyramid_pool%d_relu', namePrefix, poolIdx);
end
sNameList{end+1} = sprintf('%s_AttentionGumbelNoisySoftMax', namePrefix);
netbasemodel.addLayer(lName, block,  ...
    sNameList, lName);

%sName = lName;


end
