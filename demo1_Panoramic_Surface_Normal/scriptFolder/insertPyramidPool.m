function netbasemodel = insertPyramidPool(netbasemodel, resX, inDim, outDim)

metaInputLayerName = [resX '_2_1relu'];
metaMergeLayerName = [resX '_2_2relu'];
metaNextLayerName = [resX '_2_3conv'];

resX_2_2conv_f = netbasemodel.params(netbasemodel.getParamIndex([resX '_2_2conv_f'])).value;
resX_2_2ChaAtten_multiplier = netbasemodel.params(netbasemodel.getParamIndex([resX '_2_2ChaAtten_multiplier'])).value;
resX_2_2ChaAtten_bias = netbasemodel.params(netbasemodel.getParamIndex([resX '_2_2ChaAtten_bias'])).value;
globalMean = netbasemodel.layers(netbasemodel.getLayerIndex([resX '_2_2ChaAtten'])).block.globalMean;
globalVariance = netbasemodel.layers(netbasemodel.getLayerIndex([resX '_2_2ChaAtten'])).block.globalVariance;


for poolIdx = [2 4 6 8 10] % [1 2 4 8 16]
    sName = metaInputLayerName;
    baseName = sprintf('%s_pyramid_pool%d', resX, poolIdx);
    lName = [baseName, '_conv'];
    paramName = [lName '_f'];
    block = dagnn.Conv('size', [3 3 inDim outDim], 'hasBias', false, 'stride', 1, 'pad', poolIdx, 'dilate', poolIdx);
    netbasemodel.addLayer(lName, block, sName, lName, {paramName});
    ind = netbasemodel.getParamIndex(paramName);
    netbasemodel.params(ind).value = resX_2_2conv_f;
    netbasemodel.params(ind).learningRate = 5;
    sName = lName;
    
    
    lName = [baseName, '_ChaAtten'];
    block = attentionChannel('numChannels', outDim);
    block.globalMean = globalMean; %zeros(outDim, 1, 'single');
    block.globalVariance = globalVariance; %ones(outDim, 1, 'single');
    netbasemodel.addLayer(lName, block, sName, lName, {[lName '_multiplier'], [lName '_bias']});
    pidx = netbasemodel.getParamIndex({[lName '_multiplier'], [lName '_bias']});
    netbasemodel.params(pidx(1)).weightDecay = 1;
    netbasemodel.params(pidx(2)).weightDecay = 1;
    netbasemodel.params(pidx(1)).learningRate = 5;
    netbasemodel.params(pidx(2)).learningRate = 5;
    netbasemodel.params(pidx(1)).value = resX_2_2ChaAtten_multiplier; %ones(outDim, 1, 'single'); %ones(512, 1, 'single'); % slope
    netbasemodel.params(pidx(2)).value = resX_2_2ChaAtten_bias; %zeros(outDim, 1, 'single'); %zeros(512, 1, 'single');  % bias
    sName = lName;
    
    
    lName = [baseName, '_relu'];
    block = dagnn.ReLU('leak', 0);
    netbasemodel.addLayer(lName, block, sName, lName);
end



sName = metaInputLayerName;
baseName = [resX '_AttentionLayerOne'];
outDim = 128;
lName = [baseName, '_conv'];
block = dagnn.Conv('size', [1 1 inDim outDim], 'hasBias', false, 'stride', 1, 'pad', 0, 'dilate', 1);
netbasemodel.addLayer(lName, block, sName, lName, {[lName '_f']});
ind = netbasemodel.getParamIndex([lName '_f']);
netbasemodel.params(ind).value = randn([1 1 inDim 128], 'single')*sqrt(2/outDim);
netbasemodel.params(ind).learningRate = 5;
sName = lName;
inDim = outDim;

lName = [baseName, '_bn'];
block = dagnn.BatchNorm('numChannels', outDim);
netbasemodel.addLayer(lName, block, sName, lName, {[lName '_g'], [lName '_b'], [lName '_m']});
pidx = netbasemodel.getParamIndex({[lName '_g'], [lName '_b'], [lName '_m']});
netbasemodel.params(pidx(1)).weightDecay = 1;
netbasemodel.params(pidx(1)).learningRate = 5;
netbasemodel.params(pidx(2)).weightDecay = 1;
netbasemodel.params(pidx(2)).learningRate = 5;
netbasemodel.params(pidx(3)).learningRate = 0.05;
netbasemodel.params(pidx(3)).trainMethod = 'average';

netbasemodel.params(pidx(1)).value = ones([outDim,1], 'single'); % slope
netbasemodel.params(pidx(2)).value = zeros([outDim,1], 'single');  % bias
netbasemodel.params(pidx(3)).value = zeros([outDim,2], 'single'); % moments
netbasemodel.layers(netbasemodel.getLayerIndex(lName)).block.usingGlobal = false;
sName = lName;

lName = [baseName, '_relu'];
block = dagnn.ReLU('leak', 0);
netbasemodel.addLayer(lName, block, sName, lName);
sName = lName;



baseName = [resX '_AttentionLayerTwo'];
outDim = 32;
lName = [baseName, '_conv'];
block = dagnn.Conv('size', [3 3 inDim outDim], 'hasBias', false, 'stride', 1, 'pad', 1, 'dilate', 1);
netbasemodel.addLayer(lName, block, sName, lName, {[lName '_f']});
ind = netbasemodel.getParamIndex([lName '_f']);
netbasemodel.params(ind).value = randn([3 3 inDim outDim], 'single')*sqrt(2/outDim);
netbasemodel.params(ind).learningRate = 5;
sName = lName;
inDim = outDim;


lName = [baseName, '_bn'];
block = dagnn.BatchNorm('numChannels', outDim);
netbasemodel.addLayer(lName, block, sName, lName, {[lName '_g'], [lName '_b'], [lName '_m']});
pidx = netbasemodel.getParamIndex({[lName '_g'], [lName '_b'], [lName '_m']});
netbasemodel.params(pidx(1)).weightDecay = 1;
netbasemodel.params(pidx(1)).learningRate = 5;
netbasemodel.params(pidx(2)).weightDecay = 1;
netbasemodel.params(pidx(2)).learningRate = 5;
netbasemodel.params(pidx(3)).learningRate = 0.05;
netbasemodel.params(pidx(3)).trainMethod = 'average';

netbasemodel.params(pidx(1)).value = ones([outDim,1], 'single'); % slope
netbasemodel.params(pidx(2)).value = zeros([outDim,1], 'single');  % bias
netbasemodel.params(pidx(3)).value = zeros([outDim,2], 'single'); % moments
netbasemodel.layers(netbasemodel.getLayerIndex(lName)).block.usingGlobal = false;
sName = lName;

lName = [baseName, '_relu'];
block = dagnn.ReLU('leak', 0);
netbasemodel.addLayer(lName, block, sName, lName);
sName = lName;


lName = [resX '_AttentionLayer_conv'];
outDim = 7;
block = dagnn.Conv('size', [1 1 inDim outDim], 'hasBias', true, 'stride', 1, 'pad', 0, 'dilate', 1);
netbasemodel.addLayer(lName, block, sName, lName, {[lName '_f'], [lName '_b']});
ind = netbasemodel.getParamIndex([lName '_f']);
netbasemodel.params(ind).value = randn([1 1 inDim outDim], 'single')*sqrt(2/outDim);
netbasemodel.params(ind).learningRate = 5;
ind = netbasemodel.getParamIndex([lName '_b']);
netbasemodel.params(ind).value = zeros([outDim, 1], 'single');
netbasemodel.params(ind).learningRate = 5;
sName = lName;


lName = [resX '_AttentionSoftmax'];
netbasemodel.addLayer(lName, dagnn.SoftMax(), sName, lName);


lName = [resX '_AttentionGatingLayer'];
block = dagnn.MaskGating();
netbasemodel.addLayer(lName, block,  ...
    {[resX '_pyramid_pool10_relu'], ...
    [resX '_pyramid_pool8_relu'], ...
    [resX '_pyramid_pool6_relu'], ...
    [resX '_pyramid_pool4_relu'], ...
    [resX '_pyramid_pool2_relu'], ...
    metaMergeLayerName, metaInputLayerName,  [resX '_AttentionSoftmax']}, lName);
sName = lName;


netbasemodel.setLayerInputs(metaNextLayerName, {sName});