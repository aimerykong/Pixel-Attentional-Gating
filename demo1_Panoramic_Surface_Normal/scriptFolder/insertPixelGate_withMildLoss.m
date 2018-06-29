function [netbasemodel, lossName] = insertPixelGate_withMildLoss(netbasemodel, ...
    rootLayerName, GateLayerNames, baseName, topLayerName, ...
    inputDim, lambda, channelIndex, withLossFlag, stride, GaussianBlurFlag)
%
%
%
% Shu Kong @ UCI
% 20171219
%%
if nargin < 10
    stride = 1;
end
if nargin < 11
    GaussianBlurFlag = false;
end
%%
sName = rootLayerName;
lName = [baseName, '_attentionConv'];
block = dagnn.Conv('size', [1 1 inputDim 2], 'hasBias', true, 'stride', stride, 'pad', 0, 'dilate', 1);
netbasemodel.addLayer(lName, block, sName, lName, {[lName '_f'], [lName '_b']});
ind = netbasemodel.getParamIndex([lName '_f']);
netbasemodel.params(ind).value = randn([1 1 inputDim 2], 'single')*sqrt(2/2);
netbasemodel.params(ind).learningRate = 1;
ind = netbasemodel.getParamIndex([lName '_b']);
netbasemodel.params(ind).value = zeros(2, 1, 'single');
netbasemodel.params(ind).learningRate = 1;
sName = lName;

if GaussianBlurFlag
    lName = [sName, '_GaussBlur'];
    
    filtersAvg = ones(3, 3, 'single')/9.0;
    filters0 = zeros(3, 3, 'single');
    filters = cat(4, cat(3,filtersAvg,filters0), cat(3,filters0,filtersAvg));
        
    block = dagnn.Conv('size', [3 3 2 2], 'hasBias', false, 'stride', 1, 'pad', 1, 'dilate', 1);
    netbasemodel.addLayer(lName, block, sName, lName, {[lName '_f']});
    ind = netbasemodel.getParamIndex([lName '_f']);
    netbasemodel.params(ind).value = filters;
    netbasemodel.params(ind).learningRate = 0;
    sName = lName;        
end


lName = [baseName '_attentionProb'];
% netbasemodel.addLayer(lName, GumbelNoisySoftMax(), sName, lName);
netbasemodel.addLayer(lName, dagnn.SoftMax(), sName, lName);
sName = lName;

if withLossFlag
    lName = [baseName '_attentionProbSparse_Loss'];
%     netbasemodel.addLayer(lName, ...
%         spatialSparsityKLDLoss('loss', 'spatialsparsity', 'lambda', lambda, 'channelIndex', channelIndex), ...
%         {sName}, lName)
    netbasemodel.addLayer(lName, ...
        spatialSparsityLoss('loss', 'spatialsparsity', 'lambda', lambda, 'channelIndex', channelIndex), ...
        {sName}, lName)
    lossName = lName;
else
    lossName = sName;    
end

sNameList = {GateLayerNames{1}, GateLayerNames{2}, sName};
lName = [baseName, '_attentionGating'];
block = dagnn.MaskGating();
netbasemodel.addLayer(lName, block, sNameList, lName);
sName = lName;

netbasemodel.setLayerInputs(topLayerName, {sName});