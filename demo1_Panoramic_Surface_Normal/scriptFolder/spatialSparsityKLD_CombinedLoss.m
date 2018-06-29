classdef spatialSparsityKLD_CombinedLoss < dagnn.Loss
    properties
        lambda = 0.8
        channelIndex = 2
        meanValPerLayer
        numPixelPerLayer
        meanValue
    end
    
    methods
        function outputs = forward(obj, inputs, params)            
            outputs{1} = 0;
            obj.meanValPerLayer = gpuArray(zeros(length(inputs),2, 'single'));
            obj.numPixelPerLayer = gpuArray(zeros(length(inputs),1, 'single'));
            obj.meanValue = gpuArray(zeros(2,1, 'single')); 
            for i = 1:length(inputs)
                obj.meanValPerLayer(i,1) = mean(reshape(inputs{i}(:,:,1), [], 1));
                obj.meanValPerLayer(i,2) = mean(reshape(inputs{i}(:,:,2), [], 1));                
                obj.numPixelPerLayer(i) = size(inputs{i},1)*size(inputs{i}, 2);
                obj.meanValue(1) = obj.meanValue(1) + obj.meanValPerLayer(i,1)*obj.numPixelPerLayer(i);
                obj.meanValue(2) = obj.meanValue(2) + obj.meanValPerLayer(i,2)*obj.numPixelPerLayer(i);
            end
            obj.meanValue(1) = obj.meanValue(1) / sum(obj.numPixelPerLayer);
            obj.meanValue(2) = obj.meanValue(2) / sum(obj.numPixelPerLayer);
            
            outputs{1} = obj.lambda*log(obj.lambda/obj.meanValue(2)) + (1-obj.lambda)*log((1-obj.lambda)/(1-obj.meanValue(2))) + ...
                (1-obj.lambda)*log((1-obj.lambda)/obj.meanValue(1)) + obj.lambda*log(obj.lambda/(1-obj.meanValue(1)));
            
            
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            der1 = 1/numel(inputs{1}(:,:,1)) * (-(1-obj.lambda)/obj.meanValue(1) + obj.lambda/(1-obj.meanValue(1)));
            der2 = 1/numel(inputs{1}(:,:,2)) * (-obj.lambda/obj.meanValue(2) + (1-obj.lambda)/(1-obj.meanValue(2)));
            for i = 1:length(inputs)
                derInputs{i} = gpuArray(ones(size(inputs{i}),'single'));
                derInputs{i}(:,:,1) = der1*derInputs{i}(:,:,1);
                derInputs{i}(:,:,2) = der2*derInputs{i}(:,:,2);
            end 
%             derInputs{1} = gpuArray(ones(size(inputs{1}),'single'));
%             derInputs{1}(:,:,1) = der1*derInputs{1}(:,:,1);
%             derInputs{1}(:,:,2) = der2*derInputs{1}(:,:,2);
            derParams = {} ;
        end
        
        function obj = spatialSparsityKLD_CombinedLoss(varargin)
            obj.load(varargin) ;
        end
    end
end
