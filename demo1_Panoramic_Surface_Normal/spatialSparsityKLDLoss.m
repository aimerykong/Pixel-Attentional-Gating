classdef spatialSparsityKLDLoss < dagnn.Loss
    properties
        lambda = 0.8
        channelIndex = 2
        meanValue
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            obj.meanValue = [];
            outputs{1} = 0;
            obj.meanValue = gpuArray(zeros(1,2, 'single'));
            obj.meanValue(1) = mean(reshape(inputs{1}(:,:,1), [], 1));
            obj.meanValue(2) = mean(reshape(inputs{1}(:,:,2), [], 1));
            
            outputs{1} = obj.lambda*log(obj.lambda/obj.meanValue(2)) + (1-obj.lambda)*log((1-obj.lambda)/(1-obj.meanValue(2))) + ...
                (1-obj.lambda)*log((1-obj.lambda)/obj.meanValue(1)) + obj.lambda*log(obj.lambda/(1-obj.meanValue(1)));
            
%             outputs{1} = log(1+(obj.meanValue(1)-(1-obj.lambda))^2) + ...
%                 log(1+(obj.meanValue(2)-obj.lambda)^2);
            
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
%             der1 = 2/numel(inputs{1}(:,:,1)) / (1+(obj.meanValue(1)-1+obj.lambda)^2) * (obj.meanValue(1)-1+obj.lambda);
%             der2 = 2/numel(inputs{1}(:,:,2)) / (1+(obj.meanValue(2)-obj.lambda)^2) * (obj.meanValue(2)-obj.lambda);
            
            der1 = 1/numel(inputs{1}(:,:,1)) * (-(1-obj.lambda)/obj.meanValue(1) + obj.lambda/(1-obj.meanValue(1)));
            der2 = 1/numel(inputs{1}(:,:,2)) * (-obj.lambda/obj.meanValue(2) + (1-obj.lambda)/(1-obj.meanValue(2)));
            
            %der = 2/(1+(obj.meanValue-obj.lambda)^2) * (obj.meanValue-obj.lambda);
            derInputs{1} = gpuArray(ones(size(inputs{1}),'single'));
            derInputs{1}(:,:,1) = der1*derInputs{1}(:,:,1);
            derInputs{1}(:,:,2) = der2*derInputs{1}(:,:,2);
            derParams = {} ;
        end
        
        function obj = spatialSparsityKLDLoss(varargin)
            obj.load(varargin) ;
        end
    end
end
