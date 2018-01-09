classdef Conv < dagnn.Filter
  properties
    size = [0 0 0 0]
    hasBias = true
    opts = {'cuDNN'}
  end

  methods
    function outputs = forward(obj, inputs, params)
      if ~obj.hasBias, params{2} = [] ; end
      params_1 = params{1};
      %%binary
      [a,b,c,d]=size(params{1});
        for g=1:a
            for p=1:b
		 params_1(g,p,:,:)=sign(params{1}(g,p,:,:));
            end       
        end
%%ternary
%        threshold = 0.0044;
%        [a,b,c,d]=size(params{1});
%         for g=1:a
%             for p=1:b
% 		 params_1(g,p,:,:)=sign(sign(params{1}(g,p,:,:)-threshold)+sign(params{1}(g,p,:,:)-threshold));
%             end       
%         end
      outputs{1} = vl_nnconv(...
        inputs{1}, params_1, params{2}, ...
        'pad', obj.pad, ...
        'stride', obj.stride, ...
        'dilate', obj.dilate, ...
        obj.opts{:}) 
%     [a1,b1,c1,d1]=size(outputs{1});
%         for g1=1:a1
%             for p1=1:b1
%                 for l1 = 1:c1 
%                     for h1 = 1:d1
%                     outputs{1}(g1,p1,c1,d1)=fi(outputs{1}(g1,p1,c1,d1),1,8,5);
%                     end
%                 end
%             end       
%         end   
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      if ~obj.hasBias, params{2} = [] ; end
      params_1 = params{1};
      %%binary
      [a,b,c,d]=size(params{1});
        for g=1:a
            for p=1:b
		 params_1(g,p,:,:)=sign(params{1}(g,p,:,:));
            end       
        end
       %%dorefa_net
%        threshold = 0.01;
%        [a,b,c,d]=size(params{1});
%         for g=1:a
%             for p=1:b
% 		 params_1(g,p,:,:)=sign(sign(params{1}(g,p,:,:)-threshold)+sign(params{1}(g,p,:,:)-threshold));
%             end       
%         end


      [derInputs{1}, derParams{1}, derParams{2}] = vl_nnconv(...
        inputs{1}, params_1, params{2}, derOutputs{1}, ...
        'pad', obj.pad, ...
        'stride', obj.stride, ...
        'dilate', obj.dilate, ...
        obj.opts{:}) ;
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.size(1:2) ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = obj.size(4) ;
    end

    function params = initParams(obj)
      % Xavier improved
      sc = sqrt(2 / prod(obj.size(1:3))) ;
      %sc = sqrt(2 / prod(obj.size([1 2 4]))) ;
      params{1} = randn(obj.size,'single') * sc ;
      if obj.hasBias
        params{2} = zeros(obj.size(4),1,'single') ;
      end
    end

    function set.size(obj, ksize)
      % make sure that ksize has 4 dimensions
      ksize = [ksize(:)' 1 1 1 1] ;
      obj.size = ksize(1:4) ;
    end

    function obj = Conv(varargin)
      obj.load(varargin) ;
      % normalize field by implicitly calling setters defined in
      % dagnn.Filter and here
      obj.size = obj.size ;
      obj.stride = obj.stride ;
      obj.pad = obj.pad ;
    end
  end
end
