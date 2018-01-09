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
      params_2 = params{1};
      params_final = params{1};
    %%==============================================binary
    %   [a,b,c,d]=size(params{1});
    %     for g=1:a
    %         for p=1:b
		%  params_1(g,p,:,:)=sign(params{1}(g,p,:,:));
    %         end       
    %     end
    %%==============================================tenary operation 
    %   %  sigma_layer = max(max(max(max(params{1}))));
    %   %  delta = 0.15*sigma_layer;
    %    [a,b,c,d]=size(params{1});
    %    Num_para = a*b*c*d;
    %    threshold = 0.7*sum(sum(sum(sum(abs(params{1})))))/Num_para;
    %     for g=1:a
    %         for p=1:b
		%           params_1(g,p,:,:)=sign(sign(params{1}(g,p,:,:)-threshold)+sign(params{1}(g,p,:,:)-threshold));
    %         end       
    %     end
    %     Num_I = sum(sum(sum(sum(abs(params_1)))));
    %     alpha_1 = sum(sum(sum(sum(abs(params{1})))))/Num_I;
     sigma_layer = max(max(max(max(params{1}))));
     threshold_1 = 0.43*sigma_layer/3;
     threshold_2 = 1.28*sigma_layer/3;
      [a,b,c,d] = size(params{1});
      % Num_para = a*b*c*d;
      % totalvalue = sum(sum(sum(sum(abs(params{1})))));
      % threshold_1 = 0.5*totalvalue/Num_para;
      % threshold_2 = 1.28*totalvalue/Num_para;
      for g=1:a
            for p=1:b
              params_1(g,p,:,:)=sign(sign(params{1}(g,p,:,:)-threshold_1)+sign(params{1}(g,p,:,:)-threshold_1));
		          params_2(g,p,:,:)=sign(sign(params{1}(g,p,:,:)-threshold_2)+sign(params{1}(g,p,:,:)-threshold_2));
              params_final= params_1+params_2;
            end       
      end
      Num_I_1  = sum(sum(sum(sum(abs(params_1)))));
      Num_I_2  = sum(sum(sum(sum(abs(params_2)))));
      Num_I = Num_I_1 + 3*Num_I_2; 
      % alpha_1 =  1.6*sum(sum(sum(sum(abs(params{1})))))/(Num_I);
      outputs{1} = vl_nnconv(...
        inputs{1}, params_final, params{2}, ...
        'pad', obj.pad, ...
        'stride', obj.stride, ...
        'dilate', obj.dilate, ...
        obj.opts{:}) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      if ~obj.hasBias, params{2} = [] ; end
      params_1 = params{1};
      params_2 = params{1};
      params_final = params{1};
    %%==============================================binary
    %   [a,b,c,d]=size(params{1});
    %     for g=1:a
    %         for p=1:b
		%  params_1(g,p,:,:)=sign(params{1}(g,p,:,:));
    %         end       
    %     end
    %%==============================================tenary operation 
    %   %  sigma_layer = max(max(max(max(params{1}))));
    %   %  delta = 0.15*sigma_layer;
    %    [a,b,c,d]=size(params{1});
    %    Num_para = a*b*c*d;
    %    threshold = 0.7*sum(sum(sum(sum(abs(params{1})))))/Num_para;
    %     for g=1:a
    %         for p=1:b
		%           params_1(g,p,:,:)=sign(sign(params{1}(g,p,:,:)-threshold)+sign(params{1}(g,p,:,:)-threshold));
    %         end       
    %     end
    %     Num_I = sum(sum(sum(sum(abs(params_1)))));
    %     alpha_1 = sum(sum(sum(sum(abs(params{1})))))/Num_I;
     sigma_layer = max(max(max(max(params{1}))));
     threshold_1 = 0.43*sigma_layer/3;
     threshold_2 = 1.28*sigma_layer/3;
      [a,b,c,d] = size(params{1});
      % Num_para = a*b*c*d;
      % totalvalue = sum(sum(sum(sum(abs(params{1})))));
      % threshold_1 = 0.5*totalvalue/Num_para;
      % threshold_2 = 1.28*totalvalue/Num_para;
      for g=1:a
            for p=1:b
              params_1(g,p,:,:)=sign(sign(params{1}(g,p,:,:)-threshold_1)+sign(params{1}(g,p,:,:)-threshold_1));
		          params_2(g,p,:,:)=sign(sign(params{1}(g,p,:,:)-threshold_2)+sign(params{1}(g,p,:,:)-threshold_2));
              params_final= params_1+params_2;
            end       
      end
      % Num_I_1  = sum(sum(sum(sum(abs(params_1)))));
      % Num_I_2  = sum(sum(sum(sum(abs(params_2)))));
      % Num_I = Num_I_1 + 3*Num_I_2; 
      % alpha_1 =  1.6*sum(sum(sum(sum(abs(params{1})))))/(Num_I);
      [derInputs{1}, derParams{1}, derParams{2}] = vl_nnconv(...
       inputs{1}, params_final, params{2}, derOutputs{1}, ...
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
