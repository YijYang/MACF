function [results,drawCorrectPos] = MOSSEtracker(params)

ii = 1;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
V_Scale = 0;A_Scale = 0;V_Scale_Old = 0;
V_Pos_Y = 0;A_Pos_Y = 0;V_Pos_Y_Old = 0;
V_Pos_X = 0;A_Pos_X = 0;V_Pos_X_Old = 0;
predictScaLocation = 1.0;
trackedScaLocation = [1.0, 0];
i=1;j=1;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
padding = params.padding;
output_sigma_factor = params.output_sigma_factor;
lambda = params.lambda;
interp_pos_factor = params.interp_factor;
interp_sca_factor = params.interp_factor;
refinement_iterations = params.refinement_iterations;
translation_model_max_area = params.translation_model_max_area;
nScales = params.number_of_scales;
nScalesInterp = params.number_of_interp_scales;
scale_step = params.scale_step;
scale_sigma_factor = params.scale_sigma_factor;
scale_model_factor = params.scale_model_factor;
scale_model_max_area = params.scale_model_max_area;
interpolate_response = params.interpolate_response;
num_compressed_dim = params.num_compressed_dim;


s_frames = params.s_frames;
pos = floor(params.init_pos);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
minPosUpdateError = floor(min(pos/24));
isUpdate = 1;
isOcclusion = 0;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trackedPosLocation = pos;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawActualPos(1,3) = 1.0;
drawActualPos(1,2:-1:1) = pos;
drawPredictPos(1,3) = 1.0;
drawPredictPos(1,2:-1:1) = pos;
drawCorrectPos(1,3) = 1.0;
drawCorrectPos(1,2:-1:1) = pos;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

target_sz = floor(params.wsize * params.resize_factor);

visualization = params.visualization;

num_frames = numel(s_frames);

init_target_sz = target_sz;

if prod(init_target_sz) > translation_model_max_area
    currentScaleFactor = sqrt(prod(init_target_sz) / translation_model_max_area);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

%window size, taking padding into account
sz = floor( base_target_sz * (1 + padding ));

featureRatio = 4;

output_sigma = sqrt(prod(floor(base_target_sz/featureRatio))) * output_sigma_factor;
use_sz = floor(sz/featureRatio);
rg = circshift(-floor((use_sz(1)-1)/2):ceil((use_sz(1)-1)/2), [0 -floor((use_sz(1)-1)/2)]);
cg = circshift(-floor((use_sz(2)-1)/2):ceil((use_sz(2)-1)/2), [0 -floor((use_sz(2)-1)/2)]);

[rs, cs] = ndgrid( rg,cg);
y = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
yf = single(fft2(y));

interp_sz = size(y) * featureRatio;

cos_window = single(hann(floor(sz(1)/featureRatio))*hann(floor(sz(2)/featureRatio))' );


if nScales > 0
    scale_sigma = nScalesInterp * scale_sigma_factor;
    
    scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2)) * nScalesInterp/nScales;
    scale_exp_shift = circshift(scale_exp, [0 -floor((nScales-1)/2)]);
    
    interp_scale_exp = -floor((nScalesInterp-1)/2):ceil((nScalesInterp-1)/2);
    interp_scale_exp_shift = circshift(interp_scale_exp, [0 -floor((nScalesInterp-1)/2)]);
    
    scaleSizeFactors = scale_step .^ scale_exp;
    interpScaleFactors = scale_step .^ interp_scale_exp_shift;
    
    ys = exp(-0.5 * (scale_exp_shift.^2) /scale_sigma^2);
    ysf = single(fft(ys));
    scale_window = single(hann(size(ysf,2)))';
    
    %make sure the scale model is not to large, to save computation time
    if scale_model_factor^2 * prod(init_target_sz) > scale_model_max_area
        scale_model_factor = sqrt(scale_model_max_area/prod(init_target_sz));
    end
    
    %set the scale model size
    scale_model_sz = floor(init_target_sz * scale_model_factor);
    
    im = imread(s_frames{1});
    
    %force reasonable scale changes
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
    
    max_scale_dim = strcmp(params.s_num_compressed_dim,'MAX');
    if max_scale_dim
        s_num_compressed_dim = length(scaleSizeFactors);
    else
        s_num_compressed_dim = params.s_num_compressed_dim;
    end
end

% initialize the projection matrix
projection_matrix = [];

rect_position = zeros(num_frames, 4);

time = 0;

for frame = 1:num_frames
    %load image
    im = imread(s_frames{frame});
    
    tic();
    
    %do not estimate translation and scaling on the first frame, since we 
    %just want to initialize the tracker there
    if frame > 1
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        ii = ii+1;
        predictPosLocation  = predict(kalmanPosFilter);
        drawPredictPos(ii,2:-1:1) = predictPosLocation;
        pos = trackedPosLocation;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
        old_pos = inf(size(pos));
        iter = 1;
       
        %translation search
        while iter <= refinement_iterations && any(old_pos ~= pos)
            [xt_npca, xt_pca] = get_subwindow(im, pos, sz, currentScaleFactor);
            
            xt = feature_projection(xt_npca, xt_pca, projection_matrix, cos_window);
            xtf = fft2(xt);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %位置滤波器响应
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            responsef = sum(hf_num .* xtf, 3) ./ (hf_den + lambda);
            
            % if we undersampled features, we want to interpolate the
            % response so it has the same size as the image patch
            if interpolate_response > 0
                if interpolate_response == 2
                    % use dynamic interp size
                    interp_sz = floor(size(y) * featureRatio * currentScaleFactor);
                end
                
                responsef = resizeDFT2(responsef, interp_sz);
            end
            
            response = ifft2(responsef, 'symmetric');
            
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if params.occlusionDetect
                Resp = fftshift(response);
                maxResp = max(max(Resp));
                minResp = min(min(Resp));
                Resp = Resp./maxResp;
                Resp = Resp.^2;
                maxResp = max(max(Resp));
                minResp = min(min(Resp));
                Resp = Resp./maxResp;
                CSMR(i) = (maxResp - minResp)/(mean(mean(Resp - minResp)));
                tr = CSMR(i)/CSMR(1);
                if tr > 0.6
                    interp_pos_factor = 0.024;
                else
                    interp_pos_factor = 0.024*rs;
                end                
                i = i+1;
            end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
            [row, col] = find(response == max(response(:)), 1);
            disp_row = mod(row - 1 + floor((interp_sz(1)-1)/2), interp_sz(1)) - floor((interp_sz(1)-1)/2);
            disp_col = mod(col - 1 + floor((interp_sz(2)-1)/2), interp_sz(2)) - floor((interp_sz(2)-1)/2);
            
            switch interpolate_response
                case 0
                    translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor);
                case 1
                    translation_vec = round([disp_row, disp_col] * currentScaleFactor);
                case 2
                    translation_vec = [disp_row, disp_col];
            end
            
            old_pos = pos;
            pos = pos + translation_vec;
            
            drawActualPos(ii,2:-1:1) = pos; 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %calculate the speed of target in plane between current frame with pre frame
            if params.directMotionPredict    
                V_Pos_Y_Old = V_Pos_Y;
                V_Pos_Y = pos(1)-old_pos(1);
                V_Pos_X_Old = V_Pos_X;
                V_Pos_X = pos(2)-old_pos(2);
                
                drawActualPos(ii,2:-1:1) = pos;

                if frame > 2
                    A_Pos_Y = V_Pos_Y - V_Pos_Y_Old;
                    A_Pos_X = V_Pos_X - V_Pos_X_Old;
                end
            end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
            iter = iter + 1;
        end
        
        %scale search
        if nScales > 0
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        if params.kalmanScaFilter
            currentScaleFactor_DP = currentScaleFactor+V_Scale+0.5*A_Scale;
            predictScaLocation  = predict(kalmanScaFilter);
            detectedScaLocation = [currentScaleFactor_DP,0];
            trackedScaLocation  = correct(kalmanScaFilter, detectedScaLocation);
            drawCorrectPos(ii,3) = trackedScaLocation(1);
            currentScaleFactor = trackedScaLocation(1);  
            drawPredictPos(ii,3) = currentScaleFactor;
        end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                 
            %create a new feature projection matrix
            [xs_pca, xs_npca] = get_scale_subwindow(im,pos,base_target_sz,currentScaleFactor*scaleSizeFactors,scale_model_sz);
            
            xs = feature_projection_scale(xs_npca,xs_pca,scale_basis,scale_window);
            xsf = fft(xs,[],2);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %尺度滤波器响应
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
            scale_responsef = sum(sf_num .* xsf, 1) ./ (sf_den + lambda);
            
            interp_scale_response = ifft( resizeDFT(scale_responsef, nScalesInterp), 'symmetric');
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if params.occlusionDetect
                SResp = fftshift(interp_scale_response);
                SmaxResp = max(SResp);
                SminResp = min(SResp);
                SResp = SResp./SmaxResp;
                SResp = SResp.^2;
                SmaxResp = max(SResp);
                SminResp = min(SResp);
                SCSMR(j) = (SmaxResp - SminResp)/(mean(SResp - SminResp));
                str = SCSMR(j)/SCSMR(1);
                if str > 0.75
                    interp_pos_factor = 0.024;
                elseif str >0.3
                    interp_pos_factor = 0.024*srs;
                else
                    interp_pos_factor = 0;
                end                
                j = 2;
            end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
            recovered_scale_index = find(interp_scale_response == max(interp_scale_response(:)), 1);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if params.directMotionPredict    
                V_Scale_Old = V_Scale;
                V_Scale = currentScaleFactor * interpScaleFactors(recovered_scale_index) - currentScaleFactor;
                if frame > 2
                    A_Scale = V_Scale_Old - V_Scale;
                end
            end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
            %set the scale
            currentScaleFactor = currentScaleFactor * interpScaleFactors(recovered_scale_index);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%             
            drawActualPos(ii,3) = currentScaleFactor;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%             
            %adjust to make sure we are not to large or to small
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end
        end
    end
    
    %this is t1he training code used to update/initialize the tracker
    
    %Compute coefficients for the tranlsation filter
    if isUpdate || frame ==1
        [xl_npca, xl_pca] = get_subwindow(im, pos, sz, currentScaleFactor);
    end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %更新位置滤波器的分子
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if frame == 1
        h_num_pca = xl_pca;
        h_num_npca = xl_npca;
        
        % set number of compressed dimensions to maximum if too many
        num_compressed_dim = min(num_compressed_dim, size(xl_pca, 2));
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        %初始化卡尔曼滤波器:位置滤波
    if params.kalmanPosFilter
        initialPosLocation = pos;
        kalmanPosFilter = configureKalmanFilter(params.motionPosModel, ...
          initialPosLocation, params.initialPosEstimateError, ...
          params.motionPosNoise, params.measurementPosNoise);
    end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    else
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        %卡尔曼滤波器预测、更新:位置滤波
    if params.kalmanPosFilter
        detectedPosLocation = pos; 
        trackedPosLocation  = correct(kalmanPosFilter, detectedPosLocation);
        drawCorrectPos(ii,2:-1:1) = trackedPosLocation;
        EstimatePosError = min(abs(detectedPosLocation-predictPosLocation));
%         isUpdate = EstimatePosError < minPosUpdateError;
%         if params.occlusionDetect
%             isUpdate = isUpdate & isOcclusion;
%         end
    end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    if params.kalmanPosFilter
        if isUpdate
            h_num_pca = (1 - interp_pos_factor) * h_num_pca + interp_pos_factor * xl_pca;
            h_num_npca = (1 - interp_pos_factor) * h_num_npca + interp_pos_factor * xl_npca;
        end
    else 
        h_num_pca = (1 - interp_pos_factor) * h_num_pca + interp_pos_factor * xl_pca;
        h_num_npca = (1 - interp_pos_factor) * h_num_npca + interp_pos_factor * xl_npca;
    end
    end;
    
    if isUpdate || frame ==1
        data_matrix = h_num_pca;
    
        [pca_basis, ~, ~] = svd(data_matrix' * data_matrix);
        projection_matrix = pca_basis(:, 1:num_compressed_dim);
    
        hf_proj = fft2(feature_projection(h_num_npca, h_num_pca, projection_matrix, cos_window));
        hf_num = bsxfun(@times, yf, conj(hf_proj));
    
        xlf = fft2(feature_projection(xl_npca, xl_pca, projection_matrix, cos_window));
        new_hf_den = sum(xlf .* conj(xlf), 3);
    end
    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %更新位置滤波器的分母
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if frame == 1
        hf_den = new_hf_den;
    else
        if params.kalmanPosFilter
            if isUpdate;
                hf_den = (1 - interp_pos_factor) * hf_den + interp_pos_factor * new_hf_den;
            end
        else
            hf_den = (1 - interp_pos_factor) * hf_den + interp_pos_factor * new_hf_den;
        end
    end
    
    %Compute coefficents for the scale filter
    if nScales > 0
   
        %create a new feature projection matrix
        [xs_pca, xs_npca] = get_scale_subwindow(im, pos, base_target_sz, currentScaleFactor*scaleSizeFactors, scale_model_sz);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %更新尺度滤波器的分子
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if frame == 1
            s_num = xs_pca;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        %初始化卡尔曼滤波器:尺度滤波
            if params.kalmanScaFilter
                initialScaLocation = [currentScaleFactor,0];
                kalmanScaFilter = configureKalmanFilter(params.motionScaModel, ...
                initialScaLocation, params.initialScaEstimateError, ...
                params.motionScaNoise, params.measurementScaNoise);
            end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        else
            if params.kalmanScaFilter
                if isUpdate
                    s_num = (1 - interp_sca_factor) * s_num + interp_sca_factor * xs_pca;
                end
            else 
                s_num = (1 - interp_sca_factor) * s_num + interp_sca_factor * xs_pca;
            end
        end;
        
        bigY = s_num;
        bigY_den = xs_pca;
        
        if max_scale_dim
            [scale_basis, ~] = qr(bigY, 0);
            [scale_basis_den, ~] = qr(bigY_den, 0);
        else
            [U,~,~] = svd(bigY,'econ');
            scale_basis = U(:,1:s_num_compressed_dim);
        end
        scale_basis = scale_basis';
        
        %create the filter update coefficients
        sf_proj = fft(feature_projection_scale([],s_num,scale_basis,scale_window),[],2);
        sf_num = bsxfun(@times,ysf,conj(sf_proj));
        
        xs = feature_projection_scale(xs_npca,xs_pca,scale_basis_den',scale_window);
        xsf = fft(xs,[],2);
        new_sf_den = sum(xsf .* conj(xsf),1);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %更新位置滤波器的分子
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if frame == 1
            sf_den = new_sf_den;
        else
            if params.kalmanScaFilter
                if isUpdate
                    sf_den = (1 - interp_sca_factor) * sf_den + interp_sca_factor * new_sf_den;
                end
            else
                sf_den = (1 - interp_sca_factor) * sf_den + interp_sca_factor * new_sf_den;
            end
        end;
    end
    %pos = trackedPosLocation;
    target_sz = floor(base_target_sz * currentScaleFactor);
    
    %save position and calculate FPS
    rect_position(frame,:) = [pos([2,1]) - floor(target_sz([2,1])/2), target_sz([2,1])];
    
    time = time + toc();
    
    %visualization(可视化)
    if visualization == 1
        rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        if frame == 1
            hfig1 = figure;
            im_handle = imshow(im, 'Border','tight', 'InitialMag', 100 + 100 * (length(im) < 500));
            set(hfig1, 'position', 0.8*get(0,'ScreenSize')); %设置显示屏大小
            rect_handle = rectangle('Position',rect_position_vis, 'EdgeColor','b','LineWidth',3);
            text_handle = text(10, 10, ['#' int2str(frame)]);
            %rect_handle1 = rectangle('Position',temp.res(1,:), 'EdgeColor','r');
            %text_handle1 = text(temp.res(1,1), temp.res(1,2), 'SRDCF');
            text_handle2 = text(rect_position_vis(1), rect_position_vis(2)-10, 'UAV');%显示UAV
            set(text_handle, 'color', [1 1 0],'FontSize',20);
            set(text_handle2, 'color', [0 0 1]);
            %saveas(gcf,['F:\1研究生\研一\photo\Temp\2\1','.jpg']);%保存标定好的图像序列
        else
            try
                set(im_handle, 'CData', im)
                set(rect_handle, 'Position', rect_position_vis,'LineWidth',3)
                %set(rect_handle1, 'Position', temp.res(frame,:))
                set(text_handle, 'string', ['#' int2str(frame)]);
                %set(text_handle1, 'Position', [temp.res(frame,1), temp.res(frame,2)]);
                set(text_handle2, 'Position', [rect_position_vis(1), rect_position_vis(2)-10]);
                %i=i+1;
                %saveas(gcf,['F:\1研究生\研一\photo\Temp\2\',int2str(i),'.jpg']);
                %pause(0.01);
            catch
                return
            end
        end
        
        drawnow
        %pause
    end
end

figure(2),
plot(drawPredictPos(:,1),drawPredictPos(:,2),'r',drawActualPos(:,1),drawActualPos(:,2),'b',drawCorrectPos(:,1),drawCorrectPos(:,2),'y');
legend('Predict position','Actual position','Correct position');
set(gca,'xaxislocation','top','yaxislocation','left','ydir','reverse') % set origin position  
text(drawPredictPos(1,1),drawPredictPos(1,2),'*','color','r');
text(drawPredictPos(1,1)+1,drawPredictPos(1,2)+1,['initial pos'],'color','r');
text(drawPredictPos(num_frames,1),drawPredictPos(num_frames,2),'*','color','b');
text(drawPredictPos(num_frames,1)+1,drawPredictPos(num_frames,2)+1,['final pos'],'color','b');
figure(3),
plot(drawPredictPos(:,1),drawPredictPos(:,2),'r');
legend('Predict position');
set(gca,'xaxislocation','top','yaxislocation','left','ydir','reverse') % set origin position  
text(drawPredictPos(1,1),drawPredictPos(1,2),'*','color','r');
text(drawPredictPos(1,1)+1,drawPredictPos(1,2)+1,['initial pos'],'color','r');
text(drawPredictPos(num_frames,1),drawPredictPos(num_frames,2),'*','color','b');
text(drawPredictPos(num_frames,1)+1,drawPredictPos(num_frames,2)+1,['final pos'],'color','b');
figure(4),
plot(drawActualPos(:,1),drawActualPos(:,2),'b');
legend('Actual position');
set(gca,'xaxislocation','top','yaxislocation','left','ydir','reverse') % set origin position  
text(drawPredictPos(1,1),drawPredictPos(1,2),'*','color','r');
text(drawPredictPos(1,1)+1,drawPredictPos(1,2)+1,['initial pos'],'color','r');
text(drawPredictPos(num_frames,1),drawPredictPos(num_frames,2),'*','color','b');
text(drawPredictPos(num_frames,1)+1,drawPredictPos(num_frames,2)+1,['final pos'],'color','b');
figure(5),
plot(drawCorrectPos(:,1),drawCorrectPos(:,2),'y');
legend('Correct position');
set(gca,'xaxislocation','top','yaxislocation','left','ydir','reverse') % set origin position  
text(drawCorrectPos(1,1),drawCorrectPos(1,2),'*','color','r');
text(drawCorrectPos(1,1)+1,drawCorrectPos(1,2)+1,['initial pos'],'color','r');
text(drawCorrectPos(num_frames,1),drawCorrectPos(num_frames,2),'*','color','b');
text(drawCorrectPos(num_frames,1)+1,drawCorrectPos(num_frames,2)+1,['final pos'],'color','b');

figure(6),
h2 = plot3(drawPredictPos(:,3),drawPredictPos(:,1),drawPredictPos(:,2),'r',drawActualPos(:,3),drawActualPos(:,1),drawActualPos(:,2),'b',drawCorrectPos(:,3),drawCorrectPos(:,1),drawCorrectPos(:,2),'y');
legend('Predict position','Actual position','Correct position');
rotate(h2,[1,0,0],180);
set(gca,'xdir','reverse');

figure(7),
h3 = plot3(drawPredictPos(:,3),drawPredictPos(:,1),drawPredictPos(:,2),'r');
legend('Predict position');
rotate(h3,[1,0,0],180);
set(gca,'xdir','reverse');

figure(8),
h4 = plot3(drawActualPos(:,3),drawActualPos(:,1),drawActualPos(:,2),'b');
legend('Actual position');
rotate(h4,[1,0,0],180);
set(gca,'xdir','reverse');

figure(9),
h4 = plot3(drawCorrectPos(:,3),drawCorrectPos(:,1),drawCorrectPos(:,2),'y');
legend('Correct position');
rotate(h4,[1,0,0],180);
set(gca,'xdir','reverse');


fps = numel(s_frames) / time;

% disp(['fps: ' num2str(fps)])

results.type = 'rect';
results.res = rect_position;
results.fps = fps;
end
