%%MACF

close all;
% clear all;

%choose the path to the videos (you'll be able to choose one with the GUI)
base_path = '.\sequences\';
%base_path = 'F:\1研究生\研二\目标跟踪\FDSST\tracker_benchmark_v1.0\Benchmark\';
%Auto choose target%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%hfig = figure;
%imshow(imread('C:\Users\yang\my_fDSST_code\sequences\UAV\img\img00001.jpg'));
%set(hfig, 'position', 0.9*get(0,'ScreenSize')); %设置显示屏大小
%h = imrect;
%target=uint16(h.getPosition);
%fid = fopen('C:\Users\yang\my_fDSST_code\sequences\UAV\dog1_gt.txt','w');
%fprintf(fid,'%g,',target);
%fclose(fid);
%close all;

%parameters according to the paper
params.padding = 1.8;                   % extra area surrounding the target
params.output_sigma_factor = 1/16;		% standard deviation for the desired translation filter output
params.scale_sigma_factor = 1/16;       % standard deviation for the desired scale filter output
params.lambda = 2e-2;					% regularization weight (denoted "lambda" in the paper)
params.interp_factor = 0.024;			% tracking model learning rate (denoted "eta" in the paper)
params.num_compressed_dim = 18;         % 压缩特征维数the dimensionality of the compressed features
params.refinement_iterations = 1;       % 用于细化帧中的结果位置的迭代次数number of iterations used to refine the resulting position in a frame
params.translation_model_max_area = inf;% maximum area of the translation model
params.interpolate_response = 1;        % interpolation method for the translation scores
params.resize_factor = 1;               % initial resize

params.number_of_scales = 17;           % number of scale levels
params.number_of_interp_scales = 33;    % number of scale levels after interpolation（插值）
params.scale_model_factor = 1.0;        % relative size of the scale sample
params.scale_step = 1.03;               % Scale increment factor (denoted "a" in the paper)
params.scale_model_max_area = 512;      % the maximum size of scale examples
params.s_num_compressed_dim = 'MAX';    % number of compressed scale feature dimensions

params.visualization = 1;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
params.occlusionDetect = 1;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
params.directMotionPredict = 1;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Kalman Pos Filter params
params.motionPosModel           = 'ConstantAcceleration';
params.initialPosEstimateError  = 1E5 * ones(1, 3);
params.motionPosNoise           = [25, 10, 1];
params.measurementPosNoise      = 25;
params.kalmanPosFilter = 1;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Kalman Scale Filter params
params.motionScaModel           = 'ConstantAcceleration';%'ConstantVelocity';
params.initialScaEstimateError  = 1E3 * ones(1, 3);%[200, 50];
params.motionScaNoise           = [2.5, 1.0, 0.1];%[100, 25];
params.measurementScaNoise      = 2.5;%100;
params.kalmanScaFilter = 1;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ask the user for the video
video_path = choose_video(base_path);
if isempty(video_path), return, end  %user cancelled
[img_files, pos, target_sz, ground_truth, video_path] = ...
	load_video_info(video_path);

params.init_pos = floor(pos([2 1])) + floor(target_sz([2 1])/2);%取整函数
params.wsize = floor(target_sz([2 1]));
params.s_frames = img_files;
params.video_path = video_path;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[results,prediction]= MACF(params);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
positions = results.res;
fps = results.fps;

% calculate precisions
% [distance_precision, PASCAL_precision, average_center_location_error] = ...
%    compute_performance_measures(positions, ground_truth);
fprintf('Speed: %.3g fps\n',fps);
% fprintf('Center Location Error: %.3g pixels\nDistance Precision: %.3g %%\nOverlap Precision: %.3g %%\nSpeed: %.3g fps\n', ...
%    average_center_location_error, 100*distance_precision, 100*PASCAL_precision, fps);
