%% Normalization Factor
load('../result/coeff_10000_maxPD.mat') %3GPP
maxDelay0 = maxDelay; 
maxP0 = maxP;

load('../result/coeff_10000_maxPD_path15_new_config_eoa2.mat') %gan
maxDelay1 = maxDelay; 
maxP1 = maxP;

load('../result/coeff_maxPD_corridor_new.mat') %ray-tracing
maxDelay2 = maxDelay;
maxP2 = maxP;

load('../result/new_coeff_21_maxPD.mat') %% measurement
maxDelay3 = maxDelay*1e-9;
maxP3 = maxP;


%% Position information
load('../result/coeff_rx_position_21.mat')
dis1 = dis*31;  %% real measurement


load('../result/coeff_rx_position_path15_new_config_eoa2.mat')
dis2 = dis;     %% 3GPP


load('../result/coeff_rx_position_corridor.mat')
dis3 = dis;     %% ray_tracing


load('../result/coeff_rx_position_10000.mat') %% 3Gpp position
dis4 = dis;

fc=315e9;
c=3e8;

%% TGAN generated channel
load('../new_result/before_tgan_generator_real21_coeff.mat')
channel1 = channel;

%% TTGAN generated channel
load('../new_result/after_tgan_generator_real21_coeff_l2norm_scale_180')
channel2 = channel;



% raytracing channel
load('../result/coeff_corridor_ray_trace_new.mat')
H1 = H;

% 3GPP Genereated Channel by LOS_1
% load('../result/coeff_10000.mat')
% H2 = H;
%load('../result/coeff_10000_path15_new_config_eoa2.mat')
%H2 = H;

% benchmark generated channel
load('../result/bench_tgan_generator_10000e_coeff_latent=32_r.mat','channel')
H2 = channel;


% real channel not provided
load('../result/new_coeff_real_channel_21.mat')
H3 = H;




%% Testing


% for real channel
H3(:,:,1) = 10.^((H3(:,:,1)-1)*maxP3/20); 
H3(:,:,2) = H3(:,:,2)*360;
H3(:,:,3) = H3(:,:,3)*maxDelay3;
H3(:,:,4) = H3(:,:,4)*360;
%H3(:,:,4) = H3(:,:,2);


% for ray tracing channel
H1(:,:,1) = 10.^((H1(:,:,1)-1)*maxP2/20); 
H1(:,:,2) = H1(:,:,2)*360;
H1(:,:,3) = H1(:,:,3)*maxDelay2;
H1(:,:,4) = H1(:,:,4)*360;



% for benchmark GAN channel
H2(:,:,1) = 10.^((H2(:,:,1)-1)*maxP1/20);
H2(:,:,2) = H2(:,:,2)*360;
H2(:,:,3) = H2(:,:,3)*maxDelay1;
H2(:,:,4) = H2(:,:,4)*360;
H2(:,1,1) = c./(4*pi*fc.*dis2);
H2(:,1,3) = dis2/c;
for i =1:size(H2,1)
    H2(i,H2(i,:,1)>H2(i,1,1),3)=maxDelay1;
    H2(i,H2(i,:,1)>H2(i,1,1),1)=10.^(-maxP1/20);
    H2(i,H2(i,:,3)<H2(i,1,3),1)=10.^(-maxP1/20);
    H2(i,H2(i,:,3)<H2(i,1,3),3)=maxDelay1;
end



% TGAN Generated channel


channel1(:,:,1) = 10.^((channel1(:,:,1)-1)*maxP1/20); 
channel1(:,:,2) = channel1(:,:,2)*360;
channel1(:,:,3) = channel1(:,:,3)*maxDelay1;
channel1(:,:,4) = channel1(:,:,4)*360;





% TTGAN Generated channel

channel2(:,:,1) = 10.^((channel2(:,:,1)-1)*maxP3/20); %after transfering TT-GAN
channel2(:,:,2) = channel2(:,:,2)*360;
channel2(:,:,3) = channel2(:,:,3)*maxDelay3;
channel2(:,:,4) = channel2(:,:,4)*360;


%% SSIM Spread
H3_scale = H3;
H3_scale(:,:,4) = mod(H3_scale(:,:,4)+180,360); 
NumRx = size(H3_scale,1);
k1=10^(-8)*rand([NumRx,36,401]); 

[pdap1,pdap_avg1] = pdap(H3_scale,k1); %real channel
[pdap2,pdap_avg2] = pdap(channel1,k1); %generated channel by T-GAN
[pdap3,pdap_avg3] = pdap(channel2,k1); %transfered channel by TT-GAN

%% SINGLE PDAP
figure('color',[1,1,1],'DefaultAxesFontSize',25,'DefaultAxesFontName','Arial')
mesh(1:401,(0:1:35)*10,squeeze(pdap3(1,:,:)))
xlim([0,401])
ylim([0,361])
xlabel('Delay [ns]','FontName','Arial','FontSize',25)
ylabel('AoA [degree]','FontName','Arial','FontSize',25)
zlabel('Power [dB]','FontName','Arial','FontSize',25)
set(gca,'LineWidth',1.5)
grid off





%% SINGLE PDAP
figure('color',[1,1,1],'DefaultAxesFontSize',20)
subplot(2,2,1) %% real channel
mesh(1:401,(0:1:35)*10,squeeze(pdap1(1,:,:)))
xlim([0,401])
ylim([0,361])
xlabel('Delay [ns]','FontName','Arial','FontSize',20)
ylabel('AoA [degree]','FontName','Arial','FontSize',20)
zlabel('Power [dB]','FontName','Arial','FontSize',20)
set(gca,'linewidth',1.5)
grid off

subplot(2,2,3) %% 3GPP
mesh(1:401,(0:1:35)*10,squeeze(pdap4(1,:,:)))
xlim([0,401])
ylim([0,361])
xlabel('Delay [ns]','FontName','Arial','FontSize',20)
ylabel('AoA [degree]','FontName','Arial','FontSize',20)
zlabel('Power [dB]','FontName','Arial','FontSize',20)
set(gca,'linewidth',1.5)
grid off


subplot(2,2,4) %% T-GAN
mesh(1:401,(0:1:35)*10,squeeze(pdap2(1,:,:)))
xlim([0,401])
ylim([0,361])
xlabel('Delay [ns]','FontName','Arial','FontSize',20)
ylabel('AoA [degree]','FontName','Arial','FontSize',20)
zlabel('Power [dB]','FontName','Arial','FontSize',20)
set(gca,'linewidth',1.5)
grid off

subplot(2,2,2) %% TT-GAN
mesh(1:401,(0:1:35)*10,squeeze(pdap3(1,:,:)))
xlim([0,401])
ylim([0,361])
xlabel('Delay [ns]','FontName','Arial','FontSize',20)
ylabel('AoA [degree]','FontName','Arial','FontSize',20)
zlabel('Power [dB]','FontName','Arial','FontSize',20)
set(gca,'linewidth',1.5)
grid off

rmse_single0 = sqrt(sum((pdap1-pdap4).^2,[2,3])/(401*36));
rmse_single1 = sqrt(sum((pdap1-pdap2).^2,[2,3])/(401*36));
rmse_single2 = sqrt(sum((pdap1-pdap3).^2,[2,3])/(401*36));


%% Average PDAP

figure('color',[1,1,1],'DefaultAxesFontSize',20)
subplot(1,3,1) %% real channel
[X,Y] = meshgrid(1:401,(1:36)*10);
surf(X,Y,squeeze(pdap_avg1(:,:)));
xlim([0,401])
ylim([0,361])
xlabel('Delay [ns]','FontName','Times New Roman','FontSize',20)
ylabel('AoA [degree]','FontName','Times New Roman','FontSize',20)
zlabel('Power [dB]','FontName','Times New Roman','FontSize',20)

subplot(1,3,2) %% 3GPP
[X,Y] = meshgrid(1:401,(1:36)*10);
surf(X,Y,squeeze(pdap_avg4(:,:)));
xlim([0,401])
ylim([0,361])
xlabel('Delay [ns]','FontName','Times New Roman','FontSize',20)
ylabel('AoA [degree]','FontName','Times New Roman','FontSize',20)
zlabel('Power [dB]','FontName','Times New Roman','FontSize',20)


subplot(1,3,3) %% T-GAN
[X,Y] = meshgrid(1:401,(1:36)*10);
surf(X,Y,squeeze(pdap_avg3(:,:)));
xlim([0,401])
ylim([0,361])
xlabel('Delay [ns]','FontName','Times New Roman','FontSize',20)
ylabel('AoA [degree]','FontName','Times New Roman','FontSize',20)
zlabel('Power [dB]','FontName','Times New Roman','FontSize',20)


rmse1 = sqrt(sum(sum((pdap_avg1-pdap_avg4).^2))/(401*36));
rmse2 = sqrt(sum(sum((pdap_avg1-pdap_avg2).^2))/(401*36));
