fc=315e9;
c=3e8;
%% Normalization Factor
load('../result/coeff_10000_maxPD.mat') %3GPP
maxDelay0 = maxDelay; 
maxP0 = maxP;

load('../new_result/coeff_10000_maxPD_path15_new_config_eoa2.mat') %gan
maxDelay1 = maxDelay; 
maxP1 = maxP;

load('../twc_result/coeff_maxPD_corridor_new.mat') %ray-tracing
maxDelay2 = maxDelay;
maxP2 = maxP;

load('../new_result/new_coeff_21_maxPD.mat') %% measurement
maxDelay3 = maxDelay*1e-9;
maxP3 = maxP;


%% Position information
load('../result/coeff_rx_position_21.mat')
dis1 = dis*31;  %% real measurement


load('../new_result/coeff_rx_position_path15_new_config_eoa2.mat')
dis2 = dis;     %% 3GPP


load('../new_result/coeff_rx_position_corridor.mat')
dis3 = dis;     %% ray_tracing


load('../result/coeff_rx_position_10000.mat') %% 3Gpp original
dis4 = dis;



%% GAN generated channel
load('../new_result/before_tgan_generator_real21_coeff.mat')
channel1 = channel;

load('../new_result/after_tgan_generator_real21_coeff_l2norm_scale_180')
channel2 = channel;



% raytracing channel
load('../twc_result/coeff_corridor_ray_trace_new.mat')
H1 = H;


% GAN generated channel
%load('../twc_result/bench_tgan_generator_10000e_coeff_latent=32_r.mat','channel')
load('../twc_result/bench1_coeff_10000.mat','channel')
H2 = channel;


% real channel
load('../new_result/new_coeff_real_channel_21.mat')
H3 = H;




%% Testing


% for real channel
H3(:,:,1) = 10.^((H3(:,:,1)-1)*maxP3/20); %for testing real channel, H(:,:,1)-1 otherwise -H(:,:,1）
H3(:,:,2) = H3(:,:,2)*360;
H3(:,:,3) = H3(:,:,3)*maxDelay3;
H3(:,:,4) = H3(:,:,4)*360;



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




% TGAN Generated channel
channel1(:,:,1) = 10.^((channel1(:,:,1)-1)*maxP1/20); 
channel1(:,:,2) = channel1(:,:,2)*360;
channel1(:,:,3) = channel1(:,:,3)*maxDelay1;
channel1(:,:,4) = channel1(:,:,4)*360;


% TTGAN Generated channel
channel2(:,:,1) = 10.^((channel2(:,:,1)-1)*maxP3/20); 
channel2(:,:,2) = channel2(:,:,2)*360;
channel2(:,:,3) = channel2(:,:,3)*maxDelay3;
channel2(:,:,4) = channel2(:,:,4)*360;


%% SSIM Spread
H3_scale = H3;
H3_scale(:,:,4) = mod(H3_scale(:,:,4)+180,360); 
channel3 = H2(1:21,:,:);
NumRx = size(H3_scale,1);
k1=10^(-10)*rand([NumRx,36,401]); %-10

[pdap1,pdap_avg1] = pdap(H3_scale,k1); %real channel
[pdap2,pdap_avg2] = pdap(channel1,k1); %generated channel by T-GAN
[pdap3,pdap_avg3] = pdap(channel2,k1); %transfered channel by TT-GAN
[pdap4,pdap_avg4] = pdap(channel3,k1); %GAN channel



ssimv1 = zeros(20,1);
ssimv2 = zeros(20,1);
ssimv3 = zeros(20,1);
ssimv4 = zeros(20,1);
maxP =  200;
for i = 1:20
        ssimv1(i,1) = ssim(abs(squeeze(pdap1(i,:,:)))/maxP,abs(squeeze(pdap2(i,:,:)))/maxP);%T-GAN
        ssimv2(i,1) = ssim(abs(squeeze(pdap1(i,:,:)))/maxP,abs(squeeze(pdap3(i,:,:)))/maxP);%TT-GAN
        ssimv3(i,1) = ssim(abs(squeeze(pdap1(i,:,:)))/maxP,abs(squeeze(pdap4(i,:,:)))/maxP);%GAN
end

figure('color',[1,1,1],'DefaultAxesFontSize',25)
hold off
hold on
cdf3 = cdfplot(ssimv3);
cdf1 = cdfplot(ssimv1);
hold on
cdf2 = cdfplot(ssimv2);

l1 = legend('GAN','T-GAN','TT-GAN');
set(cdf1,'linewidth',3.0, 'linestyle','--','Color','#EDB120')
set(cdf2,'linewidth',3.0, 'linestyle','-','Color','#D95319')
set(cdf3,'linewidth',3.0,'linestyle','-.','Color','#77AC30')
set(l1,'FontSize',25, 'FontName','Arial')
set(gca,'LineWidth',1.5)
title('')
xlabel('SSIM','FontName','Arial','FontSize',25)
ylabel('Culmultative distribution function','FontName','Arial','FontSize',25)