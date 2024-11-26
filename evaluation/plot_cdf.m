fc=315e9;
c=3e8;

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


load('../result/coeff_rx_position_10000.mat') %% 3Gpp original
dis4 = dis;



%% TGAN and TTGAN generated channel
load('../result/d1_10000e_tgan_generator_scale_180_l2norm_before_transfer_twc_eoa2.mat','channel') %T-GAN
channel1 = channel;

load('../result/d1_10000e_tgan_generator_scale_180_l2norm_after_transfer_twc_eoa2_01.mat','channel') %TT-GAN
channel2 = channel;



% raytracing channel
load('../result/coeff_corridor_ray_trace_new.mat')
H1 = H;


% benchmark GAN generated channel
load('../result/bench_tgan_generator_10000e_coeff_latent=32_r.mat','channel')
H2 = channel;


% real channel
load('../result/new_coeff_real_channel_21.mat') %%not provided measurement
H3 = H;




%% Testing


% for real channel
H3(:,:,1) = 10.^((H3(:,:,1)-1)*maxP3/20); 
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


% TT-GAN Generated channel
channel2(:,:,1) = 10.^((channel2(:,:,1)-1)*maxP3/20); 
channel2(:,:,2) = channel2(:,:,2)*360;
channel2(:,:,3) = channel2(:,:,3)*maxDelay3;
channel2(:,:,4) = channel2(:,:,4)*360;





coeff1 = H3(:,:,1).*exp(1i*H3(:,:,2));
coeff1 = permute(coeff1,[2,1]);


coeff2 = channel1(:,:,1).*exp(1i*channel1(:,:,2));
coeff2 = permute(coeff2,[2,1]);

coeff3 = channel2(:,:,1).*exp(1i*channel2(:,:,2));
coeff3 = permute(coeff3,[2,1]);

coeff4 = H2(:,:,1).*exp(1i*H2(:,:,2));
coeff4 = permute(coeff4,[2,1]);

coeff5 = H1(:,:,1).*exp(1i*H1(:,:,2));
coeff5 = permute(coeff5,[2,1]);


[rms1,aoa1] = calculate_rms(H3); %real channel
[rms2,aoa2] = calculate_rms(channel1); %generated channel by T-GAN
[rms5,aoa5] = calculate_rms(channel2); %transfered channel by TT-GAN
[rms6,aoa6] = calculate_rms(H2); %3GPP channel
[rms7,aoa7] = calculate_rms(H1); %raytracing channel 

power1_los = 10*log10(abs(coeff1(1,:).^2));
power2_los = 10*log10(abs(coeff2(1,:).^2));
power3_los = 10*log10(abs(coeff3(1,:).^2));
power4_los = 10*log10(abs(coeff4(1,:).^2));
power5_los = 10*log10(abs(coeff5(1,:).^2));


power1 = sum(abs(coeff1).^2);
power_db1 = 10*log10(power1);
power2 = sum(abs(coeff2).^2);
power_db2 = 10*log10(power2);
power3 = sum(abs(coeff3).^2);
power_db3 = 10*log10(power3);
power4 = sum(abs(coeff4).^2);
power_db4 = 10*log10(power4);
power5 = sum(abs(coeff5).^2);
power_db5 = 10*log10(power5);

%%  path loss exponent
reference_dis = 1;
fspl = fri(reference_dis,fc);
relative_distance_3gpp = log10(dis2/reference_dis);

ple_3gpp = PLE(dis2,power_db4,fc);
ple_raytracing = PLE(dis3,power_db5,fc);
ple_measurement = PLE(dis1,power_db1,fc);
ple_tgan = PLE(dis2,power_db2,fc);
ple_ttgan = PLE(dis2,power_db3,fc);

figure('color',[1,1,1],'DefaultAxesFontSize',25)
scatter(dis1,-power_db1,'MarkerEdgeColor','#A2142F')
set(gca,'xscale','log')
hold on
grid on
semilogx(dis1,log10(dis1)*ple_measurement+fspl,'LineWidth',3.0,'linestyle',':','Color','#7E2F8E')
semilogx(dis1,log10(dis1)*ple_raytracing+fspl,'LineWidth',3.0,'linestyle','-.','Color','#0072BD')
semilogx(dis1,log10(dis1)*ple_3gpp+fspl,'LineWidth',3.0,'linestyle','--','Color','#77AC30')
semilogx(dis1,log10(dis1)*ple_tgan+fspl,'LineWidth',3.0,'linestyle','-','Color','#EDB120')
semilogx(dis1,log10(dis1)*ple_ttgan+fspl,'LineWidth',3.0,'linestyle','-','Color','#D95319')
l1=legend('Measurement','Measurement-CI','Ray-tracing','GAN','T-GAN','TT-GAN');
a1 = -power_db1;
a2 = log10(dis1)*ple_3gpp+fspl;
mse_error = sum((a1-a2').^2)/length(a1);


title('')
set(l1,'FontSize',25, 'FontName','Arial')
xlabel('Distance [m]','FontName','Arial','FontSize',25)
ylabel('Path Loss [dB]','FontName','Arial','FontSize',25)
set(gca,'LineWidth',1.5)

mse_ci = sum((log10(dis1)*ple_measurement+fspl+power_db1').^2)/21;
axis([1,30,90,110])


%% AOA spread
figure('color',[1,1,1],'DefaultAxesFontSize',25)
ecdf(aoa1,'Bounds','on','Alpha',0.01);
hold on
cdf2 = cdfplot(aoa7); %ray_tracing
cdf3 = cdfplot(aoa6); %3gpp
cdf4 = cdfplot(aoa2); %T-gan
cdf5 = cdfplot(aoa5); %TT-gan
set(cdf2,'linewidth',3.0,'linestyle','--','Color','#0072BD')
set(cdf3,'linewidth',3.0,'linestyle','-.','Color','#77AC30')
set(cdf4,'linewidth',3.0,'linestyle','--','Color','#EDB120')
set(cdf5,'linewidth',3.0,'linestyle','-','Color','#D95319')
l1=legend('Measurement','Lower confidence bound','Upper confidence bound','Ray-tracing','GAN','T-GAN','TT-GAN');

title('')
grid on
set(l1,'FontSize',25, 'FontName','Arial')
xlabel('Angular spread [degree]','FontName','Arial','FontSize',25)
ylabel('Cumulative distribution function','FontName','Arial','FontSize',25)
set(gca,'linewidth',1.5)


%% delay spread
figure('color',[1,1,1],'DefaultAxesFontSize',25)
ecdf(rms1,'Bounds','on','Alpha',0.01);
hold on
cdf2=cdfplot(rms7); %rms7
rms6(rms6>40)=[];
cdf3 = cdfplot(rms6);
cdf4 = cdfplot(rms2);
cdf5 = cdfplot(rms5);
set(cdf2,'linewidth',3.0,'linestyle','--','Color','#0072BD')
set(cdf3,'linewidth',3.0, 'linestyle','-.','Color','#77AC30')
set(cdf4,'linewidth',3.0,'linestyle','--','Color','#EDB120')
set(cdf5,'linewidth',3.0,'linestyle','-','Color','#D95319')
l1=legend('Measurement','Lower confidence bound','Upper confidence bound','Ray-tacing','GAN','T-GAN','TT-GAN');
title('')
set(l1,'FontSize',25, 'FontName','Arial')
xlabel('Delay spread [ns]','FontName','Arial','FontSize',25)
ylabel('Cumulative probability function','FontName','Arial','FontSize',25)
set(gca,'linewidth',1.5)

%% Power spread
figure('color',[1,1,1],'DefaultAxesFontSize',20)
cdf1=cdfplot(power_db1); %measurement
hold on
cdf2 = cdfplot(power_db5); %ray-tacing
cdf3 = cdfplot(power_db4); %3gpp
cdf4=cdfplot(power_db2); %T-GAN
set(cdf1,'linewidth',3.0, 'linestyle','-')
set(cdf2,'linewidth',3.0,'linestyle','--')
cdf5=cdfplot(power_db3); %TT-GAN
set(cdf3,'linewidth',3.0, 'linestyle','-.')
set(cdf4,'linewidth',3.0,'linestyle','--')
set(cdf5,'linewidth',3.0,'linestyle','--')

l1=legend('Measurement','Ray-tracing','GAN','T-GAN','TT-GAN');
title('')
set(l1,'FontSize',25, 'FontName','Times New Roman')
xlabel('Total Power [dB]','FontName','Times New Roman','FontSize',25)
ylabel('Cumulative probability function','FontName','Times New Roman','FontSize',25)




