function [pdap,pdap_mean] = pdap(c,k)


fc=315e9;                               % Carrier frequency
W=1e9;                                  % Bandwidth
ntap=400;                               % Number of taps
fk=((fc-W/2)):W/ntap:(fc+W/2);
NumRx = size(c,1);

delay = c(:,:,3);
delay = permute(delay,[2,1]);
coeff = c(:,:,1).*exp(1i*c(:,:,2));
coeff= permute(coeff,[2,1]);
aoa = c(:,:,4)+180;
aoa = permute(aoa,[2,1]);
raw_H=bsxfun(@times,coeff,exp(-1i*2*pi*reshape(kron(fk,delay),size(delay,1),size(delay,2),[])));

pdap = zeros(NumRx,36,401);
for i = 1:NumRx
for j=1:36
    pdap(i,j,:)=ifft(squeeze(sum(raw_H(split_angle(aoa(1:15,i))==j,i,:),1 )));
end
end
 
c1 = squeeze(abs(pdap(:,:,:)))+k; % added noise floor
pdap_mean = squeeze((20*log10(mean(c1))));
pdap = 20*log10(c1);


% plot the PDAP
figure('color',[1,1,1],'DefaultAxesFontSize',20)
[X,Y] = meshgrid(1:401,(1:36)*10);
surf(X,Y,pdap_mean);
xlim([0,401])
ylim([0,361])
xlabel('Delay [ns]','FontName','Times New Roman','FontSize',20)
ylabel('AoA [degree]','FontName','Times New Roman','FontSize',20)
zlabel('Power [dB]','FontName','Times New Roman','FontSize',20)

end

