fc=315e9;                               % Carrier frequency
W=1e9;                                  % Bandwidth
ntap=400;                               % Number of taps
NumRx=10;                            % Number of receivers
maxDis=30;                              % Maximum distance between Tx and Rxs
maxHeightRx=1.5;                        % Maximum height of generated Rxs
minHeightRx=1;                          % Minimum height of generated Rxs
HeightTx=3;                             % Height of Tx
scenario='TWC_Indoor_Corridor_LOS';     % Simulation scenario configuration

fk=((fc-W/2)):W/ntap:(fc+W/2);
%maxDelay=1/W*ntap;
s = qd_simulation_parameters;                           % Set up simulation parameters
s.center_frequency = fc;                                % Set center frequency
s.use_absolute_delays = 1;                              % Include delay of the LOS path
s.show_progress_bars = 0;                               % Donot show progress bars

l = qd_layout(s);                                         % Create new QuaDRiGa layout
l.no_rx = NumRx;                                          % Set number of MTs
l.randomize_rx_positions( maxDis , HeightTx , HeightTx , 0 ,[]);  % 200 m radius, 1.5 m Rx height
l.set_scenario(scenario);                               

l.tx_position(3) = HeightTx;                            
l.tx_array = qd_arrayant( 'omni' );                     % Omni-directional BS antenna
l.rx_array = qd_arrayant( 'omni' );                     % Omni-directional MT antenna

tic
c = l.get_channels;                                     % Generate channel coefficients
toc

coeff = squeeze(cat( 5, c.coeff ));                     % Extract amplitudes and phases
delay = squeeze(cat( 5, c.delay ));                     % Extract path delays
aoa = c(1,1).par.AoA_cb;
eoa = c(1,1).par.EoA_cb;

for i = 2:NumRx
    aoa = cat(5,aoa,c(i,1).par.AoA_cb);
end

for i = 2:NumRx
    eoa = cat(5,eoa,c(i,1).par.EoA_cb);
end

eoa = squeeze(eoa);
aoa = squeeze(aoa);

m = size(aoa,1);
raw_H=squeeze(bsxfun(@times,coeff,exp(-1i*2*pi*reshape(kron(fk,delay),size(delay,1),size(delay,2),[]))));
Npath = size(delay,1);
rx = l.rx_position;
H = zeros(NumRx,Npath,4);
for i = 1:NumRx
    H(i,:,1) = abs(squeeze(coeff(:,i)));
    H(i,:,2) = eoa(:,i);
    H(i,:,3) = delay(:,i);
    H(i,:,4) = aoa(:,i);
end



H(:,:,1) = -20*log10(H(:,:,1));
maxP = max(max(H(:,:,1)));
maxDelay = max(max(H(:,:,3)));
maxTheta = 2*pi;
maxAoa = 2*pi;
maxEoa = 2*pi;



H(:,:,1) = mod(1-H(:,:,1)/maxP,1);  % amplitude
H(:,:,2) = mod((H(:,:,2)+360),360)/360; % eoa angle
H(:,:,3) = H(:,:,3)/maxDelay;  % delay
H(:,:,4) = mod((H(:,:,4)+360),360)/360; % aoa angle




for i = 1:NumRx
    d = squeeze(abs(H(i,:,3)));
    [t,p] = min(d);
    [~,I] = sort(d,'ascend');
    H(i,:,4)= mod(H(i,:,4)-H(i,p,4)+1,1);% relative angles for aoa
    H(i,:,2)= mod(H(i,:,2)-H(i,p,2)+1,1);% relative angles for eoa 
    H(i,:,:) = H(i,I,:);
end

H(:,:,4) = H(:,:,4)*360;
H(:,:,4) = (mod(H(:,:,4)+180,360))/360;

H(:,:,2) = H(:,:,2)*360;
H(:,:,2) = (mod(H(:,:,2)+180,360))/360;


rx = l.rx_position;
rx = permute(rx,[2,1]);
dis = sqrt(rx(:,1).^2+rx(:,2).^2);
maxd = max(abs(d));


save('../new_result/coeff_10000_path15_new_config_eoa2.mat', 'H','-v7.3')
save('../new_result/coeff_rx_position_path15_new_config_eoa2.mat','rx','dis','d','maxd','-v7.3');
save('../new_result/coeff_10000_maxPD_path15_new_config_eoa2.mat', 'maxP','maxDelay','-v7.3')




