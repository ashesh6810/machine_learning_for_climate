clc;
clear all;
close all;
load coastlines

load ('/Users/ashes/Development/clustering/ExtremeEvents/Z500/grid.mat');

load(['TanomalyClustering_smallbox.mat'],'Psi','PsiF','idx');

nC=4;
samples=21;
s_tr=floor(3*samples/4);
s_ts=floor(samples/4);
lat_north_index=96;
lat_south_index=31;
lon_west_index=157;
lon_east_index=253;
lat1=lat(97:end);
lat11=lat1(31:65);
[qx,qy]=meshgrid(lon(lon_west_index:lon_east_index),lat11);

LABELS=zeros(nC*samples,nC);

for j=1:nC
    count=1;
 for i=1:length(idx)
     
    if (idx(i)==j)
     cluster{j}(:,count)=Psi(:,i);
     count=count+1;
    end
 end
end

    false_center=mean(PsiF,2);



for j=1:nC
    class=squeeze(cluster{j});
    Z500_class_centre(:,j)=mean(class,2);
    clear class;
end    
figure

for j=1:nC
    Z=reshape(Z500_class_centre(:,j),97,35);
    subplot(2,3,j)
    contourf(qx',qy',Z,10,'LineColor','k','LineWidth',2);%caxis([-max(max(abs(Z))) max(max(abs(Z)))])     
    hold on;

    plot(coastlon+360,coastlat,'Linewidth',1,'Color','k');
    xlim([195 315])
    ylim([25 66])
end

figure (1)

Z=reshape(false_center,97,35);
subplot(2,3,5)
    contourf(qx',qy',Z,10,'LineColor','k','LineWidth',2);%caxis([-max(max(abs(Z))) max(max(abs(Z)))])     
    hold on;

    plot(coastlon+360,coastlat,'Linewidth',1,'Color','k');
    xlim([195 315])
    ylim([25 66])

% count=1;
% for j=1:nC
%     for i=1:size(cluster{j},2)
%         Z=reshape(cluster{j}(:,i),97,66);
%         h=figure(1);
%         contourf(qx',qy',Z,10,'LineColor','k','LineWidth',2)
%         im=frame2im(getframe(gca));
%         close(h);
%         im_rs(:,count)=reshape(imresize(double(im),[28,28]),28*28*3,1)/255;
%         
%         LABELS(count,j)=1;
%         count=count+1;
%         if(i>=samples)
%             break;
%         end
% 
%         
%     end
% end



% for i=1:nC
%    
%   X_train{i}=im_rs(:,(i-1)*samples+1:floor(3*(i*samples-(i-1)*samples)/4)+(i-1)*samples+1);
%   Y_train{i}=LABELS((i-1)*samples+1:floor(3*(i*samples-(i-1)*samples)/4)+(i-1)*samples+1,:);
%   X_test{i}=im_rs(:,ceil(3*(i*samples-(i-1)*samples)/4)+(i-1)*samples+1:i*samples);
%   Y_test{i}=LABELS(ceil(3*(i*samples-(i-1)*samples)/4)+(i-1)*samples+1:i*samples,:);
% 
% end
% 
% XX_train=[];XX_test=[];YY_train=[];YY_test=[];
% for i=1:nC
%     XX_train=[XX_train X_train{i}];
%     YY_train=[YY_train;Y_train{i}];
%     XX_test=[XX_test X_test{i}];
%     YY_test=[YY_test;Y_test{i}];
% 
% end
% 
% idx=randperm(size(XX_train,2));
% idx2=randperm(size(XX_test,2));
% %x=x(randperm(length(x)));
% IMAGE_shuffle_train=XX_train(:,idx);
% IMAGE_shuffle_test=XX_test(:,idx2);
% LABELS_shuffle_train=YY_train(idx,:);
% LABELS_shuffle_test=YY_test(idx2,:);














