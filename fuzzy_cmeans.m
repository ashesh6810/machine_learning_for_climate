clc
clear all
close all

samples=4000;
ensembles=40;
cnt_ens=1; 
load ('/scratch/03959/achattop/clustering/grid.mat');
lat_north_index=96;
lat_south_index=31;
lon_west_index=157;
lon_east_index=253;
lat1=lat(97:end);
lat11=lat1(31:65);

[qx,qy]=meshgrid(lon(lon_west_index:lon_east_index),lat1(lat_south_index:end));
nC=4;
LABELS=zeros(nC*samples,nC);

tic;

for m=1:ensembles
    load (['/work/03959/achattop/stampede2/tensorflow/Z99daily_NA_M' num2str(cnt_ens) '.mat'])
    Zave=squeeze(mean(Z99NApattern(:,:,:,18:109),2));
    %M{m}=Z99NApattern(:,:,:,18:109);
    for i=1:97
       anomalies(:,i,:,:)=squeeze(Z99NApattern(:,i,:,18:109))-Zave;
    end
    M{m}=anomalies;
    R{m}=Z99NApattern(:,:,:,18:109);
    cnt_ens=cnt_ens+1;
end
count=1;
for m=1:ensembles
 for i=61:86
    for k=1:92
       X(:,count) =reshape(M{m}(i,:,:,k),97*66,1);
       Psi(:,count)=reshape(R{m}(i,:,:,k),97*66,1);
count=count+1;
    end
 end
end
[EOFs,PCval]=EOFanalysis(X);

nEOF=22;
Xr = squeeze(EOFs(:,end-nEOF+1:end))'*X;
Xtr=Xr';
sum(PCval(end-nEOF+1:end))*100.0/sum(PCval)
[centers, idx] = fcm(Xtr,nC,[1.09,1000]);
C=squeeze(EOFs(:,end-nEOF+1:end))*centers';

h=figure(1)

for i=1:nC
    subplot(ceil(nC/2),2,i)
    Z=reshape(C(:,i),97,66);
    (contourf(qx',qy',Z,10));hold on
    axis equal
    caxis([-120 120])
    
end

savefig(h,'anomalies_fuzzy_center.fig')
close(h);

disp('anomaly centers saved ')
maxU=max(idx);
for i=1:nC
    index{i}=find(idx(i,:)==maxU);
end

for j=1:nC
    cluster{j}=Psi(:,index{j});
end

for j=1:nC
    Z500_centers(:,j)=mean(squeeze(cluster{j}),2);
end
   
h=figure(2)

for i=1:nC
    subplot(ceil(nC/2),2,i)
    Z=reshape(Z500_centers(:,i),97,66);
    (contourf(qx',qy',Z,10));hold on
    axis equal
    caxis([5400 5900])
    
end

savefig(h,'fullz500_center.fig');
close(h);

disp('fullZ centers saved')

count=1;
for j=1:nC
    for i=1:size(cluster{j},2)
        Z=reshape(cluster{j}(:,i),97,66);
        h=figure(1);
        contourf(qx',qy',Z,10,'LineColor','k','LineWidth',2)
        im=frame2im(getframe(gca));
        close(h);
        im_rs(:,count)=reshape(imresize(double(im),[28,28]),28*28*3,1)/255;

        LABELS(count,j)=1;
        count=count+1;
        if(i>=samples)
            break;
        end


    end
end

for i=1:nC

  X_train{i}=im_rs(:,(i-1)*samples+1:floor(3*(i*samples-(i-1)*samples)/4)+(i-1)*samples+1);
  Y_train{i}=LABELS((i-1)*samples+1:floor(3*(i*samples-(i-1)*samples)/4)+(i-1)*samples+1,:);
  X_test{i}=im_rs(:,ceil(3*(i*samples-(i-1)*samples)/4)+(i-1)*samples+1:i*samples);
  Y_test{i}=LABELS(ceil(3*(i*samples-(i-1)*samples)/4)+(i-1)*samples+1:i*samples,:);

end

XX_train=[];XX_test=[];YY_train=[];YY_test=[];
for i=1:nC
    XX_train=[XX_train X_train{i}];
    YY_train=[YY_train;Y_train{i}];
    XX_test=[XX_test X_test{i}];
    YY_test=[YY_test;Y_test{i}];

end

idx1=randperm(size(XX_train,2));
idx2=randperm(size(XX_test,2));
IMAGE_shuffle_train=XX_train(:,idx1);
IMAGE_shuffle_test=XX_test(:,idx2);
LABELS_shuffle_train=YY_train(idx1,:);
LABELS_shuffle_test=YY_test(idx2,:);

save('savedata_for_training_30ensemble_4classes_fullZ.mat','IMAGE_shuffle_train','LABELS_shuffle_train','IMAGE_shuffle_test','LABELS_shuffle_test','-v7.3');
csvwrite('training_40ensemble_4classes_fullZ.csv',IMAGE_shuffle_train);
csvwrite('labels_40ensemble_4classes_fullZ.csv',LABELS_shuffle_train);

csvwrite('test_40ensemble_4classes_fullZ.csv',IMAGE_shuffle_test);
csvwrite('test_labels_40ensemble_4classes_fullZ.csv',LABELS_shuffle_test);
