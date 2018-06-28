clc
clear all
close all
load coastlines
year1 = 1979;
year2 = 2017;
day1  = 1;
day2  = 122;
y1    = 35;
y2    = 25;
x1    = -105;
x2    = -85;
p1    = 850.0;
p2    = 250.0;
Swind = 2;
demean = 0;
manual = 0;
ensembles=20;
cnt_ens=1; 
load ('/scratch/03959/achattop/clustering/grid.mat');

lat_north_index=96;
lat_south_index=31;
lon_west_index=157;
lon_east_index=253;
lat1=lat(97:end);
lat11=lat1(31:65);
nC=4;
samples=1400;
LABELS=zeros(nC*samples,nC);
[qx,qy]=meshgrid(lon(lon_west_index:lon_east_index),lat1(lat_south_index:end));


tic;

for m=1:ensembles
    load (['/scratch/03959/achattop/clustering/Z99daily_NA_M' num2str(cnt_ens) '.mat']);
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
nC=4;
nEOF=22;
Xr = squeeze(EOFs(:,end-nEOF+1:end))'*X;
Xtr=Xr';

Y=tsne(Xtr,'Algorithm','exact','Distance','Cosine');
h=figure(2)
gscatter(Y(:,1),Y(:,2));hold on

[idx,C] = kmeans(Y,nC,'replicates',1000);

figure(2)
gscatter(C(:,1),C(:,2))
savefig(h,'tsneviz.fig')

for j=1:nC
    count=1;
 for i=1:length(idx)
     if (idx(i)==j)
  cluster_tsne{j}(:,count)=X(:,i);
  cluster_fullZ{j}(:,count)=Psi(:,i);
     count=count+1;
     end
 end
end

for i=1:nC
    Z500centers(:,i)=mean(cluster_tsne{i},2);
    Z500_full(:,i)=mean(cluster_fullZ{i},2);
end
h=figure(3)
for i=1:nC
    Z=reshape(Z500centers(:,i),97,66);
subplot(ceil(nC/2),2,i)
contourf(qx',qy',Z,10);hold on;
caxis([-120 120])

plot(coastlon+360,coastlat,'Linewidth',1,'Color','k');

axis equal
    xlim([195 315])
    ylim([25 97])
end

savefig(h,'anomaly.fig')

h=figure(1)
for i=1:nC
    Z=reshape(Z500_full(:,i),97,66);
subplot(ceil(nC/2),2,i)
contourf(qx',qy',Z,10);hold on;
caxis([-120 120])

plot(coastlon+360,coastlat,'Linewidth',1,'Color','k');

axis equal
    xlim([195 315])
    ylim([25 97])
end

savefig(h,'fullZ.fig')

for j=1:nC
    for i=1:size(cluster_fullZ{j},2)
        Z=reshape(cluster_fullZ{j}(:,i),97,66);
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

save('savedata_for_training_30ensemble_4classes_fullZ_tsne.mat','IMAGE_shuffle_train','LABELS_shuffle_train','IMAGE_shuffle_test','LABELS_shuffle_test','-v7.3');
csvwrite('training_40ensemble_4classes_fullZ_tsne.csv',IMAGE_shuffle_train);
csvwrite('labels_40ensemble_4classes_fullZ_tsne.csv',LABELS_shuffle_train);

csvwrite('test_40ensemble_4classes_fullZ_tsne.csv',IMAGE_shuffle_test);
csvwrite('test_labels_40ensemble_4classes_fullZ_tsne.csv',LABELS_shuffle_test);



