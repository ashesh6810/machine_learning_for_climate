% function [Sval] = kmeanscluster_PHZ(year1,year2,day1,day2,y1,y2,x1,x2,p1,p2,Swind,demean,manual)
clc
clear all
close all
load ('/scratch/03959/achattop/clustering/grid.mat');
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
manual = 1;
ensembles=10;
cnt_ens=1; 
lat_north_index=96;
lat_south_index=31;
lon_west_index=157;
lon_east_index=253;
lat1=lat(97:end);
[qx,qy]=meshgrid(lon(lon_west_index:lon_east_index),lat1(lat_south_index:end));

%{
Vincent Gonzales, Summer 2018

Parameters for inputs (NCEP2):
number of clusters, nC
year 1, year1       between 1979 and 2017
year 2, year2
Day 1, day1     [30=June30,61=July31,92=Aug31,122=sep30,153=oct31]
day 2, day2 .
latup, y1         Choose from [90:-2.5:0]
latdown, y2
lonleft, x1         Choose from [0:2.5:357.5]-360
longright, x2
levellow, p1 (usually choose 850) Choose from [850;700;600;500;400;300;250]
levelhi, p2  (usually choose 250)
Swind, Swind        Choose 1: averaged; 2: following CYL (steering)
demean, to take out monthly mean or not (1-yes,0-no)
cnt_ens denotes which ensmeble to start from
ensemble denotes the number of ensembles

can include PercentVarThresh if pca ever wants to be done in future but
unnecssary
%}

tic;

for m=1:ensembles
    load (['/work/03959/achattop/stampede2/tensorflow/Z99daily_NA_M' num2str(cnt_ens) '.mat'])
    Zave=squeeze(mean(Z99NApattern(:,:,:,18:109),2));
    %M{m}=Z99NApattern(:,:,:,18:109);
    for i=1:97
       anomalies(:,i,:,:)=squeeze(Z99NApattern(:,i,:,18:109))-Zave; 
    end
    M{m}=anomalies;
    cnt_ens=cnt_ens+1;

end
count=1;
for m=1:ensembles
for i=61:86
    for k=1:92
       X(:,count) =reshape(M{m}(i,:,:,k),97*66,1);
count=count+1;
    end
end
end
%U2=reshape(U,size(U,1)*size(U,2),size(U,3)*size(U,4));
%V2=reshape(V,size(V,1)*size(V,2),size(V,3)*size(V,4));

%X=[U2;V2];

%Do PCA
[EOFs,PCval]=EOFanalysis(X);


% figure
% plot(silval(:,1), silval(:,2),'r*-.')   %plots avg silhouette value plot

%if we want to manually select clusters from plot
if manual == 1
   % reply = input('How many clusters do you want?')   %enter number into command window and then hit 'Enter'
    %reply = input('How many clusters do you want?')   %enter number into command window and then hit 'Enter'
    %nC = reply;
   nC=10; 
   
    
    %reply = input('How many EOFs do you want?')   %enter number into command window and then hit 'Enter'
    %reply = input('How many EOFs do you want?')   %enter number into command window and then hit 'Enter'
    %nEOF = reply;
    nEOF = 22;
    
%if we want to just select highest avg silhouette value between 4-20 clusters
elseif manual == 0
    EOFmax=30;
    Sval = zeros(4,EOFmax);
    for nEOF=5:EOFmax
        nEOF
        Xr = squeeze(EOFs(:,end-nEOF+1:end))'*X;
        Xtr=Xr';
        
        %calculate mean silhouette values for certain numbers of clusters
        silval = [];
        for r = 2:4
            [idx,C] = kmeans(Xtr,r,'replicates',100);
            S = silhouette(Xtr,idx);        %calculates silhouette values
            silval = [silval; r mean(S)];   %puts average silhouette value in table
            Sval(r,nEOF) = mean(S);
        end
    end
    h=figure(1)
    pcolor(1:4,1:EOFmax,Sval');colorbar
    im=frame2im(getframe(gca));
    imwrite(im,['silhouttevalues' num2str(ensembles) '.png'])
    disp('Chosen based on the silhouette values')
    [nC,nEOF] = find(max(max(Sval)) == Sval)
    close(h);
else
    disp('manual must be 0 or 1, no or yes')
end

Xr = squeeze(EOFs(:,end-nEOF+1:end))'*X;
Xtr=Xr';
sum(PCval(end-nEOF+1:end))*100.0/sum(PCval)

%kmeans replicated 1000 times (like Souri)
[idx, Cr] = kmeans(Xtr,nC,'replicates',1000);

Count(nC,1)=0;
for n=1:nC
    for d=1:length(idx)
        if(idx(d)==n)
            Count(n)=Count(n)+1;
        end
    end
end
[sum(Count) length(idx)]
h=figure(1)
silhouette(Xtr,idx);
saveas(h,['silhouttevalues' num2str(ensembles) '.png'])
close(h);
C = squeeze(EOFs(:,end-nEOF+1:end))*Cr';
C = C';
%plot code is generalized to stepping in data for lat/long
h=figure(2)
load coastlines
for n=1:nC
    subplot(ceil(nC/3),3,n)                 %makes subplots big enough
    Z=(reshape(C(n,1:size(C,2)),97,66));
    contourf(qx',qy',Z,10);hold on
    plot(coastlon+360,coastlat,'Linewidth',3,'Color','r');
    xlim([195 315])
    ylim([25 90])
    
end

saveas(h,['clusterswithC=' num2str(nC) 'ensemblesize' num2str(ensembles) '.png'])
close(h)
hold off

h=figure(3)
for i=1:6
    k=i-1;
    subplot(3,2,i)
    contourf(qx',qy',reshape(EOFs(:,end-k),97,66),10,'LineColor','k','LineWidth',2);hold on;
    plot(coastlon+360,coastlat,'Linewidth',1,'Color','k');
    plot(coastlon+360,coastlat,'Linewidth',3,'Color','r');
    xlim([195 315])
    ylim([25 90])
    
end

saveas(h,['EOFS' num2str(nC)  '.png'])
close(h)

toc
save(['Clustereddatawithensemble' num2str(ensembles) 'withcluster' num2str(nC) '.mat'],'X','idx','Count','-v7.3')

% end
