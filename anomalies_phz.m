clear all
clc

lat  = ncread('/work/05078/enabizad/stampede2/Tsurface/1.nc','lat');
lon  = ncread('/work/05078/enabizad/stampede2/Tsurface/1.nc','lon');
date  = ncread('/work/05078/enabizad/stampede2/Tsurface/1.nc','date');

load('Average_phz','Tave','Zave')
day=124;
member=40;
year=86;
NH=97;
lat=lat(NH:end);
Tsummer=zeros(year,length(lon),length(lat),day);
Zsummer=Tsummer;
%for m=[11 12 18]
for m=17
%for m=[26 27 28 29]
%for m=[30 32 34]
    m
    Ta=zeros(year,length(lon),length(lat),day);
    Za=Ta;
    filename = sprintf('%d.nc',m);
    cd /work/05078/enabizad/stampede2/Tsurface/
    T1  = ncread(filename,'TREFHT');
    cd /work/05078/enabizad/stampede2/Z500/
    Z1  = ncread(filename,'Z500');
    for y=1:year
        Tsummer(y,:,:,:)=squeeze(T1(:,NH:end,135+365*(y-1):258+365*(y-1)));
        Zsummer(y,:,:,:)=squeeze(Z1(:,NH:end,135+365*(y-1):258+365*(y-1)));
        for d=1:day
            Ta(y,:,:,d)=squeeze(Tsummer(y,:,:,d))-squeeze(Tave(y,:,:,d));
            Za(y,:,:,d)=squeeze(Zsummer(y,:,:,d))-squeeze(Zave(y,:,:,d));
        end
    end
    filename = sprintf('%s%d.mat','Anomalies',m);
    cd /work/04435/pedramhx/stampede2/LESNanalysis/NewAnalysis/Feb18Results
    save(filename,'Ta','Za','Tsummer','Zsummer','m','-v7.3')
end
