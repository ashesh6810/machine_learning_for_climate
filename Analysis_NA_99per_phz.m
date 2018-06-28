clear all
close all
clc

%exreme in member:
member=40;

%box for the extremes:
lat_north_index=64;  
lat_south_index=32;  
lon_west_index=185;  
lon_east_index=241;  

%box for the circulation patterns
lat_north_index2=96; 
lat_south_index2=31;
lon_west_index2=157;  
lon_east_index2=253;  

total_year = 86;
summer_day_s = 18;
summer_day_e = 124-15;

NH=97;
load grid
load('LandData.mat','per99')
load('landNew.mat','I','J')

T99=false(size(I,1),summer_day_e+5,total_year,member);
count1=zeros(40,1);
for m=1:member
    Za99NApattern=zeros(total_year,lon_east_index2-lon_west_index2+1,lat_north_index2-lat_south_index2+1,summer_day_e,'single');
    Z99NApattern=zeros(total_year,lon_east_index2-lon_west_index2+1,lat_north_index2-lat_south_index2+1,summer_day_e,'single');    
    Ta99NApattern=zeros(total_year,lon_east_index2-lon_west_index2+1,lat_north_index2-lat_south_index2+1,summer_day_e,'single');
    disp([m])
    load(['/work/04435/pedramhx/stampede2/LESNanalysis/NewAnalysis/Feb18Results/Anomalies' num2str(m) '.mat'],'Ta','Za','Zsummer')
    for year=1:total_year
        for day=summer_day_s:summer_day_e+5
            Za99NApattern(year, :, :, day)=squeeze(Za(year,lon_west_index2:lon_east_index2,lat_south_index2:lat_north_index2,day));
            Z99NApattern(year, :, :, day)=squeeze(Zsummer(year,lon_west_index2:lon_east_index2,lat_south_index2:lat_north_index2,day));            
            Ta99NApattern(year, :, :, day)=squeeze(Ta(year,lon_west_index2:lon_east_index2,lat_south_index2:lat_north_index2,day));            
            for i=1:size(I)
                if(lon(I(i))>=lon(lon_west_index) && lon(I(i))<=lon(lon_east_index) && lat(J(i)+NH+31)>=lat(NH+lat_south_index) && lat(J(i)+NH+31)<=lat(NH+lat_north_index))
                    if(squeeze(Ta(year,I(i),J(i)+31,day))>=squeeze(per99(i,year)))
                        T99(i,day,year,m) = true;
                        count1(m)=count1(m)+1;                        
                    end
                end
            end
        end
    end
    count1(m)
    save(['Z99daily_NA_M' num2str(m) '.mat'],'Za99NApattern','Z99NApattern','count1','-v7.3')
    save(['T99daily_NA_M' num2str(m) '.mat'],'Ta99NApattern','count1','-v7.3')    
end

Heatwave5day99v1=false(size(I,1),summer_day_e,total_year,member);
Heatwave5day99v2=false(size(I,1),summer_day_e,total_year,member);
Heatwave5day99=false(summer_day_e,total_year,member);
count2=zeros(40,1);
for m=1:member
    disp([m])
    for year=1:total_year
        for day=summer_day_s:summer_day_e
            for i=1:size(I)
                if(T99(i,day,year,m) && T99(i,day+1,year,m) && T99(i,day+2,year,m) && T99(i,day+3,year,m) && T99(i,day+4,year,m))
                    Heatwave5day99v1(i,day:day+4,year,m) = true;
                    Heatwave5day99v2(i,day,year,m) = true;                    
                    Heatwave5day99(day,year,m) = true;  
                    count2(m)=count2(m)+1;
                end
            end
        end
    end
    count2(m)    
end
save('Heatwave99.mat','Heatwave5day99','count2','-v7.3')
save('Heatwave99v1.mat','Heatwave5day99v1','count2','-v7.3')
save('Heatwave99v2.mat','Heatwave5day99v2','count2','-v7.3')
    
