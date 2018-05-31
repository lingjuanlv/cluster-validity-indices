clc
close all;
clear all;

options = [2;100;0.00001;1];
run=100;
% total_no_of_points=1000;
true_clusters=4;
if true_clusters-floor(true_clusters/2)>=2
    clusters_set=true_clusters-floor(true_clusters/2):true_clusters+floor(true_clusters/2);
else
    clusters_set=2:true_clusters+floor(true_clusters/2);
end
true_loc=find(clusters_set==true_clusters); %true cluster location,begin with 2,so -1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% CLUSIVAT %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% odds_matrix=ceil(true_clusters*rand(1,true_clusters));

% colors1=colormap;
% colors=zeros(true_clusters,3);
% for i=1:true_clusters
%     colors(i,:)=colors1(ceil(length(colors1)*i/true_clusters),:);
% end

% [data_matrix_with_lables,mean_matrix,var_matrix] = data_generate(true_clusters,odds_matrix,total_no_of_points);
%load s1,1000 points
data_matrix_with_lables=load('s1');
data_matrix_with_lables=data_matrix_with_lables.X;
data_matrix_raw=Normalise(data_matrix_with_lables(:,1:2));
%to run 100 times, reduce variance
%initialize
%normalized to unity each record
data_matrix_raw=normr(data_matrix_raw);
% data_matrix_raw=Normalise(data_matrix_raw);
dim=size(data_matrix_raw,2);
% reduced_dim=floor(0.5*dim);
if dim==2
    reduced_dim=dim;
else
    reduced_dim=floor(0.5*dim);
end
%force range [-1,1]
% data_matrix_raw=(data_matrix_raw-min(data_matrix_raw))./(max(data_matrix_raw)-min(data_matrix_raw));
% data_matrix_raw=data_matrix_raw*2-1; %fall into [-1,1]
GT=data_matrix_with_lables(:,3);
GT(find(GT==-1))=2;
class=length(unique(GT)); %m
class1_index=find(GT==1);
class2_index=find(GT==2);
class3_index=find(GT==3);
class4_index=find(GT==4);
for i=1:length(GT) %n:records
    GT_expend(i,1:class)=0;
    GT_expend(i,GT(i))=1;   %transform GT from n*1 to n*m
end
data_n = size(data_matrix_raw, 1);

ER_raw_GT_run=0; %average over run
ER_pertured_GT_run=0;
ER_pertured_raw_run=0;
RMSE_inner=0;
RMSE_L2=0;
acc_success_raw=0;
acc_success_GT=0;
for j=1:length(clusters_set)
    clusters=clusters_set(j);
%     init_U = initfcm(clusters, data_n);			% Initial fuzzy partition fixed,not vary for each run
for k=1:run
init_U = initfcm(clusters, data_n);			% Initial fuzzy partition fixed,not vary for each run
[center_raw, U_raw, OBJ_FCN_raw] = fcm_new(init_U,data_matrix_raw, clusters,options);
% [center_raw, U_raw, OBJ_FCN_raw] = MYfcm(data_matrix_raw, clusters)
[maxU_raw,raw_class] = max(U_raw);
raw_class=raw_class';
%records belong to the same class shuold be clustered into the same cluster
if clusters==true_clusters
    raw_class1_index=find(raw_class==1);
    raw_class2_index=find(raw_class==2);
    raw_class3_index=find(raw_class==3);
    raw_class4_index=find(raw_class==4);
    raw_class1_common=max([length(intersect(class1_index,raw_class1_index)),length(intersect(class1_index,raw_class2_index)),length(intersect(class1_index,raw_class3_index)),length(intersect(class1_index,raw_class4_index))]);
    raw_class2_common=max([length(intersect(class2_index,raw_class1_index)),length(intersect(class2_index,raw_class2_index)),length(intersect(class2_index,raw_class3_index)),length(intersect(class2_index,raw_class4_index))]);
    raw_class3_common=max([length(intersect(class3_index,raw_class1_index)),length(intersect(class3_index,raw_class2_index)),length(intersect(class3_index,raw_class3_index)),length(intersect(class3_index,raw_class4_index))]);
    raw_class4_common=max([length(intersect(class4_index,raw_class1_index)),length(intersect(class4_index,raw_class2_index)),length(intersect(class4_index,raw_class3_index)),length(intersect(class4_index,raw_class4_index))]);
    ER_raw_GT=1-(raw_class1_common+raw_class2_common+raw_class3_common+raw_class4_common)/length(GT);
    ER_raw_GT_run=ER_raw_GT_run+ER_raw_GT;
end

Y=two_Gompertz(data_matrix_raw);
T=normrnd(0,1,[dim,reduced_dim]);
data_matrix_perturbed=Y*T/sqrt(reduced_dim);  %perturbed data
%RMSE of inner product matrix
if clusters==true_clusters
    inner_perturbed=data_matrix_perturbed*data_matrix_perturbed';
    inner_raw=data_matrix_raw*data_matrix_raw';
%     RMSE=sqrt(mean(abs(inner_perturbed(:) - inner_raw(:)).^2))/sqrt(mean(inner_raw(:).^2));
    RMSE=sqrt(sum((inner_perturbed(:) - inner_raw(:)).^2)/numel(inner_raw));
    RMSE_inner=RMSE_inner+RMSE;
    L2_raw=pdist(data_matrix_raw,'euclidean');
    L2_perturbed=pdist(data_matrix_perturbed,'euclidean');
%     RMSE=sqrt(mean(abs(L2_perturbed(:) - L2_raw(:)).^2))/sqrt(mean(L2_raw(:).^2));
    RMSE=sqrt(sum((L2_perturbed(:) - L2_raw(:)).^2)/numel(L2_raw));
    RMSE_L2=RMSE_L2+RMSE;
end
[center, U, OBJ_FCN] = fcm_new(init_U,data_matrix_perturbed, clusters,options)
% [center, U, OBJ_FCN] = MYfcm(data_matrix_perturbed, clusters)
[maxU,pertured_class] = max(U);
pertured_class=pertured_class';
%records belong to the same class shuold be clustered into the same cluster
if clusters==true_clusters
pertured_class1_index=find(pertured_class==1);
pertured_class2_index=find(pertured_class==2);
pertured_class3_index=find(pertured_class==3);
pertured_class4_index=find(pertured_class==4);
pertured_class1_common=max([length(intersect(class1_index,pertured_class1_index)),length(intersect(class1_index,pertured_class2_index)),length(intersect(class1_index,pertured_class3_index)),length(intersect(class1_index,pertured_class4_index))]);
pertured_class2_common=max([length(intersect(class2_index,pertured_class1_index)),length(intersect(class2_index,pertured_class2_index)),length(intersect(class2_index,pertured_class3_index)),length(intersect(class2_index,pertured_class4_index))]);
pertured_class3_common=max([length(intersect(class3_index,pertured_class1_index)),length(intersect(class3_index,pertured_class2_index)),length(intersect(class3_index,pertured_class3_index)),length(intersect(class3_index,pertured_class4_index))]);
pertured_class4_common=max([length(intersect(class4_index,pertured_class1_index)),length(intersect(class4_index,pertured_class2_index)),length(intersect(class4_index,pertured_class3_index)),length(intersect(class4_index,pertured_class4_index))]);
ER_pertured_GT=1-(pertured_class1_common+pertured_class2_common+pertured_class3_common+pertured_class4_common)/length(GT);
ER_pertured_GT_run=ER_pertured_GT_run+ER_pertured_GT;
%pertured VS raw
pertured_raw_class1_common=max([length(intersect(raw_class1_index,pertured_class1_index)),length(intersect(raw_class1_index,pertured_class2_index)),length(intersect(raw_class1_index,pertured_class3_index)),length(intersect(raw_class1_index,pertured_class4_index))]);
pertured_raw_class2_common=max([length(intersect(raw_class2_index,pertured_class1_index)),length(intersect(raw_class2_index,pertured_class2_index)),length(intersect(raw_class2_index,pertured_class3_index)),length(intersect(raw_class2_index,pertured_class4_index))]);
pertured_raw_class3_common=max([length(intersect(raw_class3_index,pertured_class1_index)),length(intersect(raw_class3_index,pertured_class2_index)),length(intersect(raw_class3_index,pertured_class3_index)),length(intersect(raw_class3_index,pertured_class4_index))]);
pertured_raw_class4_common=max([length(intersect(raw_class4_index,pertured_class1_index)),length(intersect(raw_class4_index,pertured_class2_index)),length(intersect(raw_class4_index,pertured_class3_index)),length(intersect(raw_class4_index,pertured_class4_index))]);
ER_pertured_raw=1-(pertured_raw_class1_common+pertured_raw_class2_common+pertured_raw_class3_common+pertured_raw_class4_common)/length(GT);
ER_pertured_raw_run=ER_pertured_raw_run+ER_pertured_raw;

C = confusionmat(raw_class,pertured_class);
acc_success_raw=acc_success_raw+sum(max(C))/sum(sum(C));

C = confusionmat(GT,pertured_class);
acc_success_GT=acc_success_GT+sum(max(C))/sum(sum(C));
end

[ARI,RI,MI,NMIsqrt,VI,NVI,JVI,mContingency] = fuzzyComparisonCVI('s1_raw_2stage', U',U_raw')
ARI_run_raw(k,j)=ARI;
RI_run_raw(k,j)=RI;
MI_run_raw(k,j)=MI;
NMIsqrt_run_raw(k,j)=NMIsqrt;
VI_run_raw(k,j)=VI;
NVI_run_raw(k,j)=NVI;
JVI_run_raw(k,j)=JVI;
[ARI,RI,MI,NMIsqrt,VI,NVI,JVI,mContingency] = fuzzyComparisonCVI('s1_GT_2stage', U',GT_expend)
ARI_run_GT(k,j)=ARI;
RI_run_GT(k,j)=RI;
MI_run_GT(k,j)=MI;
NMIsqrt_run_GT(k,j)=NMIsqrt;
VI_run_GT(k,j)=VI;
NVI_run_GT(k,j)=NVI;
JVI_run_GT(k,j)=JVI;

%CI
U_GT_run{k}=GT_expend;
U_raw_run{k}=U_raw';
U_perturbed_run{k}=U';
end
CI_raw=zeros(1,length(clusters_set));
CI_GT=zeros(1,length(clusters_set));

CI_ARI_sum_raw=0;
CI_RI_sum_raw=0;
CI_MI_sum_raw=0;
CI_NMIsqrt_sum_raw=0;
CI_VI_sum_raw=0;
CI_NVI_sum_raw=0;
CI_JVI_sum_raw=0;

CI_ARI_sum_GT=0;
CI_RI_sum_GT=0;
CI_MI_sum_GT=0;
CI_NMIsqrt_sum_GT=0;
CI_VI_sum_GT=0;
CI_NVI_sum_GT=0;
CI_JVI_sum_GT=0;
for m=1:run
    for n=1:run
        if m<n
            [ARI_raw,RI_raw,MI_raw,NMIsqrt_raw,VI_raw,NVI_raw,JVI_raw]=fuzzyComparisonCVI('s1_GT_2stage',U_raw_run{m},U_perturbed_run{n});
            CI_ARI_sum_raw=CI_ARI_sum_raw+ARI_raw;
            CI_RI_sum_raw=CI_RI_sum_raw+RI_raw;
            CI_MI_sum_raw=CI_MI_sum_raw+MI_raw;
            CI_NMIsqrt_sum_raw=CI_NMIsqrt_sum_raw+NMIsqrt_raw;
            CI_VI_sum_raw=CI_VI_sum_raw+VI_raw;
            CI_NVI_sum_raw=CI_NVI_sum_raw+NVI_raw;
            CI_JVI_sum_raw=CI_JVI_sum_raw+JVI_raw;
                
            [ARI_GT,RI_GT,MI_GT,NMIsqrt_GT,VI_GT,NVI_GT,JVI_GT]=fuzzyComparisonCVI('s1_GT_2stage',U_GT_run{m},U_perturbed_run{n});
            CI_ARI_sum_GT=CI_ARI_sum_GT+ARI_GT;
            CI_RI_sum_GT=CI_RI_sum_GT+RI_GT;
            CI_MI_sum_GT=CI_MI_sum_GT+MI_GT;
            CI_NMIsqrt_sum_GT=CI_NMIsqrt_sum_GT+NMIsqrt_GT;
            CI_VI_sum_GT=CI_VI_sum_GT+VI_GT;
            CI_NVI_sum_GT=CI_NVI_sum_GT+NVI_GT;
            CI_JVI_sum_GT=CI_JVI_sum_GT+JVI_GT;
        end
    end
end
CI_ARI_raw(j)=CI_ARI_sum_raw/(run*(run-1)/2);
CI_RI_raw(j)=CI_RI_sum_raw/(run*(run-1)/2);
CI_MI_raw(j)=CI_MI_sum_raw/(run*(run-1)/2);
CI_NMIsqrt_raw(j)=CI_NMIsqrt_sum_raw/(run*(run-1)/2);
CI_VI_raw(j)=CI_VI_sum_raw/(run*(run-1)/2);
CI_NVI_raw(j)=CI_NVI_sum_raw/(run*(run-1)/2);
CI_JVI_raw(j)=CI_JVI_sum_raw/(run*(run-1)/2);

CI_ARI_GT(j)=CI_ARI_sum_GT/(run*(run-1)/2);
CI_RI_GT(j)=CI_RI_sum_GT/(run*(run-1)/2);
CI_MI_GT(j)=CI_MI_sum_GT/(run*(run-1)/2);
CI_NMIsqrt_GT(j)=CI_NMIsqrt_sum_GT/(run*(run-1)/2);
CI_VI_GT(j)=CI_VI_sum_GT/(run*(run-1)/2);
CI_NVI_GT(j)=CI_NVI_sum_GT/(run*(run-1)/2);
CI_JVI_GT(j)=CI_JVI_sum_GT/(run*(run-1)/2);
end
CI_raw=[CI_ARI_raw;CI_RI_raw;CI_MI_raw;CI_NMIsqrt_raw;CI_VI_raw;CI_NVI_raw;CI_JVI_raw];
CI_GT=[CI_ARI_GT;CI_RI_GT;CI_MI_GT;CI_NMIsqrt_GT;CI_VI_GT;CI_NVI_GT;CI_JVI_GT];

CI_ARI_max_raw=find(CI_ARI_raw==max(CI_ARI_raw));
CI_RI_max_raw=find(CI_RI_raw==max(CI_RI_raw));
CI_MI_max_raw=find(CI_MI_raw==max(CI_MI_raw));
CI_NMIsqrt_max_raw=find(CI_NMIsqrt_raw==max(CI_NMIsqrt_raw));
CI_VI_max_raw=find(CI_VI_raw==min(CI_VI_raw));
CI_NVI_max_raw=find(CI_NVI_raw==min(CI_NVI_raw));
CI_JVI_max_raw=find(CI_JVI_raw==min(CI_JVI_raw));

CI_ARI_max_GT=find(CI_ARI_GT==max(CI_ARI_GT));
CI_RI_max_GT=find(CI_RI_GT==max(CI_RI_GT));
CI_MI_max_GT=find(CI_MI_GT==max(CI_MI_GT));
CI_NMIsqrt_max_GT=find(CI_NMIsqrt_GT==max(CI_NMIsqrt_GT));
CI_VI_max_GT=find(CI_VI_GT==min(CI_VI_GT));
CI_NVI_max_GT=find(CI_NVI_GT==min(CI_NVI_GT));
CI_JVI_max_GT=find(CI_JVI_GT==min(CI_JVI_GT));

%ER 
ER_raw_GT_mean=ER_raw_GT_run/run;
ER_pertured_GT_mean=ER_pertured_GT_run/run;
ER_pertured_raw_mean=ER_pertured_raw_run/run;
RMSE_inner=RMSE_inner/run;
RMSE_L2=RMSE_L2/run;
acc_success_raw=acc_success_raw/run;
acc_success_GT=acc_success_GT/run;

ARI_max_loc=zeros(run,length(clusters_set));
RI_max_loc=zeros(run,length(clusters_set));
MI_max_loc=zeros(run,length(clusters_set));
NMIsqrt_max_loc=zeros(run,length(clusters_set));
VI_max_loc=zeros(run,length(clusters_set));
NVI_max_loc=zeros(run,length(clusters_set));
JVI_max_loc=zeros(run,length(clusters_set));
%max for each run
for r=1:run
ARI_max=max(ARI_run_raw(r,:));  %max and location for each run
ARI_max_label=find(ARI_run_raw(r,:)==ARI_max);
RI_max=max(RI_run_raw(r,:));
RI_max_label=find(RI_run_raw(r,:)==RI_max);
MI_max=max(MI_run_raw(r,:));
MI_max_label=find(MI_run_raw(r,:)==MI_max);
NMIsqrt_max=max(NMIsqrt_run_raw(r,:));
NMIsqrt_max_label=find(NMIsqrt_run_raw(r,:)==NMIsqrt_max);
VI_max=min(VI_run_raw(r,:));
VI_max_label=find(VI_run_raw(r,:)==VI_max);
NVI_max=min(NVI_run_raw(r,:));
NVI_max_label=find(NVI_run_raw(r,:)==NVI_max);
JVI_max=min(JVI_run_raw(r,:));
JVI_max_label=find(JVI_run_raw(r,:)==JVI_max);

ARI_max_loc(r,ARI_max_label)=1;  %rth run,max location 1,others 0
RI_max_loc(r,RI_max_label)=1;
MI_max_loc(r,MI_max_label)=1;
NMIsqrt_max_loc(r,NMIsqrt_max_label)=1;
VI_max_loc(r,VI_max_label)=1;
NVI_max_loc(r,NVI_max_label)=1;
JVI_max_loc(r,JVI_max_label)=1;
end
ARI_success=sum(ARI_max_loc(:,true_loc))/run; %true_clusters max counts/run
RI_success=sum(RI_max_loc(:,true_loc))/run; 
MI_success=sum(MI_max_loc(:,true_loc))/run; 
NMIsqrt_success=sum(NMIsqrt_max_loc(:,true_loc))/run; 
VI_success=sum(VI_max_loc(:,true_loc))/run; 
NVI_success=sum(NVI_max_loc(:,true_loc))/run; 
JVI_success=sum(JVI_max_loc(:,true_loc))/run; 
save('out/s1_raw_2stage_normal','ARI_success','RI_success','MI_success','NMIsqrt_success','VI_success','NVI_success','JVI_success');

ARI_max_loc=zeros(run,length(clusters_set));
RI_max_loc=zeros(run,length(clusters_set));
MI_max_loc=zeros(run,length(clusters_set));
NMIsqrt_max_loc=zeros(run,length(clusters_set));
VI_max_loc=zeros(run,length(clusters_set));
NVI_max_loc=zeros(run,length(clusters_set));
JVI_max_loc=zeros(run,length(clusters_set));
%max for each run
for r=1:run
ARI_max=max(ARI_run_GT(r,:));  %max and location for each run
ARI_max_label=find(ARI_run_GT(r,:)==ARI_max);
RI_max=max(RI_run_GT(r,:));
RI_max_label=find(RI_run_GT(r,:)==RI_max);
MI_max=max(MI_run_GT(r,:));
MI_max_label=find(MI_run_GT(r,:)==MI_max);
NMIsqrt_max=max(NMIsqrt_run_GT(r,:));
NMIsqrt_max_label=find(NMIsqrt_run_GT(r,:)==NMIsqrt_max);
VI_max=min(VI_run_GT(r,:));
VI_max_label=find(VI_run_GT(r,:)==VI_max);
NVI_max=min(NVI_run_GT(r,:));
NVI_max_label=find(NVI_run_GT(r,:)==NVI_max);
JVI_max=min(JVI_run_GT(r,:));
JVI_max_label=find(JVI_run_GT(r,:)==JVI_max);

ARI_max_loc(r,ARI_max_label)=1;  %rth run,max location 1,others 0
RI_max_loc(r,RI_max_label)=1;
MI_max_loc(r,MI_max_label)=1;
NMIsqrt_max_loc(r,NMIsqrt_max_label)=1;
VI_max_loc(r,VI_max_label)=1;
NVI_max_loc(r,NVI_max_label)=1;
JVI_max_loc(r,JVI_max_label)=1;
end
ARI_success=sum(ARI_max_loc(:,true_loc))/run; %true_clusters max counts/run
RI_success=sum(RI_max_loc(:,true_loc))/run; 
MI_success=sum(MI_max_loc(:,true_loc))/run; 
NMIsqrt_success=sum(NMIsqrt_max_loc(:,true_loc))/run; 
VI_success=sum(VI_max_loc(:,true_loc))/run; 
NVI_success=sum(NVI_max_loc(:,true_loc))/run; 
JVI_success=sum(JVI_max_loc(:,true_loc))/run; 
save('out/s1_GT_2stage_normal','ARI_success','RI_success','MI_success','NMIsqrt_success','VI_success','NVI_success','JVI_success');
save('out/s1_ER_raw_GT_2stage_fuzzy','ER_raw_GT_mean');
save('out/s1_ER_pertured_GT_2stage_fuzzy','ER_pertured_GT_mean');
save('out/s1_ER_pertured_raw_2stage_fuzzy','ER_pertured_raw_mean');
save('out/s1_RMSE_inner_2stage_fuzzy','RMSE_inner');
save('out/s1_RMSE_L2_2stage_fuzzy','RMSE_L2');
save('out/s1_acc_success_raw_2stage_fuzzy','acc_success_raw');
save('out/s1_acc_success_GT_2stage_fuzzy','acc_success_GT');
save('out/s1_CI_raw_2stage_fuzzy','CI_raw');
save('out/s1_CI_GT_2stage_fuzzy','CI_GT');