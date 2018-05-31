function [ARI,RI,MI,NMIsqrt,VI,NVI,JVI,mContingency] = fuzzyComparisonCVI(dataset,mMembership1, mMembership2)
respath=strcat('out\'); 
outpath=strcat(respath,sprintf('%s.mat',dataset));

%  mMembership1,2 : n x k matries
%  Fuzzy cluster comparison measure.


    objNum = size(mMembership1, 1);

    mContingency = mMembership1' * mMembership2; 

    scalingFactor =  objNum/ sum(sum(mContingency)); %in the cases of crisp, fuzzy or probabilistic partitions, 1.
    mContingency = mContingency * scalingFactor; % contigency table
    
  
    vRowSum = sum(mContingency,2);
    vColSum = sum(mContingency,1);
    totalSum = sum(sum(mContingency));
    totalSqSum = sumsqr(mContingency);
    rowSqSum = sumsqr(vRowSum);
    colSqSum = sumsqr(vColSum);
    
    % the number of pairs of objects both in the partition U and partition V.
    a = 0.5 * (totalSqSum - totalSum); 
    % the number of pairs of objects in neither U and V.
    d = 0.5 * (objNum^2 + totalSqSum - rowSqSum - colSqSum);  
    b = 0.5 * (colSqSum - totalSqSum);
    c = 0.5 * (rowSqSum - totalSqSum);

%% RI

    RI = (a+d) / (a+b+c+d);
    
%%Mirkin
%     Mirkin=(b+c)/(a+b+c+d);			%Mirkin 1970	%p(disagreement)
    Mirkin=2*(b + c);	
%Hubert 1977
%     HI=RI-Mirkin;	%Hubert 1977	%p(agree)-p(disagree)
    HI=((a+d)-(b+c))/(a+b+c+d);

%% ARI (Hubert and Arabie Version)
    ARI = (a - (a+c) * (a+b) / (a + b + c + d)) / ((0.5 * ((a+c) + (a+b))) - ((a+c)*(a+b)/(a+b+c+d)));

%%%%%%%% 

    Pxy = mContingency./objNum; %joint distribution of x and y
    Hxy = sum(-dot(Pxy,log2(Pxy+eps)));


    Px = sum(Pxy,2);
    Py = sum(Pxy,1);

    % entropy of Py and Px
    Hx = -dot(Px,log2(Px+eps));
    Hy = -dot(Py,log2(Py+eps));

    % mutual information
    MI = Hx + Hy - Hxy;

    % NMI_max
    NMImax = MI/max(Hx,Hy);
    
    % NMI_max
    NMIsqrt = MI/sqrt(Hx*Hy);

    % variation of information
    VI = Hx + Hy - 2*MI;
    NVI=VI/Hxy; %normalized distance
%     JVI=VI/max(Hx,Hy); % Jaccard distance between X and Y.
    JVI=1-MI/max(Hx,Hy); % Jaccard distance between X and Y.
    save(outpath,'ARI','RI','MI','NMIsqrt','VI','NVI','JVI');
end % end of function
