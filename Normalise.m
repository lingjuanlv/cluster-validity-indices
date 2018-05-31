% function normalD=Normalise(D)
% normalD=zeros(size(D));
% maxMat=max(D,[],2);  %2nd dim,row:attribute,have to transpose data first
% minMat=min(D,[],2);
% for i=1:size(D,1)
%     range=maxMat(i)-minMat(i); 
%     if range==0
%         range=0.5;
%     end
%    temp=bsxfun(@minus,D(i,:),minMat(i));
%     r=temp./range;
%     normalD(i,:)=normalD(i,:)+r;
% end
% end

%%simplified version
function normalD=Normalise(D)
maxVec = max(D,[],1);     %1st dim,column:attribute
minVec = min(D,[],1);
normalD = bsxfun (@minus, D, minVec);
normalD = bsxfun (@rdivide, normalD, maxVec-minVec);
end
