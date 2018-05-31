function out = MYdistsbd(center, data)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
out = zeros(size(center, 1), size(data, 1));
    for k = 1:size(center, 1),
        for i=1:size(data,1)
	%out(k, :) = sqrt(sum(((data-ones(size(data, 1), 1)*center(k, :)).^2)'));
    out(k,i)=SBD(center(k,:), data(i,:));
        end
    end
end

