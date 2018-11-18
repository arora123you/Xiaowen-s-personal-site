function [y] = rela1(x)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
y = []
for i = 1:length(x(:,1))
y(i) = (x(i,3))*x(i,6);
plot(y)
end

