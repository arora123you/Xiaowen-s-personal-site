function [y] = rela1(x,num)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
y = []
for i = 4:length(x(:,1))
y(i-3,1) = x(i,5);y(i-3,2) = x(i-1,5); y(i-3,3) = x(i-2,5);y(i-3,4) = x(i-3,5);
y(i-3,5) = x(i,6);y(i-3,6) = x(i,3)*x(i,6);
y(i-3,7) = (x(i,4)-x(i-1,4))/(x(i,1)-x(i-1,1)); y(i-3,8) = (x(i-1,4)-x(i-2,4))/(x(i-1,1)-x(i-2,1));
y(i-3,9) = (x(i-2,4)-x(i-3,4))/(x(i-2,1)-x(i-3,1));
y(i-3,10) = (x(i,5)-x(i-1,5))/(x(i,1)-x(i-1,1));y(i-3,11) = (x(i-1,5)-x(i-2,5))/(x(i-1,1)-x(i-2,1));
y(i-3,12) = (x(i-2,5)-x(i-3,5))/(x(i-2,1)-x(i-3,1));
y(i-3,13) = (x(i,3)*x(i,6)-x(i-1,3)*x(i-1,6))/(x(i,1)-x(i-1,1));
y(i-3,14) = (x(i-1,3)*x(i-1,6)-x(i-2,3)*x(i-2,6))/(x(i-1,1)-x(i-2,1));
y(i-3,15) = (x(i-2,3)*x(i-2,6)-x(i-3,3)*x(i-3,6))/(x(i-2,1)-x(i-3,1));
y(i-3,16) = num;
end
end

