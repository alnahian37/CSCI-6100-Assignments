clc;
clear all;
close all;

n = 10000;
CovX = [1 0.5; 0.5 1];
mu = [1;0];
S = mvnrnd(mu,CovX,n);
S=S';
size(S)
a=S(1,:);
b=S(2,:);

[Phi,Lam] = eig(CovX);
Lam
Phi=1/0.7071*Phi;
Phi
%Phi=[1 1;-1 1]
rootlam=sqrt(Lam)
rootlaminv=Lam^-0.5
%rootlaminv=inv(rootlam)
%Y = Phi*S;
%po=size(Y)

Aw=rootlaminv*Phi

%Aw=(Phi*rootlaminv)'

%Aw=rootlaminv*Y;

%W = rootlaminv*Y;
W=Aw*S;
%W=*S;

%c=Y(1,:);
%d=Y(2,:);
c=W(1,:);
d=W(2,:);


figure(1)
scatter(a,b,'r')
xlabel('x1')
ylabel('x2')
legend('Gaussian Realization')
title('Plot using 10000 data points')


figure(2)
scatter(a,b,'r')
hold on

scatter(c,d,'b.')
legend('Gaussian Realization', 'Whitened data')
xlabel('x1')
ylabel('x2')
title('Plot using 10000 data points')


hud=Phi*rootlaminv
hud2=rootlaminv*Phi


