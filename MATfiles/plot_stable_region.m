[f1 f2]=meshgrid(-1.25:.05:1.25, -1.5:.05:1);
mu1=0.5;
mu2=0.4;

% resample among those that are stable
stable=abs((f1.^2 + 4*f2).^(1/2)/2 - f1/2)<1 & abs(f1/2 + (f1.^2 + 4*f2).^(1/2)/2)<1;

% plot results
plot(f1(stable),f2(stable),'x')
hold on
plot(f1(~stable),f2(~stable),'o')
plot(mu1,mu2,'*','LineWidth',2)
hold off
xlabel('f1')
ylabel('f2')
legend('stable','unstable','centroid')
axis([-1.25 1.25 -1.5 1])
axis square
