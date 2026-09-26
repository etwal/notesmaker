% Problem 1: all roots of fp on [0, 2], plus Figures 1 and 2.
tol1 = 1.0e-5;
tol2 = 1.0e-12;

% Brute-force brackets: sign changes between adjacent sample points.
xs = linspace(0, 2, 10001);
fs = fp(xs);
idx = find(fs(1:end-1) .* fs(2:end) < 0);

xr = zeros(length(idx), 1);
for i = 1 : length(idx)
   xr(i) = hybrid(@fp, @dfpdx, xs(idx(i)), xs(idx(i) + 1), tol1, tol2);
   fprintf('%.15f  %.3e\n', xr(i), abs(fp(xr(i))));
end

% Figure 1: polynomial and roots.
xp = linspace(0, 2, 2001);
figure('Position', [100 100 760 420]);
plot(xp, fp(xp), 'b-', [0 2], [0 0], 'k:', xr, 0 * xr, 'ro');
xlabel('x');  ylabel('f(x)');
legend('f(x)', 'f = 0', 'computed roots', 'Orientation', 'horizontal', 'Location', 'north');
axis([0 2 -1.4 1.4]);
exportgraphics(gcf, 'fig1_polynomial.pdf', 'ContentType', 'vector');

% Figure 2: bisection vs Newton error for the root near 0.8436.
xstar = xr(5);
a = xstar - 0.12;  b = xstar + 0.10;
ebis = zeros(40, 1);
for k = 1 : 40
   xb = 0.5 * (a + b);
   ebis(k) = abs(xb - xstar) / xstar;
   if fp(a) * fp(xb) < 0, b = xb; else, a = xb; end
end
enew = zeros(7, 1);
xn = xstar + 0.10;
for k = 1 : 7
   enew(k) = abs(xn - xstar) / xstar;
   xn = xn - fp(xn) / dfpdx(xn);
end
figure('Position', [100 100 760 420]);
semilogy(1:40, max(ebis, 1e-17), 'b-o', 1:7, max(enew, 1e-17), 'r-s', ...
         [1 40], [tol1 tol1], 'k--', [1 40], [tol2 tol2], 'k-.');
xlabel('iteration n');  ylabel('relative error');
legend('bisection', 'Newton', 'tol1', 'tol2');
axis([1 40 1e-17 1]);
exportgraphics(gcf, 'fig2_convergence.pdf', 'ContentType', 'vector');

function fx = fp(x)
   fx = polyval([512 -5120 21760 -51200 72800 -64064 34320 -10560 1650 -100 1], x);
end

function dfx = dfpdx(x)
   dfx = polyval([5120 -46080 174080 -358400 436800 -320320 137280 -31680 3300 -100], x);
end
