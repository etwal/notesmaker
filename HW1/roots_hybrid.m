% roots_hybrid: Driver script for Problem 1.  Locates all roots of the
% degree-10 polynomial fp(x) on [0, 2] with the hybrid bisection/Newton
% routine, compares them against MATLAB's roots(), and generates the two
% figures used in the writeup.

   tol1 = 1.0e-5;      % Relative tolerance for the bisection phase.
   tol2 = 1.0e-10;     % Relative tolerance for the Newton phase.
   nsamp = 10001;      % Sample count for the brute-force bracket search.

   c = [512, -5120, 21760, -51200, 72800, -64064, 34320, -10560, 1650, -100, 1];

   [xmins, xmaxs] = brackets(@fp, 0.0, 2.0, nsamp);
   nroots = length(xmins);

   fprintf('Found %d brackets on [0, 2]\n\n', nroots);
   fprintf('    %-20s  %-12s\n', 'Root', '|f(root)|');

   xr = zeros(nroots, 1);
   for i = 1 : nroots
      xr(i) = hybrid(@fp, @dfpdx, xmins(i), xmaxs(i), tol1, tol2);
      fprintf('    %-20.15f  %-12.3e\n', xr(i), abs(fp(xr(i))));
   end

   % Independent check against MATLAB's built-in polynomial root finder.
   xm = sort(roots(c));
   xm = xm(abs(imag(xm)) < 1.0e-8 & real(xm) >= 0.0 & real(xm) <= 2.0);

   fprintf('\nMaximum deviation from roots(): %.3e\n', max(abs(xr - real(xm))));

%-----------------------------------------------------------------------
% Figure 1: the polynomial and its roots.
%-----------------------------------------------------------------------
   xs = linspace(0.0, 2.0, 2001);

   figure('Position', [100, 100, 760, 420]);
   plot(xs, fp(xs), 'b-', 'LineWidth', 1.2);  hold on;
   plot([0, 2], [0, 0], 'k:', 'LineWidth', 0.8);
   plot(xr, zeros(nroots, 1), 'ro', 'MarkerSize', 7, 'LineWidth', 1.4);
   hold off;
   xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 13);
   ylabel('$f(x)$', 'Interpreter', 'latex', 'FontSize', 13);
   legend({'$f(x)$', '$f = 0$', 'computed roots'}, ...
          'Interpreter', 'latex', 'Location', 'northeast', 'FontSize', 11);
   axis([0, 2, -1.4, 1.4]);  grid on;
   exportgraphics(gcf, 'fig1_polynomial.pdf', 'ContentType', 'vector');

%-----------------------------------------------------------------------
% Figure 2: convergence of each phase, tracked for a single root.
%
% The bisection and Newton iterations are reproduced here (rather than
% instrumenting hybrid.m, whose interface is fixed by the assignment) so
% that the error at every iteration can be recorded.
%-----------------------------------------------------------------------
   xstar = xr(5);                    % Root near x = 0.8436.
   a = xstar - 0.12;                 % A deliberately wide initial bracket.
   b = xstar + 0.10;

   nb = 40;
   ebis = zeros(nb, 1);
   fa = fp(a);
   for k = 1 : nb
      xb = 0.5 * (a + b);
      fx = fp(xb);
      ebis(k) = abs(xb - xstar) / abs(xstar);
      if fa * fx < 0
         b = xb;
      else
         a = xb;  fa = fx;
      end
   end

   nn = 7;
   enew = zeros(nn, 1);
   xn = xstar + 0.10;                % Newton started from a comparable offset.
   for k = 1 : nn
      enew(k) = abs(xn - xstar) / abs(xstar);
      xn = xn - fp(xn) / dfpdx(xn);
   end

   figure('Position', [100, 100, 760, 420]);
   semilogy(1:nb, max(ebis, 1.0e-17), 'b-o', 'LineWidth', 1.2, 'MarkerSize', 4); hold on;
   semilogy(1:nn, max(enew, 1.0e-17), 'r-s', 'LineWidth', 1.2, 'MarkerSize', 6);
   semilogy([1, nb], [tol1, tol1], 'k--', 'LineWidth', 0.9);
   semilogy([1, nb], [tol2, tol2], 'k-.', 'LineWidth', 0.9);
   hold off;
   xlabel('Iteration number $n$', 'Interpreter', 'latex', 'FontSize', 13);
   ylabel('$|x_n - x^\star| / |x^\star|$', 'Interpreter', 'latex', 'FontSize', 13);
   legend({'bisection', 'Newton', '\texttt{tol1}', '\texttt{tol2}'}, ...
          'Interpreter', 'latex', 'Location', 'northeast', 'FontSize', 11);
   axis([1, nb, 1.0e-17, 1.0]);  grid on;
   exportgraphics(gcf, 'fig2_convergence.pdf', 'ContentType', 'vector');

   fprintf('Wrote fig1_polynomial.pdf and fig2_convergence.pdf\n');

%-----------------------------------------------------------------------
% The test polynomial,
%
%   f(x) = 512 x^10 - 5120 x^9 + 21760 x^8 - 51200 x^7 + 72800 x^6
%          - 64064 x^5 + 34320 x^4 - 10560 x^3 + 1650 x^2 - 100 x + 1,
%
% evaluated with polyval (Horner's rule).
%-----------------------------------------------------------------------
function fx = fp(x)
   c = [512, -5120, 21760, -51200, 72800, -64064, 34320, -10560, 1650, -100, 1];
   fx = polyval(c, x);
end

%-----------------------------------------------------------------------
% Analytic derivative of fp,
%
%   f'(x) = 5120 x^9 - 46080 x^8 + 174080 x^7 - 358400 x^6 + 436800 x^5
%           - 320320 x^4 + 137280 x^3 - 31680 x^2 + 3300 x - 100.
%-----------------------------------------------------------------------
function dfx = dfpdx(x)
   c = [5120, -46080, 174080, -358400, 436800, -320320, 137280, -31680, 3300, -100];
   dfx = polyval(c, x);
end

%-----------------------------------------------------------------------
% Brute-force search for sub-intervals of [xmin, xmax] which bracket a root
% of f.  The interval is sampled at n equally spaced points and every
% adjacent pair straddling a sign change of f is returned as a bracket.
%-----------------------------------------------------------------------
function [xmins, xmaxs] = brackets(f, xmin, xmax, n)
   xs = linspace(xmin, xmax, n);
   fs = arrayfun(f, xs);

   % Indices i where f changes sign between xs(i) and xs(i+1).
   idx = find(fs(1:end-1) .* fs(2:end) < 0);

   xmins = xs(idx)';
   xmaxs = xs(idx + 1)';
end
