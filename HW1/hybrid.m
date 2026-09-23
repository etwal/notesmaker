function x = hybrid(f, dfdx, xmin, xmax, tol1, tol2)
% hybrid: Locates a root of f in [xmin, xmax] using bisection followed by
% Newton's method.
%
% Bisection is applied until the bracket has been reduced to a relative
% width of tol1; the resulting estimate then seeds a Newton iteration
% which is continued until the relative update falls below tol2.
%
% Arguments
%
%   f:     Function whose root is sought (takes one argument).
%   dfdx:  Derivative function (takes one argument).
%   xmin:  Initial bracket minimum.
%   xmax:  Initial bracket maximum.
%   tol1:  Relative convergence criterion for bisection.
%   tol2:  Relative convergence criterion for Newton iteration.
%
% Returns
%
%   x:     Estimate of root.

   % Maximum iteration counts: guard against non-convergence.
   itmax1 = 200;
   itmax2 = 100;

   a = xmin;
   b = xmax;
   fa = f(a);
   fb = f(b);

   % The initial interval must bracket a root.
   if fa * fb > 0
      error('hybrid: f(xmin) * f(xmax) > 0, [%g, %g] does not bracket a root', ...
            xmin, xmax);
   end

   % Endpoint may already be an exact root.
   if fa == 0
      x = a;  return;
   end
   if fb == 0
      x = b;  return;
   end

   %-----------------------------------------------------------------------
   % Phase 1: bisection to a relative bracket width of tol1.
   %-----------------------------------------------------------------------
   x = 0.5 * (a + b);
   for iter = 1 : itmax1
      x  = 0.5 * (a + b);
      fx = f(x);

      if fx == 0
         return;
      end

      % Relative convergence test on the bracket width.  The max() guards
      % the degenerate case of a root at (or extremely near) the origin.
      if (b - a) <= tol1 * max(abs(x), realmin)
         break;
      end

      % Retain the half interval which still brackets the root.
      if fa * fx < 0
         b = x;
      else
         a  = x;
         fa = fx;
      end
   end
   if iter == itmax1
      warning('hybrid: bisection did not converge in %d iterations', itmax1);
   end

   %-----------------------------------------------------------------------
   % Phase 2: Newton iteration, seeded with the bisection estimate.
   %-----------------------------------------------------------------------
   for iter = 1 : itmax2
      dfx = dfdx(x);
      if dfx == 0
         warning('hybrid: zero derivative at x = %g, halting Newton', x);
         return;
      end

      dx = -f(x) / dfx;
      x  = x + dx;

      if abs(dx) <= tol2 * max(abs(x), realmin)
         return;
      end
   end
   warning('hybrid: Newton did not converge in %d iterations', itmax2);
end
