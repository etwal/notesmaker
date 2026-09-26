function x = hybrid(f, dfdx, xmin, xmax, tol1, tol2)
% f, dfdx: function and derivative; [xmin, xmax]: bracket;
% tol1, tol2: relative tolerances for bisection, Newton.  x: root.
   a = xmin;  b = xmax;  fa = f(a);
   for iter = 1 : 200
      x = 0.5 * (a + b);
      if b - a <= tol1 * max(abs(x), realmin), break; end
      fx = f(x);
      if fa * fx < 0
         b = x;
      else
         a = x;  fa = fx;
      end
   end
   for iter = 1 : 100
      dx = -f(x) / dfdx(x);
      x = x + dx;
      if abs(dx) <= tol2 * max(abs(x), realmin), return; end
   end
end
