% test_newtond: Finds a root of the nonlinear system
%
%    x^2 + y^4 + z^6 = 2
%    cos(x y z^2)    = x + y + z
%    y^2 + z^3       = (x + y - z)^2
%
% near (x, y, z) = (-1.00, 0.75, 1.50).

   x0  = [-1.00; 0.75; 1.50];
   tol = 1.0e-12;

   x = newtond(@fsys, @jsys, x0, tol);

   fprintf('Root:      x = %.12f\n', x(1));
   fprintf('           y = %.12f\n', x(2));
   fprintf('           z = %.12f\n', x(3));
   fprintf('Residual:  |f(x)| = %.3e\n\n', norm(fsys(x)));

   % Iteration history, for the convergence table in the writeup.  The loop
   % is repeated here rather than instrumenting newtond, whose interface is
   % fixed by the assignment.
   fprintf('  n            x               y               z         |dx|/|x|\n');
   xh = x0;
   for n = 1 : 20
      dx = jsys(xh) \ (-fsys(xh));
      xh = xh + dx;
      r  = norm(dx) / norm(xh);
      fprintf(' %2d  %14.12f  %14.12f  %14.12f  %9.3e\n', n, xh(1), xh(2), xh(3), r);
      if r <= tol
         break;
      end
   end

%-----------------------------------------------------------------------
% The system, written as f(x) = 0.
%-----------------------------------------------------------------------
function fx = fsys(v)
   x = v(1);  y = v(2);  z = v(3);
   fx = [x^2 + y^4 + z^6 - 2;
         cos(x * y * z^2) - (x + y + z);
         y^2 + z^3 - (x + y - z)^2];
end

%-----------------------------------------------------------------------
% Analytic Jacobian of fsys.
%-----------------------------------------------------------------------
function J = jsys(v)
   x = v(1);  y = v(2);  z = v(3);
   s = sin(x * y * z^2);
   u = x + y - z;
   J = [2*x,              4*y^3,           6*z^5;
        -y*z^2*s - 1,     -x*z^2*s - 1,    -2*x*y*z*s - 1;
        -2*u,             2*y - 2*u,       3*z^2 + 2*u];
end
