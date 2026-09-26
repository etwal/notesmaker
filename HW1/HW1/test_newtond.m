% Problem 2: root of the 3D system near (-1, 0.75, 1.5).
x0 = [-1.00; 0.75; 1.50];
tol = 1.0e-12;

x = newtond(@fsys, @jsys, x0, tol);
fprintf('x = %.12f  y = %.12f  z = %.12f   |f| = %.1e\n', x, norm(fsys(x)));

% Update history for the writeup table.
xh = x0;
for n = 1 : 20
   dx = jsys(xh) \ (-fsys(xh));
   xh = xh + dx;
   fprintf('%2d  %.3e\n', n, norm(dx) / norm(xh));
   if norm(dx) <= tol * norm(xh), break; end
end

function fx = fsys(v)
   x = v(1);  y = v(2);  z = v(3);
   fx = [x^2 + y^4 + z^6 - 2;
         cos(x*y*z^2) - (x + y + z);
         y^2 + z^3 - (x + y - z)^2];
end

function J = jsys(v)
   x = v(1);  y = v(2);  z = v(3);
   s = sin(x*y*z^2);  u = x + y - z;
   J = [2*x,          4*y^3,          6*z^5;
        -y*z^2*s - 1, -x*z^2*s - 1,   -2*x*y*z*s - 1;
        -2*u,         2*y - 2*u,      3*z^2 + 2*u];
end
