function x = newtond(f, jac, x0, tol)
% f, jac: system and Jacobian; x0: initial guess; tol: relative
% tolerance on the update.  x: root (column vector).
   x = x0(:);
   for iter = 1 : 100
      dx = jac(x) \ (-f(x));
      x = x + dx;
      if norm(dx) <= tol * norm(x), return; end
   end
end
