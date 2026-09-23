function x = newtond(f, jac, x0, tol)
% newtond: d-dimensional Newton iteration for f(x) = 0.
%
%   f:    Function of the form f(x), x a length-d vector, returning a
%         length-d column vector.
%   jac:  Function of the form jac(x) returning the d x d Jacobian.
%   x0:   Initial estimate (length-d column vector).
%   tol:  Convergence criterion: returns when the relative magnitude of the
%         update from iteration to iteration is <= tol.
%
%   x:    Estimate of root (length-d column vector).

   itmax = 100;

   x = x0(:);
   for iter = 1 : itmax
      dx = jac(x) \ (-f(x));
      x  = x + dx;
      if norm(dx) <= tol * norm(x)
         return;
      end
   end
   warning('newtond: did not converge in %d iterations', itmax);
end
