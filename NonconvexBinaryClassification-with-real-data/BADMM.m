%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%     Copyright (C) 2023 Jianghua Yin
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <https://www.gnu.org/licenses/>.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Apply BADMM to solve the so-called nonconvex binary classification as follows:
%
% min  f(x)+g(y)
% s.t. -x+y=0,
% where $f(x)=\mu\|x\|_1,\ g(y)=\frac{1}{m}\sum_{i=1}^{m}(1- \tanh(b_ia_i^Ty))$, 
% the data pairs $(a_i,b_i)_{i=1}^m$ are a given dataset or drawn from 
% a given distribution.
%
%
%% Initialization and Preparation
% input imformation£ºinitial point |x|£¬objection function associated with $g$ |fun|, 
% (including the value of the objective function and the gradient at |y|)£¬
% parameters |opts|
%

function [feas, objF, beta] = BADMM(x0, fun, alpha, opts)
%
maxit   = opts.maxit;   epsilon  = opts.epsilon;
Lg      = opts.Lg;      mu       = opts.mu;

%%%
varphi_alpha = max(1/alpha,alpha^2/(1+alpha-alpha^2));
Lgvarphi = Lg*varphi_alpha;
mvarphi_alpha = 12*varphi_alpha;
cons_alpha = mvarphi_alpha+1;
barbeta = Lgvarphi*(6+6*sqrt(3+24*varphi_alpha))/cons_alpha;
beta = 1.01*barbeta;
Theta = sqrt(cons_alpha*beta^2-12*Lgvarphi*beta-72*Lgvarphi^2);
mcons_alpha = cons_alpha*beta;
gamma = (mvarphi_alpha/(mcons_alpha*beta+Theta)+mvarphi_alpha/(mcons_alpha-Theta))/2;
%% initial points
y0 = x0;
lam0 = x0;

%%% define a handle function
nabL_y = @(y,x,lam,beta) fun(y,2)-lam+beta*(y-x);


%% µü´úÖ÷Ñ­»·
for iter = 1:maxit
    %%% update x
    ak = y0-lam0/beta;
    x1 = prox(ak,mu/beta);
    %%% update y
    y1 = y0-gamma*nabL_y(y0,x1,lam0,beta);
    %%% update lambda
    lam1 = lam0+alpha*beta*(x1-y1);
    
    % relative accuracy
    normdif = [norm(x1-x0);norm(y1-y0);norm(lam1-lam0)];
    normpri = [norm(x0);norm(y0);norm(lam0);1];
    if max(normdif)<epsilon*max(normpri)
        feas(iter) = norm(x1-y1,inf);
        objF(iter) = mu*norm(x1,1)+fun(y1,1);
        break;
    end
    feas(iter) = norm(x1-y1,inf);
    objF(iter) = mu*norm(x1,1)+fun(y1,1);
%     % adaptive update scheme for the penalty parameter
%     % use the following heuristics to update beta
%     if norm(x1)+norm(y1)+norm(lam1)>1e10 && sum(normdif)>1000/iter
%         beta = min(2*beta,1.01*barbeta);
%     end
    % update imformation
    x0 = x1;
    y0 = y1;
    lam0 = lam1;
end
end
%%%
% the proximal operator of $\mu\|x\|_1$ is as follows: $\mathrm{sign}(x)\max\{|x|-\mu,0\}$¡£
function y = prox(x, mu)
y = max(abs(x) - mu, 0);
y = sign(x) .* y;
end
