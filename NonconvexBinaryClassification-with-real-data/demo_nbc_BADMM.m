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


%% Demo：Apply BADMM to solve the so-called nonconvex binary classification as follows:
%
% $$ \displaystyle\min_x \frac{1}{m}\sum_{i=1}^{m}(1- \tanh(b_ia_i^Tx)) + \mu\|x\|_1, $$
% 
% or equivalently,
% min  f(x)+g(y)
% s.t. -x+y=0,
% where $f(x)=\mu\|x\|_1,\ g(y)=\frac{1}{m}\sum_{i=1}^{m}(1- \tanh(b_ia_i^Ty))$, 
% the data pairs $(a_i,b_i)_{i=1}^m$ are a given dataset or drawn from 
% a given distribution.
%
% Direct calculation yields the gradient and Hessian matrix of $g$:
% $$ \begin{array}{rl}
% \displaystyle\nabla g(y) &\displaystyle\hspace{-0.5em}=
% \frac{1}{m}\sum_{i=1}^{m}b_i[tanh^{2}(b_ia_i^Ty)-1]a_i, \\
% \displaystyle H(y)&\displaystyle\hspace{-0.5em}= \frac{2}{m}\sum_{i=1}^{m} 
% \tanh(b_ia_i^Ty)[1-tanh^{2}(b_ia_i^Ty)]a_ia_i^{\top}. 
% \end{array}$$
%
%
%
% set random seed
clear all;
close all;
clc;
seed = 97006855;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss); 

%
% the given datasets
    dataset = {'a1a.t','a2a.t','a3a.t','a4a.t','a5a.t','a6a.t','a7a.t',...
    'a8a.t','a9a.t','colon-cancer','gisette_scale.t',...
    'madelon.t'};
mu = 0.01; % the regularization parameter
fid = fopen('mytext.txt','w');
%% run 
for i=1:12
    
    % compute A and b
    [b,A] = libsvmread(dataset{i});
    [m,n] = size(A);
    % compute the Lipschitz constant $L_g$ of $\nabla g(y)$
    AT = A';
    barA = AT*diag(b);
    if m < 2000
        tstart = tic;
        L_g = 2*norm(barA'*barA)/m;
        teig = toc(tstart);
    else
        clear opts
        opts.issym = 1;
        tstart = tic;
        L_g = 2*eigs(barA'*barA,1,'LM',opts)/m;
        teig = toc(tstart);
    end
    fprintf(' time for eigenvalues %g %.3e\n', teig, L_g)
    % set parameters
    clear opts
    opts = struct();
    opts.Lg = L_g;
    opts.mu = mu;
    opts.epsilon = 1e-3;
    opts.maxit = 2000;
    fun = @(y,model) nbc_loss(y,model,A,b);
    x0 = zeros(n,1);
    alpha1 = 0.8;
    tstart = tic;
    [feas1,objF1,beta1] = BADMM(x0,fun,alpha1,opts);
    Tcpu1 = toc(tstart);
    iter1 = length(feas1);
    Finfeas1 = feas1(end);
    FinobjF1 = objF1(end);
    
    alpha2 = 1;
    tstart = tic;
    [feas2,objF2,beta2] = BADMM(x0,fun,alpha2,opts);
    Tcpu2 = toc(tstart);
    iter2 = length(feas2);
    Finfeas2 = feas2(end);
    FinobjF2 = objF2(end);
    
    alpha3 = 1.2;
    tstart = tic;
    [feas3,objF3,beta3] = BADMM(x0,fun,alpha3,opts);
    Tcpu3 = toc(tstart);
    iter3 = length(feas3);
    Finfeas3 = feas3(end);
    FinobjF3 = objF3(end);
    
    alpha4 = 1.4;
    tstart = tic;
    [feas4,objF4,beta4] = BADMM(x0,fun,alpha4,opts);
    Tcpu4 = toc(tstart);
    iter4 = length(feas4);
    Finfeas4 = feas4(end);
    FinobjF4 = objF4(end);
    
    alpha5 = 1.618;
    tstart = tic;
    [feas5,objF5,beta5] = BADMM(x0,fun,alpha5,opts);
    Tcpu5 = toc(tstart);
    iter5 = length(feas5);
    Finfeas5 = feas5(end);
    FinobjF5 = objF5(end);
    
    fprintf(fid,'%s & %d %d & %0.3f & %d/%.3f/%.2e/%.2e & %d/%.3f/%.2e/%.2e\n & %d/%.3f/%.2e/%.2e & %d/%.3f/%.2e/%.2e & %d/%.3f/%.2e/%.2e\\\\\n', ...
        dataset{i}, m, n, teig, iter1, Tcpu1, Finfeas1, FinobjF1, iter2, Tcpu2, Finfeas2, FinobjF2, ...
        iter3, Tcpu3, Finfeas3, FinobjF3, iter4, Tcpu4, Finfeas4, FinobjF4, iter5, Tcpu5, Finfeas5, FinobjF5);
        % plot
%     fig = figure;   % semilogy,plot
    figure(i)
    k1 = 1:iter1;
    semilogy(k1-1, feas1, '-', 'Color',[1 0 0], 'LineWidth',2);
    hold on
    k2 = 1:iter2;
    semilogy(k2-1, feas2, '-', 'Color',[0 1 0], 'LineWidth',1.8);
    hold on
    k3 = 1:iter3;
    semilogy(k3-1, feas3, '-', 'Color',[0 0 1], 'LineWidth',1.5);
    hold on
    k4 = 1:iter4;
    semilogy(k4-1, feas4, '-', 'Color',[1 0.5 0], 'LineWidth',1.5);
    hold on
    k5 = 1:iter5;
    semilogy(k5-1, feas5, '-', 'Color',[1 0 1], 'LineWidth',1.5);
    legend('PBADMM1','PBADMM2','PBADMM3','PBADMM4','PBADMM5');
    ylabel('$\|x^k-y^k\|$', 'fontsize', 14, 'interpreter', 'latex');
    xlabel('Iteration No.');
%     grid on
%     title(dataset{i});
%     print(fig, '-depsc','lr_ILM.eps');  % -depsc2表示eps图像;-djpeg表示输出jpg图像
end
%% close file
fclose(fid);
%%%
