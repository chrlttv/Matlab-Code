function [alpha, beta, z, w, time, Kbary] = ccot_gw(X, gamma, loss, plot_vector)

%% Run CCOT-GW from "Co-clustering through Optimal Transport", C. Laclau et al., ICML 2017
%% Code by C. Laclau - This code partially uses the code of Gabriel Peyre to compute the barycenters.
%%
%% Input: - a data matrix X
%%        - gamma : regularisation parameter of the entropic term 
%%        - loss : the type of loss used ('l2' - default or 'kl')
%%        - plot_vector : 1 to plot alpha and beta sorted, 0 otherwise
%% Output: vectors alpha and beta and partitions z and w, execution time

% Define weights: first weight is for the dissimilarity matrix calculated on rows; second for columns;
% In the article we kept it balanced
lambda=[0.5 0.5];

% Default value for gamma (1) and the loss (l2)
if (nargin==1)
   gamma=1;
   loss = 'l2';
elseif (nargin==2)
   loss = 'l2';
   plot_vector=0;
elseif (nargin==3)
   plot_vector=0;
end

% Define the options for the Wasserstein-Gromov algorithm (https://hal.archives-ouvertes.fr/hal-01322992 document)

options.gw_loss = loss; 

% We keep these options fixed as it was done in the paper
options.niter = 100;
options.niter_sinkhorn = 100;
options.tol_sinkhorn = 1e-6; % tolerance on marginals for Sinkhorn
options.tol_gw = 1e-4; % early stopping for GW
options.gamma_init = [];
options.verb = 1;
options.log_domain = 0;
options.verb = 0;
options.bary_method = 'alterating';
options.niter_alternating = 10;
options.tol_alternating = 1e-4;

% Calculate similarity matrices (kernel)) for rows and columns
tic;

[n, d] = size(X);

sigma_lin = mean(reshape(pdist2(X,X),1,n*n));
sigma_col = mean(reshape(pdist2(X',X'),1,d*d));

sim_lin = rbf(X,sigma_lin);
sim_col = rbf(X',sigma_col);

sims = {};
sims{1} = sim_lin;
sims{2} = sim_col;

% Calculate the Gromov-Wasserstein barycenter (implementation from G. Peyre)
addpath(genpath('gromov-wasserstein'),'misc')
[Kbary,~,beta,alpha] = compute_gw_barycenters(sims,sims{1},lambda,gamma,options);

% Convert alpha and beta into partitions

% First sort both vectors
[sort_a,idxa]=sort(beta{1});
[sort_b,idxb]=sort(beta{2});

% Jump detection - Code from B. Mattein
[idx_edges_b] = labeling(sort_b);
[idx_edges_a] = labeling(sort_a);

% From jump to label
if (~isempty(idx_edges_a))
    z=getLabel(idx_edges_a,idxa);
else
    z=zeros(1,n);
end

if (~isempty(idx_edges_b))
    w=getLabel(idx_edges_b,idxb);
else
    w=zeros(1,d);
end

% Plot the vectors (optional)

if (plot_vector==1)
    figure();
    set(gcf,'color','w');set(gca,'FontSize',18);
    plot(sort_a,'LineWidth',2)
    xlabel('Instances','FontSize',20);
    ylabel('\alpha','Fontsize',24);

    figure();
    set(gcf,'color','w');set(gca,'FontSize',18);
    plot(sort_b,'LineWidth',2)
    xlabel('Variables','FontSize',20);
    ylabel('\beta','Fontsize',24);
end

time = toc;
