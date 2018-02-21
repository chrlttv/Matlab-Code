function [x, z_i, w_j]=simbloccont(num_samples, num_features, mu, sigma, prop_r, prop_c)

% simbloccont function: simulate continuous a (num_samples,num_features)
% dataset, with a structure into (g,m) blocks/co-clusters

% Input 
% - num_samples: the number of samples or rows of the data 
% - num_features: the desired number of features
% - mu : the mean of each block; matrix of size (g,m)
% - sigma: the standard deviation of each block; matrix of size (g,m).
% Default value is 0.1 for each block.
% - prop_r and prop_c = proportions of row and column clusters,
% respectively. Default is equal proportion.

[g,m] = size(mu);

if (nargin==3)
    sigma = 0.1*ones(g,m);
    p = (1/g * ones(1,g)')';
	q = (1/m * ones(1,m)')';
end 

if (nargin==4)
	p = (1/g * ones(1,g)')';
	q = (1/m * ones(1,m)')';
end

if (nargin==5)
    p = prop_r;
    q = (1/m * ones(1,m)')';
end 

if (nargin==6)
    p = prop_r;
    q = prop_c;
end

z_i=partitionrnd(num_samples,p,g);
w_j=partitionrnd(num_features,q,m);

for i=1:num_samples
   for j=1:num_features
      x(i,j)=normrnd(mu(z_i(i),w_j(j)),sigma(z_i(i),w_j(j)),1,1);
   end
end

function z_i=partitionrnd(n,p_k,g)
tmp=unifrnd(0,1,n,1);
z_i=g+1-sum(tmp*ones(1,g)<ones(n,1)*cumsum(p_k),2);






