function [z_i,w_j,theta,info]=bcem_poisson(x,g,m,maxIt,threshold)

% Poisson parsimonious co-clustering:
% For details see "Co-clustering, Govaert G. and Nadif M. - 2014"
% Inputs:
%        x              :     Input dataset of size n (number of
%                             objects) * p (number of features)
%        g              :     Number of row clusters required
%        m              :     Number of column clusters required
%        maxIt          :     Maximum number of iterations until
%                             convergence. Default value is set to 200.
%        initChoice      :     Parameter for selecting the inialisation method
%                             - 0 random initialisation (default value)
%                             - 1 initialisation with kmeans          
%                             
%        threshold      :    Stopping threshold for the convergence of the criterion.
%                            Default value is set to 10^-10.
% Outputs:
%  
%           z_i       - Partition vector of rows
%           w_j       - Partition vector of columns.
%           theta     - Structure array that contains the estimated
%                       parameters gamma and pi
%           info      - Another structure array containing the value of the
%                       criterion at each iteration, the number of iterations and the
%                       time (in sec) required for the convergence


%% Initialization
[n,d]=size(x);
%---- Partition------%
z_i=ceil(g*rand(1,n));
w_j=ceil(m*rand(1,d));
z_ik=sparse(1:n,z_i,1,n,g,n);
w_jl=sparse(1:d,w_j,1,d,m,d);

%---- Parameters------%
gamma=bsxfun(@rdivide,((z_ik)'*(x*w_jl)),((z_ik'*sum(x,2))*(sum(x,1)*w_jl)));
pi=sum(z_ik,1)/n;
%rho=sum(w_jl,1)/d;
L_old=realmax;
L=[];
test=1;t=1;
tic;
while (test && t<maxIt) 
%%% Estimation Step    
 x_il=x*w_jl;
 [~,z_i]=max(x_il*log(gamma)'+repmat(log(pi),n,1),[],2);   
 z_ik=sparse(1:n,z_i,1,n,g,n);
 gamma=bsxfun(@rdivide,((z_ik)'*(x*w_jl)),((z_ik'*sum(x,2))*(sum(x,1)*w_jl))); 
 z_k=sum(z_ik,1);
 pi=z_k/n;
%% M-Step
 % Computation of w
  x_kj=z_ik'*x;
 [~,w_j]=max(x_kj'*log(gamma),[],2); 
  w_jl=sparse(1:d,w_j,1,d,m,d);
  
  % Parameters
  gamma=bsxfun(@rdivide,((z_ik)'*(x*w_jl)),((z_ik'*sum(x,2))*(sum(x,1)*w_jl)));

%% Computation of Lc 
  x_i=sum(x,2);
  x_j=sum(x,1);
  Lcmat1=sum(sum(((z_ik*log(gamma))*w_jl').*x));
  Lcmat2=sum(sum(((z_ik*log(gamma))*w_jl').*(x_i*x_j)));
  L(t)=sum(sum(z_ik,1).*log(pi))+(Lcmat1-Lcmat2);
  test=(abs(L(t)-L_old)>=threshold);
  L_old=L(t);
  t=t+1;
end
time=toc;
[~,z_i]=max(z_ik,[],2);
theta.pi=pi;
theta.gamma=gamma;
info.L=L;
info.time=time;
end