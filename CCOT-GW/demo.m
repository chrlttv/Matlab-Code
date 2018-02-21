%% Example of CCOT-GW on synthetic data
%% Code by C. Laclau

% Contains the code for simulation and evaluation
addpath('misc')
%% Generate data - Initialisation
% n is the number of rows, d is the number of columns of the data matrix
n=300;d=200;

% Mean and std of each bloc to generate the data matrix
mu= [ 4.0 0.5 1.5;
      1.8 4.5 5.1;
      3.5 1.5 5.5];
[g,m]=size(mu);  
sigma=0.01*ones(g,m);


% Regularisation parameter of the entropic term
gamma = 1;

% Number of simulations
nbsim=10;

% Count the number of time CCOT-GW identify the correct number of co-clusters
success=0;

err_cc=[];m_estim=[];err_r=[];g_estim=[];err_c=[];time=[];

%% Run CCOT-GW 
for j=1:nbsim
    
disp('[Generate the data ...]');
[X,z0,w0]=simbloccont(n,d,mu,sigma);

disp('[Run CCOT GW...]');
[alpha, beta, z, w, run_time, Kbary] = ccot_gw(X, gamma);

time(j) = run_time;

% If the number of co-clusters is correct, compute the co-clustering error
if ((length(unique(w))==length(unique(w0)))&&(length(unique(z))==length(unique(z0))))
           [err_cc(j),err_r(j),err_c(j)]=coClusError(z0,w0,z,w);
           success=success+1;         

% otherwise assign NaN value 
else
           err_cc(j)=NaN;err_r(j)=NaN;err_c(j)=NaN;        
end
  disp(['error on rows=',num2str(err_r(j)), ' | error on columns=',num2str(err_c(j)),' | co-clustering error=',num2str(err_cc(j)),...
      ' | time=',num2str(time(j))]);
  disp(['g: ', int2str(max(z)), ' - m: ', int2str(max(w))]);
  
end

disp('[*******End*******]');
disp(['The average cce is ',num2str(round(nanmean(err_cc),3))]);
