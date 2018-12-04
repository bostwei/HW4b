%************************************************    	
%Econ 899 HW4b 
%Shiyan Wei
% 10/26/2018
%************************************************

%***********************************************
% This Script is to calcualte the value function and other variable given
% the price q
%**************************************************


clear all
clc

%% Paramterized and initiate
% # Paramterized 

alpha = 1.5;
beta = 0.8; 

% # Income
y=[1;
   0.05];

% # Markov procss matrix
Pi =[0.75, 0.25;
    0.25, 0.75];

% open economy real interest rate
r = 0.04;

% pooling contract borrowing constraint
a_ = -0.525;

% legal record keep technology paramter
rho = 0.9;

N = 200;



%% Pooling EQL Policy Function

% Initial guess the pooling price


% Asset holding space 
% - Pooling EQL using the pooling EQL constraint
Alb=a_;
Aub =5;


A = linspace(Alb,Aub,N)';

% the increment
inc = (Aub-Alb)/N;


% Find the location which is closed to 0
<<<<<<< HEAD
loc_0 = ceil((0-Alb)/inc)-1;
=======
loc_0 = ceil((0-Alb)/inc)+1;
>>>>>>> master


% generate the possible asset a grid, aa stead for a' grid
a = repmat(A,1,N);
aa = repmat(A,1,N);


%----------------- Initial guess for q-------------------
    neg_a= [ones(loc_0,1);zeros(N-loc_0,1)];
<<<<<<< HEAD
    q = 0.8179*neg_a + 1/(1+r)*(1-neg_a);
=======
    q = 0.9962*neg_a + 1/(1+r)*(1-neg_a);
>>>>>>> master
    
% q = 0.9962;



%------------Original Value Function---------------
% The value function when h = 0
% % - The frist vector is value function of employed, second is unemployed 
 v0_h0 = zeros(N,2);
 v0_h0d0 = zeros(N,2);
 v0_h0d1 = zeros(N,2);
% The value function when h = 1
 v0_h1 = zeros(N,2);
 

 

    
%-------------- calcuate the utility for h = 0 d = 0 --------------
% generate the instant consumption for h = 0 d = 0 
% It means given a row of a, how much will he/she consume if his/her aa is
% chosen the responded number 
c_h0d0e = y(1) + a - q.*aa';
% generate the consumption for unemployed
c_h0d0ue = y(2) + a - q.*aa';


% find consumption if c <0 the make it NAA
c_h0d0e(find(c_h0d0e <=0)) = NaN;
c_h0d0ue(find(c_h0d0ue <=0)) = NaN;

% compute the utility based on consumption
u_h0d0e = (c_h0d0e.^(1-alpha)-1)/(1-alpha) ;
u_h0d0ue = (c_h0d0ue.^(1-alpha)-1)/(1-alpha) ;

u_h0d0e(find(isnan(u_h0d0e))) = -inf;
u_h0d0ue(find(isnan(u_h0d0ue))) = -inf;
%-------------- calcuate the utility for h = 0 d = 1 --------------
% generate the instant consumption for h = 0 d = 1 
% It means given a row of a, how much will he/she consume if his/her aa is
% chosen the responded number 
c_h0d1 = (y * ones(1,N))';

% compute the utility based on consumption
u_h0d1 = (c_h0d1.^(1-alpha)-1)/(1-alpha) ;


%-------------- calcuate the utility for h = 1  --------------
% generate the instant consumption for h = 1  
% It means given a row of a, how much will he/she consume if his/her aa is
% chosen the responded number 
% Since aa could not be negtive, then replace aa<0 to be 0
aa_noneg = aa;
aa_noneg(find(aa <=0)) = 0; %?????????????????????????????????????????????????????
% q_h1 = 0*neg_a + 1/(1+r)*(1-neg_a);

c_h1e = y(1) + a - 1/(1+r).*aa';
% generate the consumption for unemployed
c_h1ue = y(2) + a - 1/(1+r).*aa';


% find consumption if c <0 the make it NAA
c_h1e(find(c_h1e <=0)) = NaN;
c_h1ue(find(c_h1ue <=0)) = NaN;
% compute the utility based on consumption
u_h1e = (c_h1e.^(1-alpha)-1)/(1-alpha) ;
u_h1ue = (c_h1ue.^(1-alpha)-1)/(1-alpha) ;

u_h1e(find(isnan(u_h1e))) = -inf;
u_h1ue(find(isnan(u_h1ue))) = -inf;


%%% prepare for the loop
% initiate generate value function for both state
 v1_h0 = zeros(N,2);
 v1_h0d0 = zeros(N,2);
 v1_h0d1 = zeros(N,2);
% The value function when h = 1
 v1_h1 = zeros(N,2);

 
% initiate decesion rules
dec_h0d0 = zeros(N,2);


% initial the distance of two value function
metric = 10;
iter = 0;
tol = 0.0001;
MaxIt = 2000;

% Find the location which is closed to 0
<<<<<<< HEAD
loc_0 = ceil((0-Alb)/inc)-1;
=======
loc_0 = ceil((0-Alb)/inc)+1;
>>>>>>> master

%time0 = cputime;
while metric > tol %&& iter < MaxIt 

    %--------Value function for h=0 d=0----------------
    w_h0d0e=(u_h0d0e + beta  * (v0_h0(:,1)' *  Pi(1,1) + v0_h0(:,2)' *  Pi(1,2)));
    w_h0d0ue = (u_h0d0ue + beta  * (v0_h0(:,1)' *  Pi(2,1) +v0_h0(:,2)' *  Pi(2,2)));
    [v1_h0d0(:,1), dec_h0d0(:,1)] = max(w_h0d0e,[],2) ; 
    [v1_h0d0(:,2), dec_h0d0(:,2)] = max(w_h0d0ue,[],2) ;
    
    % calculate the supnorm of two value function
    metric_h0d0 = max(max(abs(v1_h0d0-v0_h0d0)./v1_h0d0));
    
    % update the value function;
    v0_h0d0 = v1_h0d0;
    %--------Value function for h=0 d=1----------------

    % Begin calculate Value function 
     w_h0d1e  = (u_h0d1(:,1) + beta  * (v0_h1(loc_0,1)' *  Pi(1,1) + v0_h1(loc_0,2)' *  Pi(1,2)));
     w_h0d1ue = (u_h0d1(:,2) + beta  * (v0_h1(loc_0,1)' *  Pi(2,1) + v0_h1(loc_0,2)' *  Pi(2,2)));
    v1_h0d1(:,1) = w_h0d1e ; 
    v1_h0d1(:,2) = w_h0d1ue ;
    
   % update the value function; 
     v0_h0d1 = v1_h0d1;   
     
      %--------Value function for h=0----------------  
    v1_h0 = max(v1_h0d0,v1_h0d1);
    % default decesion on h = 0
    default =  v1_h0d0 < v1_h0d1;
    default = double(default);
    % asset decesion on h = 0 
    dec_h0 = dec_h0d0.*(1-default);
    dec_h0(find(dec_h0 == 0)) = loc_0;
    
     metric_h0 = max(max(abs(v1_h0-v0_h0)./v1_h0));
     v0_h0 = v1_h0;
    
      %--------Value function for h=1---------------- 
    w_h1e  = (u_h1e +  beta  * ((rho*v0_h1(:,1)' +(1- rho)* v0_h0(:,1)') *  Pi(1,1) + (rho*v0_h1(:,2)' +(1- rho)* v0_h0(:,2)') *  Pi(1,2)));
    w_h1ue = (u_h1ue + beta  * ((rho*v0_h1(:,1)' +(1- rho)* v0_h0(:,1)') *  Pi(2,1) + (rho*v0_h1(:,2)' +(1- rho)* v0_h0(:,2)') *  Pi(2,2)));
    % only allow select a'>0
    [v1_h1(:,1), dec_h1(:,1)] = max(w_h1e(:,loc_0:N),[],2) ; %(loc_0,N)
    [v1_h1(:,2), dec_h1(:,2)] = max(w_h1ue(:,loc_0:N),[],2) ;
    
%     dec_h1(find(dec_h1<loc_0)) = loc_0;
%       dec_h1 = dec_h1+loc_0;
    % calculate the supnorm of two value function
    metric_h1 = max(max(abs(v1_h1-v0_h1)./v1_h1));
    
    % update the value function;
    v0_h1 = v1_h1;     
      
     metric = max([metric_h1,metric_h0, metric_h0d0]); 
    iter = iter +1;
    %fprintf('The iteration is: %d, the distance is: %.3f.\n',iter,metric);
%     
end

<<<<<<< HEAD
 
=======
% plot the policy function for h0d0
figure(1)
plot(A,A(dec_h0d0(:,1)),A,A(dec_h0d0(:,2)));% the policy function for employment state
legend({'employed policy function','umemployed policy function'},'Location','southeast')
xlabel('a') 
ylabel('aa')
refline(1,0) 

% plot the policy function h1
figure(2)
plot(A,A(dec_h1(:,1)),A,A(dec_h1(:,2)));% the policy function for employment state
legend({'employed policy function','umemployed policy function'},'Location','southeast')
xlabel('a') 
ylabel('aa')
refline(1,0) 
>>>>>>> master

%% Law of Motion and Cross-Sectional Distribution

%--------------- Transition Funtion ----------------
%%% Transition for Non-Default
% Forming asset holding transition matrix only for h = 0

g_h0e = sparse(N,N);
g_h0ue = sparse(N,N);

for i = 1:N
%   if the state is e, given your asset choice in a is i, what is your 
%   asset holding choice at a' 
    g_h0e(i,dec_h0(i,1)) = 1;
%   if the state is ue, given your asset choice in a is i, what is your 
%   asset holding choice at a'

    g_h0ue(i,dec_h0(i,2)) =1;
end


% Forming asset holding transition maxtrix only for h = 1
g_h1e = sparse(N,N);
g_h1ue = sparse(N,N);

for i = 1:N
%   if the state is e, given your asset choice in a is i, what is your 
%   asset holding choice at a' 
    g_h1e(i,dec_h1(i,1)) = 1;
%   if the state is ue, given your asset choice in a is i, what is your 
%   asset holding choice at a'
    g_h1ue(i,dec_h1(i,2)) =1;

end

%---------Generate the cross-sectional asset distribution  
%   Trans is the transition matrix from state at t(row) to the state at t+1
%   (column). The eigenvector associate wit the unit eigenvalue of trans'
%   is the strationatry distribution.
 

trans_h0 = [g_h0e * Pi(1,1),  g_h0e * Pi(1,2) ;
             g_h0ue * Pi(2,1), g_h0ue * Pi(2,2)];
% after the trans, the row will be state t+1, column will be t
% Will be trans = [p(1,1)*g_e   p(2,1)*g_ue;
%                  p(1,2)*g_e   p(2,2)*g_ue];
trans_h0 = trans_h0'; 

trans_h1 = [g_h1e * Pi(1,1), g_h1e * Pi(1,2);
           g_h1ue * Pi(2,1),g_h1ue * Pi(2,2)];
% after the trans, the row will be state t+1, column will be t
% Will be trans = [p(1,1)*g_e   p(2,1)*g_ue;
%                  p(1,2)*g_e   p(2,2)*g_ue];
trans_h1 = trans_h1';   
 
% initiate the stationary probability
% the first N is given now is employment state, the second N is given now
% is unemployment.
mu_h0 = ones(2*N,1).*(1/(4*N));
mu_h1 = ones(2*N,1).*(1/(4*N));
d = default(:);

test =1 ;
Itrate = 0;

MaxIt = 2000;
while test> 10^(-8) && Itrate <= MaxIt
    % given my probability of 
    Tmu_h0 = trans_h0*mu_h0.*(1-d) + (1-rho)*trans_h1*mu_h1;
    Tmu_h1 = trans_h0*mu_h0.*d + rho*trans_h1*mu_h1;
    metrics_h0 = max(abs(Tmu_h0-mu_h0));
    metrics_h1 = max(abs(Tmu_h1-mu_h1));
    test = max(metrics_h0,metrics_h1);
    mu_h0 = Tmu_h0;
    mu_h1 = Tmu_h1;
%    disp(test);
    Itrate = Itrate +1;
end


%% Check the market clearing condition 
%-------------The lost rate D -------------
mu = mu_h0+mu_h1;
mu_e = mu(1:N);
mu_ue = mu_h0(N+1:2*N);

mu_d = trans_h0*mu_h0.*d;
D = [A;A]'*mu_d;

% generatet the borrowing amount L
borrow = [ones(loc_0,1);zeros(N-loc_0,1)];
d_borrow = [borrow;borrow];
mu_l = d_borrow.*trans_h0*mu_h0;
L = [A;A]'*mu_l;


Delta  = D/L;

<<<<<<< HEAD
%% Graphing

% plot the policy function for h0d0
figure(1)
plot(A,A(dec_h0d0(:,1)),A,A(dec_h0d0(:,2)));% the policy function for employment state
legend({'employed policy function','umemployed policy function'},'Location','southeast')
xlabel('a') 
ylabel('aa')
refline(1,0) 

% plot the policy function h1
figure(2)
plot(A,A(dec_h1(:,1)),A,A(dec_h1(:,2)));% the policy function for employment state
legend({'employed policy function','umemployed policy function'},'Location','southeast')
xlabel('a') 
ylabel('aa')
refline(1,0)

% plot the value function
=======
%% Graphing the Value function figure 

>>>>>>> master
figure(3)
plot(A,v0_h0(:,1),A,v0_h0(:,2),A,v0_h1(:,1),'--',A,v0_h1(:,2),'--');
legend({'employed value function No default History',...
    'umemployed value function No default History',...
    'employed value function with default History',...
    'umemployed value function with default History',}...
    ,'Location','southeast')
xlabel('a') 
ylabel('v')
xlim([-1 Aub ])
% ylim([-4 4])

<<<<<<< HEAD
%% debt to income level.
% Debt level is the debt amount times the mu
asset_e = A.* mu_e;
asset_ue = A.* mu_ue;

debt = sum(asset_e(1:loc_0)) + sum(asset_ue(1:loc_0));
income = sum(mu_e * y(1) + mu_ue *y(2));
debt_income = abs(debt/income);

%% The default decision
% plot the value function
default_rand = 0.015*normrnd(0,1,N,2) + default;
figure(4)
plot(A,default_rand(:,1),'o',A,default_rand(:,2),'o');
legend({'employed default decision',...
    'umemployed default decision'}...
    ,'Location','northeast')
xlabel('a') 
ylabel('default')
xlim([-1 Aub ])

%% Plot the measurement distribution
figure(5)
bar(A,mu_e);
hold on 
bar(A,mu_ue);
legend({'employed population',...
    'umemployed population'}...
    ,'Location','northeast')
xlabel('a') 
ylabel('population')
xlim([-1 2.5 ])
title('Pooling Contract Equilibrium Distribution');

%% calculating the statistics
%- average income
avg_inc = sum(mu_e) * y(1)  + sum(mu_ue) * y(2);

avg_saving = sum(asset_e(loc_0:length(A))) + sum(asset_e(loc_0:length(A)));

avg_default = sum(d.*[A;A].*mu);

avg_bondp = q'*(mu_ue+mu_e) / sum(mu_ue+mu_e);

%% %% wealth of each (s,a)

load('separating.mat')

lambda_v1_h1 = ((v1_h1_sep + 1/((1-alpha)*(1-beta)))./(v1_h1 + 1/((1-alpha)*(1-beta)))).^(1/(1-alpha)) - 1;

lambda_v1_h0 = ((v1_h0_sep + 1/((1-alpha)*(1-beta)))./(v1_h0 + 1/((1-alpha)*(1-beta)))).^(1/(1-alpha)) - 1;

figure(6)
plot(A,lambda_v1_h1(:,1),A,lambda_v1_h1(:,2));
hold on
plot(A,lambda_v1_h0(:,1),A,lambda_v1_h0(:,2));
legend({'Employed Default History', 'Unemployed Default History'...
    'Employed No Default history', 'Unemployed No Default History'}...
    ,'Location','northeast')
xlabel('a') 
ylabel('Comsumption Equivalence')
ylim([-.005 .02 ])
xlim([Alb,Aub])
title('Consumption Equivalence for different group of agent');
=======
>>>>>>> master
