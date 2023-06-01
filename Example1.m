clear;
clc;
T = 200;   % set sample size 
cf = 0.95; % set significance level
B = 299;  % set bootstrap replications
LOOP = 1000; % set total replications
%%
c = 0; % data is generated under the null
rng('default'); 
rng(2023); % set the random seed
Y_set = cell(LOOP,1);
for ct = 1:LOOP
    x_0 = [0;0];
    Y = [];
    for r = 1 : T      
        sigma = [1.1-cos(11*r/T),c*sin(2*pi*r/T);c*sin(2*pi*r/T),1.1+sin(11*r/T)];
        u_1 = (sigma^0.5) * randn([2,1]);
        x_1 = [0.64,-1;-0.01,0.44]*x_0 + u_1;
        Y(:,r) = x_1;
        x_0 = x_1;
    end
    % Y is d X T matrix
    Y = Y(1:2,1:T);
    Y_set{ct} = Y;
end
%%
c = 0.5; % set the deviation from the null
Y_set_alter = cell(LOOP,1);
for ct = 1:LOOP
    x_0 = [0;0];
    Y = [];
    for r = 1 : T      
        sigma = [1.1-cos(11*r/T),c*sin(2*pi*r/T);c*sin(2*pi*r/T),1.1+sin(11*r/T)];
        u_1 = (sigma^0.5) * randn([2,1]);
        x_1 = [0.64,-1;-0.01,0.44]*x_0 + u_1;
        Y(:,r) = x_1;
        x_0 = x_1;
    end
    % Y is d X T matrix
    Y = Y(1:2,1:T);
    Y_set_alter{ct} = Y;
end

%% Compute the empirical size under the null hypothesis
size_count = [];
size_count_cv = [];
parfor ct = 1 :LOOP
    tic
    ct
    Y = Y_set{ct};
    pval = Test_inst_causal_by_npbs(Y,1,B,1,cf,0.75); % the bandwidth is set to 0.75T^(-1/5) 
    pval_cv = Test_inst_causal_by_npbs_cv(Y,1,B,1,cf); 
    size_count(ct) = pval;
    size_count_cv(ct) =  pval_cv;
    toc;
end
size099 = sum(size_count < 0.01)/LOOP;
size099_cv = sum(size_count_cv< 0.01)/LOOP;

size095 = sum(size_count < 0.05)/LOOP;
size095_cv = sum(size_count_cv< 0.05)/LOOP;

size09 = sum(size_count < 0.1)/LOOP;
size09_cv = sum(size_count_cv< 0.1)/LOOP;

Final_size = [size099,size095,size09;
		size099_cv,size095_cv,size09_cv];
%% Compute the empirical power under alternative hypothesis with deviation parameter c = 0.5

power_count = [];
power_count_cv = [];
parfor ct = 1 :LOOP
    tic
    ct
    Y = Y_set_alter{ct};
    pval = Test_inst_causal_by_npbs(Y,1,B,1,cf,0.75); % the bandwidth is set to 0.75T^(-1/5) 
    pval_cv = Test_inst_causal_by_npbs_cv(Y,1,B,1,cf); 
    power_count(ct) = pval;
    power_count_cv(ct) =  pval_cv;
    toc;
end
power099 = sum(power_count < 0.01)/LOOP;
power099_cv = sum(power_count_cv< 0.01)/LOOP;

power095 = sum(power_count < 0.05)/LOOP;
power095_cv = sum(power_count_cv< 0.05)/LOOP;

power09 = sum(power_count < 0.1)/LOOP;
power09_cv = sum(power_count_cv< 0.1)/LOOP;

Final_power = [power099,power095,power09;
		power099_cv,power095_cv,power09_cv];
%% Compute powers with ascending c from 0 to 1
c_set = 0:0.1:1;
rng('default');
rng(1);

power_local = [];
power_local_cv = [];
for i = 1 : length(c_set)
    c = c_set(i); % set the deviation from the null
    Y_set_alter = cell(LOOP,1);
    for ct = 1:LOOP
        x_0 = [0;0];
        Y = [];
        for r = 1 : T      
            sigma = [1.1-cos(11*r/T),c*sin(2*pi*r/T);c*sin(2*pi*r/T),1.1+sin(11*r/T)];
            u_1 = (sigma^0.5) * randn([2,1]);
            x_1 = [0.64,-1;-0.01,0.44]*x_0 + u_1;
            Y(:,r) = x_1;
            x_0 = x_1;
        end
        % Y is d X T matrix
        Y = Y(1:2,1:T);
        Y_set_alter{ct} = Y;
    end
    
    parfor ct = 1 :LOOP
    tic
    ct
    Y = Y_set_alter{ct};
    pval = Test_inst_causal_by_npbs(Y,1,B,1,cf,0.75); % the bandwidth is set to 0.75T^(-1/5) 
    pval_cv = Test_inst_causal_by_npbs_cv(Y,1,B,1,cf); 
    power_count(ct) = pval;
    power_count_cv(ct) =  pval_cv;
    toc;
    end
    power095 = sum(power_count < 0.05)/LOOP;
    power095_cv = sum(power_count_cv< 0.05)/LOOP;
    
    power_local(i) = power095;
    power_local_cv(i) = power095_cv;
    
    
end
    
%%  Plot the local curves of the tests
plot(0:0.1:1,power_local,'m-.','MarkerSize',4,'LineWidth',1);
hold on 
plot(0:0.1:1,power_local_cv,'b-.x','MarkerSize',4,'LineWidth',1);
legend('0.75', 'CV');
xlabel('The deviation c from the null');
ylabel('Power');


