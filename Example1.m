clear;
clc;
T = 200;
h = 1;
cfl = 0.95;
B = 299;
LOOP = 1000;
%%
c = 0.6;
rng('default'); 
rng(1); %random seed
Y_set = cell(LOOP,1);
for count0 = 1:LOOP
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
    Y_set{count0} = Y;
end

%%
size_count = [];
parfor count1 = 1 :LOOP
    tic
    count1
    Y = Y_set{count1};
    [~,J_T,ifa] = Test_inst_causal_by_npbs(Y,1,B,1,cfl,h); 
    size_count(count1) = ifa;
    toc;
end
powerb = sum(size_count)/LOOP

%%
local_power = [0.058	0.106	0.226	0.455	0.683	0.873	0.965	1	1	1	1];
c_set = 0:0.1:1;
plot(c_set,local_power,'m-.','MarkerSize',4,'LineWidth',1);
xlabel('The deviation c from the null');
ylabel('Power');


