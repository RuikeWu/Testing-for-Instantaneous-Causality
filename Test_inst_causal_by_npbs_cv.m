% Y id input data with 2 X T, T is sample size
% p is the number of lag term
% B is the loop for bootstrap procedure 
% d_1 is the dimension for first part
% cf is confidence level such as 0.95

function [pval,J_T,ifa] = Test_inst_causal_by_npbs_cv(Y,p,B,d_1,cf)
    [d,T] = size(Y);

    % Compute OLS estimator
    Id = eye(d);   
    V0 = zeros(p*d,p*d);
    V3 = zeros(d,p*d);
    for t = (p+1):T
        X_t_1 = [];
        for i = 1 : p
            X_t_1 = [X_t_1,(Y(:,t-i))'];
        end
        X_t_1 = X_t_1';
        V0 = V0 + X_t_1* X_t_1';
        V3 = V3 + Y(:,t) * X_t_1';
    end 
    V1 = kron(V0,Id);
    V4 = reshape(V3,[d*p*d,1]);
    II_hat = inv(V1)*V4; 
    
    %  obtain residual u_hat
    u_hat = [];
    X_t_1_set = [];
    for t = p+1:T %从p+1开始循环，p是lag数
         X_t_1 = [];
        for i = 1 : p
            X_t_1 = [X_t_1,(Y(:,t-i))'];
        end
        X_t_1 = X_t_1';
        X_t_1_set(:,t-p) = X_t_1;
        u_hat(:,t-p) = Y(:,t)-(kron(X_t_1',Id))*II_hat; 
    end
    % calculate u1*u2
    cov_x_set = [];
    for x = 1 : size(u_hat,2)
        u1_hat = u_hat(1:d_1,x);
        u2_hat = u_hat(d_1+1:d,x);
        cov_x_set(x) = u1_hat * u2_hat';
    end
    
    %define u1 and u2 with d1 d2 , d1 + d2 = d
    d_2 = d -d_1; % dimension of u2
    
    %Bootstrap procedure for critical values
    J_it_set = []; 
    for ii = 1: B     
        m_hat_i = []; 
        yibusu_set = [];
        cov_i_set = []; 
        for i = 1: size(u_hat,2)
            u1i_hat = u_hat(1:d_1,i);
            u2i_hat = u_hat(d_1+1:d,i);
            cov_i = u1i_hat * u2i_hat';
            mi_hat = reshape(cov_i,[d_1*d_2,1]);
            yibusu_it = randn(1);
            m_hat_i(:,i) = yibusu_it * mi_hat;
            cov_i_set(i) = yibusu_it * cov_i;
        end
        
        % using cross-validation to select bandwidth
        h = cv_for_band_bt(cov_i_set,T,p);
        
        
        % obtain lambda_hat phi_hat and J_T
        lambda_hat = 0;
        phi_hat = 0;
        for t = 1 : size(m_hat_i,2)
            for s = 1:size(m_hat_i,2)
                if s == t
                    continue;
                end
                lambda_hat = lambda_hat + (1/h)*kernel(T,t,s,h) * m_hat_i(:,t)'*m_hat_i(:,s);
                phi_hat = phi_hat  + (1/(h^2))* kernel(T,t,s,h)^2 * (m_hat_i(:,t)'*m_hat_i(:,s))^2;
            end
        end
        lambda_hat = (1/(T^2))*lambda_hat;
        phi_hat = 2*h*(T^(-2))*phi_hat;
        J_T_i = (T*h^0.5)*lambda_hat/sqrt(phi_hat);
        J_it_set(ii) = J_T_i;
    end
    
    
    %Calculate J_T
    h = cv_for_band_bt(cov_x_set,T,p);    
    m_hat = [];
    d_2 = d -d_1; % dimension of u2
    for i = 1: size(u_hat,2)
        u1i_hat = u_hat(1:d_1,i);
        u2i_hat = u_hat(d_1+1:d,i);
        cov_i = u1i_hat * u2i_hat';
        mi_hat = reshape(cov_i,[d_1*d_2,1]);
        m_hat(:,i) = mi_hat;
    end

    % obtain lambda_hat phi_hat and J_T
    lambda_hat = 0;
    phi_hat = 0;
    for t = 1 : size(m_hat,2)
        for s = 1:size(m_hat,2)
            if s == t
                continue;
            end
            lambda_hat = lambda_hat + (1/h)*kernel(T,t,s,h) * m_hat(:,t)'*m_hat(:,s);
            phi_hat = phi_hat  + (1/(h^2))* kernel(T,t,s,h)^2 * (m_hat(:,t)'*m_hat(:,s))^2;
        end
    end
    lambda_hat = (1/(T^2))*lambda_hat;
    phi_hat = 2*h*(T^(-2))*phi_hat;
    J_T = (T*h^0.5)*lambda_hat/sqrt(phi_hat);

    q = quantile(J_it_set,cf);
    ifa = (J_T > q);
    pval = 1 - sum(J_T > J_it_set)/B;
end
