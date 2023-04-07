% Y is d X T input data, d is the dimension, T is sample size
% p is lag length of vector autoregressive model
% d_1 is the dimension of u1
% h is bandwidth coefficient, used bandwidth is  hT^(-0.2);
function [p_val,J_T] = Test_inst_causal_by_nonpara(Y,p,d_1,h)
    [d,T] = size(Y);
    h = h*T^(-0.2);
    Id = eye(d);
    
    % Compute OLS estimator
    md = varm(d,p);
    md.Constant = [0 ; 0];
    A = estimate(md,Y');
    K_part1 = [];
    for k_loop =  1 : p
        K_part1 = [K_part1,A.AR{k_loop}];
    end
    II_hat_matlab = K_part1;
    II_hat = reshape(II_hat_matlab,[size(II_hat_matlab,1)*size(II_hat_matlab,2) 1]);
    
    % Obtain residual u_hat
    u_hat = [];
    for t = p+1:T
         X_t_1 = [];
        for i = 1 : p
            X_t_1 = [X_t_1,(Y(:,t-i))'];
        end
        X_t_1 = X_t_1';
        u_hat(:,t-p) = Y(:,t)-(kron(X_t_1',Id))*II_hat; 
    end

    %obtain m_hat((d1*d2) X (T-p) matrix)
    %define u1 and u2 with d1 d2 , d1 + d2 = d
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
    p_val = 1 - normcdf(J_T);

end


