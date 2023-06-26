function [h_opt,c_opt,j_opt] = cv_for_band_bt(cov_x_set,T,p)
    
    j_set = 1:30; 
    c_set = 1.05.^(j_set-15); 
    h_set = (T^(-1/5)).*c_set; % set the range of candidate bandwidth
    CV_set = [];
    for jj = 1 : length(h_set)
        h0 = h_set(jj);
        tau_x_set = [];
        s_set = 1:T;
        for  x = p+1 :T
            ker_set = kernel(T,x,s_set,h0);
            ker_set(x) = 0; % leave-one-out estimator         
            ker_set(1:p) =[];
            hat_tau_x_set = ker_set.*cov_x_set;
            hat_tau_x = sum(hat_tau_x_set)/sum(ker_set);
            tau_x_set(x-p)= hat_tau_x;
        end  
        CV_set(jj) =  sum((cov_x_set - tau_x_set).^2); 
    end
    [~,loc] = min(CV_set);
    h_opt = h_set(loc);
    j_opt = j_set(loc);
    c_opt = c_set(loc);

end