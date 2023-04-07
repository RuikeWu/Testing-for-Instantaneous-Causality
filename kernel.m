% This function is the kernel function with the form 3/4 (1-x^2)*1(abs(x) < 1)
function result = kernel(T,t,x,h);
    d = (t - x)/(T*h);
    result = 0.75*(1-d*d)*(abs(d)<=1);
end