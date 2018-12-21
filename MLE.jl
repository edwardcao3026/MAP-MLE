#################################################################
# Maximum Likelihood Estimator for inferring feedback loop with bursts
# Author: Edward Cao (the University of Edinburgh)
# Email: edward.cao@ed.ac.uk
#################################################################

using BlackBoxOptim, Random

include("$(homedir())/Desktop/MAP-MLE/GMM_basics.jl")

Para = zeros(6)

theta_r = [13.0,3.0,1.0,0.001,0.1]

# Load moment measurement data
m1 = readdlm("m1e5.csv",'\n',Float64,'\n');
m2 = readdlm("m2e5.csv",'\n',Float64,'\n');
s1 = readdlm("s1e5.csv",'\n',Float64,'\n');
s2 = readdlm("s2e5.csv",'\n',Float64,'\n');
s_m = sqrt.([s1 s2]);
  
x_2ma = zeros(length(m1),2);
x_2ma[:,1] = m1;
x_2ma[:,2] = m2;

println(m1[1])
Random.seed!(1236)

errfun(theta) = Err_3ma(x_2ma,s_m,exp.(theta),Tl);
SRange = [(1.0,5.0),(-2.0,3.0),(-2.0,1.5),(-15.0,-0.5),(-5.0,-0.2)]
opts = bbsetup(errfun; Method = :adaptive_de_rand_1_bin_radiuslimited, SearchRange = SRange, NumDimensions = 5, MaxSteps = 8e4)
res = bboptimize(opts)

thetax = best_candidate(res);
Para = [exp.(thetax)..., sum(abs.(   (exp.(thetax)-theta_r)./theta_r   )) / 5]
println(Para)
