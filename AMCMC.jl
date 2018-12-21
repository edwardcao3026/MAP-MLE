#################################################################
# Adaptive MCMC algorithm for inferring feedback loop with bursts
# Author: Edward Cao (the University of Edinburgh)
# Email: edward.cao@ed.ac.uk
#################################################################

using DelimitedFiles, DifferentialEquations, Distributions, Plots, LinearAlgebra

# Specify the location of moment measurement data

# Load moment measurement data
m1 = readdlm("$(homedir())/Desktop/MAP-MLE/Moment_Extract/m1e5.csv",'\n',Float64,'\n');
m2 = readdlm("$(homedir())/Desktop/MAP-MLE/Moment_Extract/m2e5.csv",'\n',Float64,'\n');
s1 = readdlm("$(homedir())/Desktop/MAP-MLE/Moment_Extract/s1e5.csv",'\n',Float64,'\n');
s2 = readdlm("$(homedir())/Desktop/MAP-MLE/Moment_Extract/s2e5.csv",'\n',Float64,'\n');

# Time points of interest
Tl = collect(1:1:30);

# Select proposal distribution
vj = 1^2*Matrix{Float64}(I,5,5);
prop(gamma_n,gamma_o) = pdf(MvLogNormal(log.(gamma_o),vj),gamma_n);

# Metropolis Hastings MCMC
n_samp = 10000000;
burnin = 3000000;
gamma_o = [6.0;1.0;2;2e-3;0.2];
gamma_r = zeros(5,Int(n_samp));
gamma_logr = zeros(5,Int(n_samp));
gamma_i = gamma_o;

# Initial condition for 3MA and LMA
x0_3ma = [0.0;1.0;0.0;0.0;1.0;0.0;0.0;0.0;1.0];

# Moment equations
function me_3ma(dx,x,p,t)
    np = x[1];
    ng = x[2];
    npp = x[3];
    npg = x[4];
    ngg = x[5];
    nppp = x[6];
    nppg = x[7];
    npgg = x[8];
    nggg = x[9];

    npppg = 3*npg*(-2*np^2+npp) + ng*(nppp+6*np^3-6*np*npp) + 3*np*nppg;
    npggg = 6*ng^3*np - 6*ngg*npg + 3*npg*ngg + 3*ng*(-2*np*ngg+npgg) + np*nggg;
    nppgg = 2*(npg^2 - np^2*ngg + np*npgg + ng*(-4*np*npg+nppg)) + ng^2*(6*np^2-2*npp) + npp*ngg;

    ru = p[1];
    b = p[2];
    d = p[3];
    sb = p[4];
    su = p[5];

    dx[1] = (ru*b-su)*ng - d*np - sb*npg + su;
    dx[2] = -sb*npg - su*ng + su;
    dx[3] = 2*ru*b*npg + ru*(2*b^2+b)*ng - 2*d*npp + d*np + sb*npg - 2*sb*nppg + su*(1-ng+2*np-2*npg);
    dx[4] = ru*b*ngg - d*npg + sb*(npg-npgg-nppg) + su*(1-ngg+np-npg);
    dx[5] = sb*npg - 2*sb*npgg + su + su*ng - 2*su*ngg;
    dx[6] = ru*b*(6*b^2+6*b+1)*ng + 3*ru*b*(2*b+1)*npg + 3*ru*b*nppg - d*np +3*d*npp - 3*d*nppp - sb*npg + 3*sb*nppg - 3*sb*npppg + su*(1-ng+3*np-3*npg+3*npp-3*nppg);
    dx[7] = ru*b*(1+2*b)*ngg + 2*ru*b*npgg + d*npg - 2*d*nppg + sb*(-npg+npgg+2*nppg-2*nppgg-npppg) + su*(1-ngg+2*np-2*npgg+npp-nppg);
    dx[8] = ru*b*nggg - d*npgg + sb*(-npg+2*npgg-npggg+nppg-2*nppgg) + su*(1+ng-ngg-nggg+np+npg-2*npgg);
    dx[9] = sb*(-npg+3*npgg-3*npggg) + su*(1+2*ng-3*nggg);
end

# Select prior distribution for gamma
prior(gamma) = pdf(LogNormal(log(gamma_i[1]),1.0),gamma[1])*pdf(LogNormal(log(gamma_i[2]),1.0),gamma[2])*pdf(LogNormal(log(gamma_i[3]),0.5),gamma[3])*pdf(LogNormal(log(gamma_i[4]),0.5),gamma[4])*pdf(LogNormal(log(gamma_i[5]),0.5),gamma[5]);

# Time sampling for solving ODE
spp = 0.01;

# Time span for solving ODE
tspan = (0.0,30.0);

u_til_o = zeros(2,length(Tl));
u_til_n = zeros(2,length(Tl));

p_accept = 0.0;
acc = 0;

#Random.seed!(1234)
# Adaptive parameters
ϵ_d = 2e-5*Matrix{Float64}(I,length(gamma_o),length(gamma_o));
s_d = (2.38)^2/length(gamma_o);
thresh_i = 100;
optimal = 0.23;
beta = 0.05

gamma_o =  [10.5055285019;3.0697881064;0.734155495;0.000584985;0.0656114629]

@time for i = 1 : n_samp
    # Propose new gamma
    gamma_n = rand(MvLogNormal(log.(gamma_o),vj));

    # Calculate tilde u for gamma old
    prob1 = ODEProblem(me_3ma,x0_3ma,tspan,gamma_o);
    sol1 = solve(prob1,Tsit5(),saveat=spp);
    # Extract moments from moment equations and transform to central moments
    for j = 1 : length(Tl)
        t_index = Int(Tl[j]/spp);
        u_til_o[:,j] = [sol1.u[t_index][1];sol1.u[t_index][3]-(sol1.u[t_index][1])^2];
    end
    # Calculate tilde u for gamma new
    prob2 = ODEProblem(me_3ma,x0_3ma,tspan,gamma_n);
    sol2 = solve(prob2,Tsit5(),saveat=spp);
    for j = 1 : length(Tl)
        t_index = Int(Tl[j]/spp);
        u_til_n[:,j] = [sol2.u[t_index][1];sol2.u[t_index][3]-(sol2.u[t_index][1])^2];
    end

    # Calculate proposal pdf for both gamma new and gamma old
    prop_on = prop(gamma_o,gamma_n);
    prop_no = prop(gamma_n,gamma_o);

    # Calculate posterior distribution ratio (avoiding numerical issue)
    post_r = prior(gamma_n)/prior(gamma_o);

    for j = 1 : length(Tl)
        post_r = post_r*pdf(Normal(u_til_n[1,j],sqrt(s1[j])),m1[j])*pdf(Normal(u_til_n[2,j],sqrt(s2[j])),m2[j])/pdf(Normal(u_til_o[1,j],sqrt(s1[j])),m1[j])/pdf(Normal(u_til_o[2,j],sqrt(s2[j])),m2[j]);
    end
    # Calculate acceptance probability
    p_accept = min(1.0,post_r*prop_on/prop_no);

    global acc
    global gamma_o
    global ind
    ind = 0.0;
    if rand() < p_accept
        gamma_o = gamma_n;
        acc = acc + 1;
        ind = 1.0;
    end
    acr = acc/i;

    # Record gamma old
    gamma_r[:,i] = gamma_o;
    gamma_logr[:,i] = log.(gamma_o)
    gamma_i = log.(gamma_o)
    println("Step: $i, Acceptance: $acr")

    # Update covariance for adaptive purposes
    global vj, s_d

    delta = i^(-0.9)*100*(ind-optimal)
    s_d = exp(log(s_d)+delta);
    if i > thresh_i && i < 500000
        vj = s_d*cov(gamma_logr[:,1:i]')+s_d*ϵ_d;
    end
end

nbins = 100;

writedlm("$(homedir())/Desktop/MAP-MLE/MCMC_Infer/ru_e5.csv",gamma_r[1,:])
writedlm("$(homedir())/Desktop/MAP-MLE/MCMC_Infer/b_e5.csv",gamma_r[2,:])
writedlm("$(homedir())/Desktop/MAP-MLE/MCMC_Infer/d_e5.csv",gamma_r[3,:])
writedlm("$(homedir())/Desktop/MAP-MLE/MCMC_Infer/sb_e5.csv",gamma_r[4,:])
writedlm("$(homedir())/Desktop/MAP-MLE/MCMC_Infer/su_e5.csv",gamma_r[5,:])

plt1 = histogram(gamma_r[1,burnin:n_samp],bins=nbins,normed=true,linewidth=0,title="r_u")
plt2 = histogram(gamma_r[2,burnin:n_samp],bins=nbins,normed=true,linewidth=0,title="b")
plt3 = histogram(gamma_r[3,burnin:n_samp],bins=nbins,normed=true,linewidth=0,title="d")
plt4 = histogram(gamma_r[4,burnin:n_samp],bins=nbins,normed=true,linewidth=0,title="s_b")
plt5 = histogram(gamma_r[5,burnin:n_samp],bins=nbins,normed=true,linewidth=0,title="s_u")
plt6 = plot()

plot(plt1,plt2,plt3,plt4,plt5,plt6,layout=(3,2),legend=false)


#plot(sol.t,sol.u)
