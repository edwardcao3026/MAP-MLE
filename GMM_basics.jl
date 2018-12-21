#################################################################
# Maximum Likelihood Estimator for inferring feedback loop with bursts
# Author: Edward Cao (the University of Edinburgh)
# Email: edward.cao@ed.ac.uk
#################################################################

using DelimitedFiles, DifferentialEquations, Distributions, LinearAlgebra

# Specify the location of moment measurement data
cd("$(homedir())/Desktop/MAP-MLE/Moment_Extract")

# Time points of interest
Tl = collect(1:1:30);

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


# Moment prediction error calculator
function Err_3ma(x_m,s_m,theta,Tl)
    x0_3ma = [0.0;1.0;0.0;0.0;1.0;0.0;0.0;0.0;1.0];
    spp = 0.01;
    tspan = (0.0,30.0);
    prob = ODEProblem(me_3ma,x0_3ma,tspan,theta)
    error = 1e5;
    try
        sol = solve(prob,saveat=spp);
        me_temp = zeros(length(Tl),2);
        for j = 1 : length(Tl)
            t_index = Int(Tl[j]/spp);
            me_temp[j,:]=[sol.u[t_index][1];sol.u[t_index][3]-(sol.u[t_index][1]).^2];
        end
        error = sum(sum( ((me_temp-x_m)./s_m).^2 ));
    catch
    end
    return error
end
