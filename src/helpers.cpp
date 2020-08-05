
#include <RcppArmadillo.h>
#include "helpers.h"


using namespace Rcpp;
using namespace arma;
using namespace std;


const vector<double> lgamma_computed = compute_lgamma(max_precompute_lgamma);

/****/

double log_exp_x_plus_exp_y(double x, double y)
{
    double result;
    if((std::isinf(fabs(x)) == 1 ) && (std::isinf(fabs(y)) == 0))
        result = y;
    else if ((std::isinf(fabs(x)) == 0) && (std::isinf(fabs(y)) == 1))
        result = x;
    else if ((std::isinf(fabs(x)) == 1) && (std::isinf(fabs(y)) == 1))
        result = x;
    else if (x - y >= 100) result = x;
    else if (x - y <= -100) result = y;
    else
    {
        if (x > y)
        {
            result = y + log(1 + exp(x-y));
        }
        else result = x + log(1 + exp(y-x));
    }
    return result;
}


/****/



vector<double> compute_lgamma(const int val_max)
{
    vector<double> lgamma_computed;
    lgamma_computed.push_back(0);
    
    for (int i = 1; i <= 100 * val_max; i++)
    {
        lgamma_computed.push_back(lgamma(i/100.00));
    }
    
    //std::cout << lgamma_computed[1186330] << std::endl;
    return lgamma_computed;
}




/****/

double eval_log_g(unsigned int y_l, unsigned int y_r, double theta, double tau)
{
    double res = R::lchoose(y_l + y_r, y_l);
    
    //double res1 = R::lchoose(y_l + y_r, y_l);
    //res1 += R::lbeta(theta * tau + y_l, (1 - theta) * tau + y_r);
    //res1 -= R::lbeta(theta * tau, (1 - theta) * tau);
    
    //std::cout << "theta=" << theta  << " val = " << theta * tau + y_l << " lgamma = " << lgamma(theta * tau + y_l) << "  my=  " << lgamma_computed[theta * tau + y_l] << std::endl;
    
    //int plus_1 = 100 * (theta * tau + y_l);
    //res += ( (theta * tau + y_l >= 0.01) ? lgamma_computed[100 * (theta * tau + y_l)] : lgamma(theta * tau + y_l) ) +
    //       ( ((1 - theta) * tau + y_r >= 0.01) ? lgamma_computed[100 * ((1 - theta) * tau + y_r)] : lgamma((1 - theta) * tau + y_r) ) -
    //       ( (tau + y_l + y_r >= 0.01) ?  lgamma_computed[100 * (tau + y_l + y_r)] : lgamma(tau + y_l + y_r) );
     
    //res -= ( (theta * tau >= 0.01) ? lgamma_computed[100 * (theta * tau)] : lgamma(theta * tau)) +
    //       ( ((1 - theta) * tau >= 0.01) ? lgamma_computed[100 * ((1 - theta) * tau)] : lgamma((1 - theta) * tau) ) -
    //       ( (tau >= 0.01) ? lgamma_computed[100 * tau] : lgamma(tau) );
    
    if (theta * tau < 0.01 || (1 - theta) * tau < 0.01)
    {
        res += R::lbeta(theta * tau + y_l, (1 - theta) * tau + y_r);
        res -= R::lbeta(theta * tau, (1 - theta) * tau);
    }
    else if (y_l + y_r + tau > max_precompute_lgamma)
    {
        res += R::lbeta(theta * tau + y_l, (1 - theta) * tau + y_r);
        res -= R::lbeta(theta * tau, (1 - theta) * tau);
    }
    else
    {
        res += lgamma_computed[100 * (theta * tau + y_l)] + lgamma_computed[100 * ((1 - theta) * tau + y_r)] - lgamma_computed[100 * (tau + y_l + y_r)] ;
        res -= lgamma_computed[100 * (theta * tau)] + lgamma_computed[100 * ((1 - theta) * tau)] - lgamma_computed[100 * tau];
    }
    
    //if(abs(res - res1) > 2){
    //    std::cout <<res - res1 << "   " << lgamma_computed[100 * (theta * tau)] << " truth = " << lgamma(theta * tau) << " at " << theta * tau << std::endl;
    //}
    
    
    return res;
}


/****/

double eval_log_h(double theta, double nu, double theta_0)
{
    //double res = -R::lbeta(theta_0 * nu, (1 - theta_0) * nu);
    double res = -lgamma_computed[100 * theta_0 * nu] - lgamma_computed[100 * (1 - theta_0) * nu] + lgamma_computed[100 * nu];
    res += (theta_0 * nu - 1) * log(theta) + ((1 - theta_0) * nu - 1) * log(1 - theta);
    return res;
}


/****/

double eval_log_int_g(Col<unsigned int> Y_l, Col<unsigned int> Y_r, double theta, vec tau_vec)
{
    int n_grid = tau_vec.n_elem;
    int n_sample = Y_l.n_elem;
    double sum_log_g = 0;
    double res = log(0.0);
    
    for (int g = 0; g < n_grid; ++g)
    {
        sum_log_g = 0;
        for (int s = 0; s < n_sample; ++s)
        {   
            sum_log_g += eval_log_g(Y_l(s), Y_r(s), theta, tau_vec(g));
        }
        
        res = log_exp_x_plus_exp_y(res, sum_log_g - log(n_grid));
    }
    
    return res;
}


/****/

double eval_log_int_g_exp(Col<unsigned int> Y_l, Col<unsigned int> Y_r, double theta, vec tau_vec, bool which)
{
    int n_grid = tau_vec.n_elem;
    int n_sample = Y_l.n_elem;
    double sum_log_g = 0;
    double res = log(0.0);
    /*
    if (which == 1)
    {
        for (int g = 0; g < n_grid; ++g)
        {
            sum_log_g = 0;
            for (int s = 0; s < n_sample; ++s)
            {
                sum_log_g += eval_log_g(Y_l(s), Y_r(s), theta, tau_vec(g));
            }
            
            res = log_exp_x_plus_exp_y(res, sum_log_g - log(n_grid));
        }
        res += log(theta);
    }
    else
    {
        for (int g = 0; g < n_grid; ++g)
        {
            sum_log_g = 0;
            for (int s = 0; s < n_sample; ++s)
            {
                sum_log_g += (eval_log_g(Y_l(s), Y_r(s), theta, tau_vec(g)) + log(tau_vec(g)) );
            }
            
            res = log_exp_x_plus_exp_y(res, sum_log_g - log(n_grid));
        }
    }
    */
    
    for (int g = 0; g < n_grid; ++g)
    {
        sum_log_g = 0;
        for (int s = 0; s < n_sample; ++s)
        {
            sum_log_g += eval_log_g(Y_l(s), Y_r(s), theta, tau_vec(g));
        }
        
        res = log_exp_x_plus_exp_y(res, sum_log_g - log(n_grid));
    }
    res += log(theta);
    
    return res;
}


/****/

double eval_log_int_h(double theta, vec nu_vec, double theta_0)
{
    int n_grid = nu_vec.n_elem;
    double res = log(0.0);
    
    for (int h = 0; h < n_grid; ++h)
    {
        res = log_exp_x_plus_exp_y(res, eval_log_h(theta, nu_vec(h), theta_0) - log(n_grid));
    }

    return res;
}


/****/

unsigned int rmultinom(vector<double> prob)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::discrete_distribution<int> multinom(prob.begin(), prob.end());
    
    unsigned int number = multinom(generator);
    return number;
}




























