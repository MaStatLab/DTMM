
// [[Rcpp::depends(RcppArmadillo)]]
#include "RcppArmadillo.h"
#include "helpers.h"
#include "HDTM.hpp"
#include <vector>
#include <random>

using namespace Rcpp;
using namespace arma;
using namespace std;


// [[Rcpp::export]]
Rcpp::List HDTMcpp(arma::Mat<unsigned int> Y, arma::Mat<unsigned int> edge, arma::vec tau_vec, arma::vec nu_vec, arma::vec theta_vec, arma::Col<unsigned int> c,
                   arma::Col<unsigned int> gamma_sample, double alpha, unsigned int mcmc_iter, bool select)
{
    int n = Y.n_rows;
    
    class_HDTM myHDTM(Y, edge, tau_vec, nu_vec, theta_vec, c, gamma_sample, alpha, select);
    ////Col<unsigned int> obss = {3};
    
    ////double ml = myHDTM.compute_log_ml(obss);
    ////cout << ml << endl;
    
    //Mat<unsigned int> post_sample(n, mcmc_iter);
    //post_sample.fill(0);
    //post_sample = myHDTM.sample_mcmc(mcmc_iter);

    //arma::mat trans_data = myHDTM.transformed_data;
    vector<double> post_alpha;
    vector<double> post_lambda_global;
    
    vector<Mat<unsigned int> > post_sample = myHDTM.sample_mcmc(mcmc_iter, post_alpha, post_lambda_global); ////
    Mat<unsigned int> post_c = post_sample[0];
    Mat<unsigned int> post_gamma = post_sample[1];
    
    
    return Rcpp::List::create(
                              Rcpp::Named("post_c") = post_c,
                              Rcpp::Named("post_gamma") = post_gamma,
                              Rcpp::Named("post_alpha") = post_alpha,
                              Rcpp::Named("post_lambda") = post_lambda_global
                              );
}



// [[Rcpp::export]]
Rcpp::List HDTMcpp_centroid(arma::Mat<unsigned int> Y, arma::Mat<unsigned int> edge, arma::vec tau_vec, arma::vec nu_vec, arma::vec theta_vec,
                            arma::Col<unsigned int> c, arma::Col<unsigned int> gamma_sample)
{
    int n = Y.n_rows;
    class_HDTM myHDTM(Y, edge, tau_vec, nu_vec, theta_vec, c, gamma_sample, 1, 0);
    
    vector<vector<double> > centroid;
    centroid = myHDTM.compute_centroid();
    //  centroid = myHDTM.compute_diff_disp();
    return Rcpp::List::create(
                              Rcpp::Named("centroid") = centroid
                              );
}






