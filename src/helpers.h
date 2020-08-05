#ifndef HELPERS_H
#define HELPERS_H

#include "RcppArmadillo.h"


using namespace Rcpp;
using namespace arma;
using namespace std;


const int max_precompute_lgamma = 100000;

struct Node
{
    int key;
    Col<unsigned int> data;
    double theta_0_temp;
    double theta_0; //prior mean
    double log_ml; //log marginal likelihood
    unsigned int gamma; //coupling indicator
    double log_L0 = 0;
    double log_L1 = 0;
    double node_prob = 1;
    double log_theta;
    
    Node* left;
    Node* right;
    
};


vector<double> compute_lgamma(const int val_max);

double log_exp_x_plus_exp_y(double x, double y);

double eval_log_g(unsigned int y_l, unsigned int y_r, double theta, double tau);

double eval_log_h(double theta, double nu, double theta_0);

double eval_log_int_g(Col<unsigned int> Y_l, Col<unsigned int> Y_r, double theta, vec tau_vec);

double eval_log_int_g_exp(Col<unsigned int> Y_l, Col<unsigned int> Y_r, double theta, vec tau_vec, bool which);

double eval_log_int_h(double theta, vec nu_vec, double theta_0);

unsigned int rmultinom(vector<double> prob);

#endif
