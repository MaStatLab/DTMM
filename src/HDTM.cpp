

#include <RcppArmadillo.h>
#include "helpers.h"
#include "HDTM.hpp"

using namespace Rcpp;
using namespace arma;
using namespace std;


/***********************************************************************************************************************/
//constructor and destructor
/****/

class_HDTM::class_HDTM(Mat<unsigned int> Y,
                       Mat<unsigned int> edge,
                       vec tau_vec,
                       vec nu_vec,
                       vec theta_vec,
                       Col<unsigned int> c,
                       Col<unsigned int> gamma_sample,
                       double alpha,
                       bool select): Y(Y), edge(edge), tau_vec(tau_vec), nu_vec(nu_vec), theta_vec(theta_vec), c(c), gamma_sample(gamma_sample), alpha(alpha), select(select)
{
    n = Y.n_rows;
    p = Y.n_cols;
    //transformed_data.zeros(n, p - 1);
    init();
    
    compute_lambda_L0();
    //compute_subtree_log_ml(root, Y, obss);
    //std::cout << (root->left)->key << std::endl;
    
    /*auto uit = unique_label.cbegin();
    while (uit != unique_label.cend()) {
        cout << *uit << " ";
        ++uit;
    } */
    /*
    traverse(root, 1);
    std::cout  << std::endl;
    traverse(root, 2);
     */
    //get_transformed_data(root, transformed_data);
    // traverse(root, 2);
    //traverse(root, 1);
    
}


/****/
 
class_HDTM::~class_HDTM()
{
    clear_node(root);
}


/***********************************************************************************************************************/
//functions to initialize and deconstruct the tree

/****/

Node* class_HDTM::get_new_node(unsigned int key)
{
    Node* new_node = new Node;
    new_node->key = key;
    new_node->left = nullptr;
    new_node->right = nullptr;
    return new_node;
}


/****/

map<unsigned int, vector<unsigned int> > class_HDTM::get_edge_map(Mat<unsigned int> edge)
{
    map<unsigned int, vector<unsigned int> > edge_map;
    for (int i = 0; i < edge.n_rows; ++i)
    {
        edge_map[edge(i, 0)].push_back(edge(i, 1));
    }
    return edge_map;
}


/****/

Node* class_HDTM::construct_subtree(Node* root, map<unsigned int, vector<unsigned int> > edge_map)
{
    unsigned int curr_key = root->key;
    
    if (edge_map.find(curr_key) == edge_map.end())
    {
        return root;
    }
    else
    {
        Node* new_node_r = get_new_node(edge_map[curr_key][0]);
        root->right = construct_subtree(new_node_r, edge_map);
        
        Node* new_node_l = get_new_node(edge_map[curr_key][1]);
        root->left = construct_subtree(new_node_l, edge_map);
        return root;
    }
}


/****/

Node* class_HDTM::aggregate_data(Node* root, Mat<unsigned int> Y)
{
    if (root->left == nullptr)
    {
        root->data = Y.col(root->key - 1);
        root->theta_0_temp = 1.0/p; //uniform
        
        return root;
    }
    else
    {
        aggregate_data(root->left, Y);
        aggregate_data(root->right, Y);
        root->data = (root->left)->data + (root->right)->data;
        root->theta_0_temp = (root->left)->theta_0_temp + (root->right)->theta_0_temp;
        return root;
    }
}


/****/

Node* class_HDTM::get_theta_0(Node* root)
{
    if (root->left == nullptr)
    {
        root->theta_0 = root->theta_0_temp;
        return root;
    }
    else
    {
        get_theta_0(root->left);
        get_theta_0(root->right);
        //root->theta_0 = (root->left)->theta_0_temp / root->theta_0_temp;
        root->theta_0 = 0.5;
        return root;
    }
}


/****/

Node* class_HDTM::get_gamma(Node* root)
{
    if (root->left == nullptr)
    {
        root->gamma = 0;
        return root;
    }
    else
    {
        get_gamma(root->left);
        get_gamma(root->right);
        //root->theta_0 = (root->left)->theta_0_temp / root->theta_0_temp;
        root->gamma = gamma_sample(root->key - p - 1);
        return root;
    }
}


/****/

void class_HDTM::init()
{
    map<unsigned int, vector<unsigned int> > edge_map = get_edge_map(edge);
    auto it = edge_map.begin();
    root = new Node;
    root->key = it->first;
    
    root = construct_subtree(root, edge_map); //construct the tree from the edge matrix
    root = aggregate_data(root, Y); //add data to the tree (data at each node is the sum of the children's)
    root = get_theta_0(root);
    root = get_gamma(root);
    
    cluster_label = init_label(c);
    unique_label = get_unique_label(cluster_label);
    available_label = get_available_label(unique_label);
    
    
}


/****/

void class_HDTM::clear_node(Node* root)
{
    if (root)
    {
        clear_node(root->left);
        clear_node(root->right);
        delete root;
    }
}

 

/***********************************************************************************************************************/
//functions to compute the ML

/****/

double class_HDTM::compute_node_log_ml(Col<unsigned int> Y_l, Col<unsigned int> Y_r, double theta_0, vec tau_vec, vec nu_vec, vec theta_vec)
{
    int n_grid = theta_vec.n_elem;
    double res = log(0.0);
    
    for (int t = 0; t < n_grid; ++t)
    {
        res = log_exp_x_plus_exp_y(res, eval_log_int_g(Y_l, Y_r, theta_vec(t), tau_vec) + eval_log_int_h(theta_vec(t), nu_vec, theta_0) - log(n_grid));
    }
   
    return res;
}


/****/

Col<unsigned int> class_HDTM::get_data_subset(Col<unsigned int> obs, Node* root)
{
    int n_sub = obs.n_elem;
    Col<unsigned int> y = root->data;
    Col<unsigned int> y_sub(n_sub);
    
    for (int i = 0; i < n_sub; ++i)
    {
        y_sub(i) = y(obs(i));
    }
    return y_sub;
}


/****/

Node* class_HDTM::compute_subtree_log_ml(Node* root, Mat<unsigned int> Y, Col<unsigned int> obs, bool select)
{
    if (root->left == nullptr)
    {
        root->log_ml = 0;
        return root;
    }
    else
    {
        Col<unsigned int> Y_l = get_data_subset(obs, root->left);
        Col<unsigned int> Y_r = get_data_subset(obs, root->right);
        
        compute_subtree_log_ml(root->left, Y, obs, select);
        compute_subtree_log_ml(root->right, Y, obs, select);
        
        double res = 0;
        
        if (select)
        {
            if (root->gamma)
            //if (root->key == 8)
            {
                res = compute_node_log_ml(Y_l, Y_r, root->theta_0, tau_vec, nu_vec, theta_vec);
            }
        }
        else
        {
            res = compute_node_log_ml(Y_l, Y_r, root->theta_0, tau_vec, nu_vec, theta_vec);
        }
        
        root->log_ml = res + (root->left)->log_ml + (root->right)->log_ml;
        //std::cout << root->key << ":" << root->log_ml << " /";
        return root;
    }
}


/****/

double class_HDTM::compute_log_ml(Col<unsigned int> obs, bool select)
{
    double log_ml_tree = 0;
    compute_subtree_log_ml(root, Y, obs, select);
    log_ml_tree = root->log_ml;
    
    return log_ml_tree;
}



/***********************************************************************************************************************/
//functions to sample cluster label

/****/
//transform the cluster label to a map in which the key-value pair saves the obs for each unique value.

map<unsigned int, vector<unsigned int> > class_HDTM::init_label(Col<unsigned int> c)
{
    map<unsigned int, vector<unsigned int> > cluster_label;
    
    for (int i = 0; i < n; ++i)
    {
        cluster_label[c(i)].push_back(i);
    }
    return cluster_label;
}


/****/
//use a set to save the keys of the unique clusters.

set<unsigned int> class_HDTM::get_unique_label(map<unsigned int, vector<unsigned int> > cluster_label)
{
    auto label_it = cluster_label.cbegin();
    set<unsigned int> unique_label;
    
    while (label_it != cluster_label.cend())
    {
        unique_label.insert(label_it->first);
        ++label_it;
    }
    return unique_label;
}


/****/

vector<unsigned int> class_HDTM::get_available_label(set<unsigned int> unique_label)
{
    vector<unsigned int> all_label;
    for (int i = 1; i <= n; ++i)
    {
        all_label.push_back(i);
    }
    for (auto set_it = unique_label.begin(); set_it != unique_label.end(); ++set_it)
    {
        auto vit = find(all_label.begin(), all_label.end(), *set_it);
        all_label.erase(vit);
    }
    return all_label;
}


/****/
//implementing Neal's algorithm 3. One iteration at a time.

void class_HDTM::sample(Col<unsigned int> &c, map<unsigned int, vector<unsigned int> > &cluster_label, set<unsigned int> &unique_label,
                        vector<unsigned int> &available_label)
{
    for (int i = 0; i < c.n_elem; ++i)
    {
        if (cluster_label[c(i)].size() > 1)  //i share cluster with some other obs
        {
            auto it = find(cluster_label[c(i)].begin(), cluster_label[c(i)].end(), i);
            cluster_label[c(i)].erase(it);
            
            vector<double> prob;
    
            for (auto set_it = unique_label.begin(); set_it != unique_label.end(); ++set_it)
            {
                Col<unsigned int> obs_de(cluster_label[*set_it]);
                Col<unsigned int> obs_nu(obs_de.n_elem + 1);
                for (int j = 0; j < obs_de.n_elem; j++)
                {
                    obs_nu(j) = obs_de(j);
                }
                
                obs_nu(obs_de.n_elem) = i;
                double ml_de = compute_log_ml(obs_de, select);
                double ml_nu = compute_log_ml(obs_nu, select);
                
                prob.push_back(exp(log(cluster_label[*set_it].size()) + ml_nu - ml_de));
            }

            Col<unsigned int> obs_i(1);
            obs_i(0) = i;
            prob.push_back(exp(log(alpha) + compute_log_ml(obs_i, select)));
            
            unsigned int  offset = rmultinom(prob);
            
            if (offset < unique_label.size())  //assign i to one of the existing clusters
            {
                c(i) = *next(unique_label.begin(), offset);
                cluster_label[c(i)].push_back(i);
            }
            else  //a new cluster is created
            {
                auto avsize = available_label.size();
                c(i) = available_label[avsize - 1];
                available_label.pop_back();
     
                unique_label.insert(c(i));
                cluster_label[c(i)].push_back(i);
            
            }
        }
        else  //i has its own cluster
        {
            available_label.push_back(c(i)); //first make the label available again
            auto eit = find(unique_label.begin(), unique_label.end(), c(i));
            unique_label.erase(eit);
            
            auto it = cluster_label.find(c(i));
            cluster_label.erase(it);
            
            vector<double> prob;
            
            for (auto set_it = unique_label.begin(); set_it != unique_label.end(); ++set_it)
            {
                Col<unsigned int> obs_de(cluster_label[*set_it]);
                Col<unsigned int> obs_nu(obs_de.n_elem + 1);
                for (int j = 0; j < obs_de.n_elem; j++)
                {
                    obs_nu(j) = obs_de(j);
                }
                obs_nu(obs_de.n_elem) = i;
                double ml_de = compute_log_ml(obs_de, select);
                double ml_nu = compute_log_ml(obs_nu, select);
                
                prob.push_back(exp(log(cluster_label[*set_it].size()) + ml_nu - ml_de));
            }
                                    
            Col<unsigned int> obs_i(1);
            obs_i(0) = i;
            prob.push_back(exp(log(alpha) + compute_log_ml(obs_i, select)));
            
            unsigned int  offset = rmultinom(prob);
            
            if (offset < unique_label.size())
            {
                c(i) = *next(unique_label.begin(), offset);
               
                cluster_label[c(i)].push_back(i);
            }
            else
            {
                auto avsize = available_label.size();
                c(i) = available_label[avsize - 1];
                available_label.pop_back();

                unique_label.insert(c(i));
                cluster_label[c(i)].push_back(i);
            }
        }
    }
}



/***********************************************************************************************************************/
//functions to sample gamma for node selection

/****/

Node* class_HDTM::compute_lambda_L1(map<unsigned int, vector<unsigned int> > &cluster_label)
{
    reset_L1(root); //reset L1 to zero before each iteration
    for (auto const &label : cluster_label)
    {
        vector<unsigned int> id = label.second;
        unsigned int l = id.size();
        Col<unsigned int> obs(l);
        for (auto i = 0; i < l; i++)
        {
            obs(i) = id[i];
        }
        compute_subtree_lambda_L1(root, obs);
    }
    return root;
}


/****/

Node* class_HDTM::reset_L1(Node* root)
{
    if (root->left == nullptr)
    {
        return root;
    }
    else
    {
        reset_L1(root->left);
        reset_L1(root->right);
        root->log_L1 = 0;
    }
    return root;
}


/****/

Node* class_HDTM::compute_subtree_lambda_L1(Node* root, Col<unsigned int> obs)
{
    if (root->left == nullptr)
    {
        return root;
    }
    else
    {
        Col<unsigned int> Y_l = get_data_subset(obs, root->left);
        Col<unsigned int> Y_r = get_data_subset(obs, root->right);
        
        compute_subtree_lambda_L1(root->left, obs);
        compute_subtree_lambda_L1(root->right, obs);
        
        root->log_L1 += compute_node_log_ml(Y_l, Y_r, root->theta_0, tau_vec, nu_vec, theta_vec);
        return root;
    }
}


/****/

Node* class_HDTM::compute_lambda_L0()
{
    compute_subtree_lambda_L0(root);
    return root;
}


/****/

Node* class_HDTM::compute_subtree_lambda_L0(Node* root)
{
    if (root->left == nullptr)
    {
        return root;
    }
    else
    {
        Col<unsigned int> Y_l = (root->left)->data;
        Col<unsigned int> Y_r = (root->right)->data;
        
        compute_subtree_lambda_L0(root->left);
        compute_subtree_lambda_L0(root->right);
        root->log_L0 = compute_node_log_ml(Y_l, Y_r, root->theta_0, tau_vec, nu_vec, theta_vec);
        return root;
    }
}


/****/

void class_HDTM::sample_gamma(map<unsigned int, vector<unsigned int> > &cluster_label, Col<unsigned int> &gamma_sample)
{
    compute_lambda_L1(cluster_label);
    sample_subtree_gamma(root, gamma_sample);
    return;
}


/****/

Node* class_HDTM::sample_subtree_gamma(Node* root, Col<unsigned int> &gamma_sample)
{
    if (root->left == nullptr)
    {
        return root;
    }
    else
    {
        sample_subtree_gamma(root->left, gamma_sample);
        sample_subtree_gamma(root->right, gamma_sample);
        
        double M10 = exp(root->log_L1 - root->log_L0);
        double prob = lambda_global * M10/(1 - lambda_global + lambda_global * M10);  //might add prior information
        
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed);
        std::binomial_distribution<int> distribution(1, prob);
        
        root->gamma = distribution(generator);
        gamma_sample(root->key - p - 1) = root->gamma;
        
        std::cout << root->key << " " << root->log_L1 << " " << root->log_L0 << " " << prob << " " << root->gamma << std::endl;
        
        return root;
    }
}



/***********************************************************************************************************************/
//functions to sample DP precision

/****/

void class_HDTM::sample_precision(double &alpha, unsigned int num_cluster, vector<double> precision_grid)
{
    vector<double> prob;
    
    for (auto val : precision_grid)
    {
        double temp = val/(1 - val);
        double log_prob = num_cluster * log(temp) + lgamma(temp) - lgamma(temp + n);
        prob.push_back(log_prob);
    }
    int n = prob.size();
    double last = prob[n - 1];
    
    for (auto it = prob.begin(); it != prob.end(); ++it)
    {
        *it -= last;
        *it = exp(*it);
    }
    
    unsigned int  offset = rmultinom(prob);
    double t = precision_grid[offset];
    alpha = t/(1 - t);
    return;
}


/****/

void class_HDTM::get_precision_grid(vector<double> &precision_grid, double start, double step)
{
    double value = start;
    while (value < 1)
    {
        precision_grid.push_back(value);
        value += step;
    }
    return;
}


/***********************************************************************************************************************/
//functions to sample the global base selection probabiliy lambda

void class_HDTM::sample_lambda_global(double &lambda_global, unsigned int num_nodes)
{
    double gam_1 = 0;
    double gam_2 = 0;
    
    gam_1 = randg<double>(distr_param(1 + num_nodes, 1));
    gam_2 = randg<double>(distr_param(1 + p - 1 - num_nodes, 1));
    
    lambda_global = gam_1/(gam_1 + gam_2);
    return;
}




/***********************************************************************************************************************/
//Gibbs sampler
/****/

vector<Mat<unsigned int> > class_HDTM::sample_mcmc(unsigned int mcmc_iter, vector<double> &post_sample_alpha, vector<double> &post_sample_lambda_global)
{
    
    //cout << "0" << "|------------------------------|" << "100%" << endl;
    //cout << " |" ;
    
    //int r = mcmc_iter/30;
    vector<Mat<unsigned int> > post_sample;
    Mat<unsigned int> post_sample_c(n, mcmc_iter);
    Mat<unsigned int> post_sample_gamma(p-1, mcmc_iter);
    
    get_precision_grid(precision_grid, 0.02, 0.05);
    
    for (int iter = 0; iter < mcmc_iter; ++iter)
    {
        std::cout << "iter = " << iter << std::endl;
        
        post_sample_c.col(iter) = c;
        post_sample_gamma.col(iter) = gamma_sample;
        
        if (select)
        {
            sample_gamma(cluster_label, gamma_sample);
        }
        
        sample(c, cluster_label, unique_label, available_label);
        unsigned int K = unique_label.size();
        
        
        sample_precision(alpha, K, precision_grid);
        
        unsigned int num_nodes = sum(gamma_sample);
        sample_lambda_global(lambda_global, num_nodes);
        
        //std::cout << "alpha = " << alpha << std::endl;
        
        post_sample_alpha.push_back(alpha);
        post_sample_lambda_global.push_back(lambda_global);
        /*if (iter % r == 0) {
         cout << "*" ;
         } */
        
        //cout << iter << endl;
    }
    //cout << "|";
    post_sample.push_back(post_sample_c);
    post_sample.push_back(post_sample_gamma);
    //post_sample.push_back(post_sample_gamma);
    
    return post_sample;
}



/***********************************************************************************************************************/
//functions to summary cluster centroids

/****/

double class_HDTM::compute_node_log_exp(Col<unsigned int> Y_l, Col<unsigned int> Y_r, double theta_0, vec tau_vec, vec nu_vec, vec theta_vec, bool which)
{
    int n_grid = theta_vec.n_elem;
    double res = log(0.0);
    
    for (int t = 0; t < n_grid; ++t)
    {
        res = log_exp_x_plus_exp_y(res, eval_log_int_g_exp(Y_l, Y_r, theta_vec(t), tau_vec, which) + eval_log_int_h(theta_vec(t), nu_vec, theta_0) - log(n_grid));
        
        //std::cout <<eval_log_int_g_exp(Y_l, Y_r, theta_vec(t), tau_vec, which) << " " << eval_log_int_g(Y_l, Y_r, theta_vec(t), tau_vec) << std::endl;
    }
    
    return res;
}



/****/

Node* class_HDTM::compute_subtree_log_exp(Node* root, Mat<unsigned int> Y, Col<unsigned int> obs, bool which)
{
    if (root->left == nullptr)
    {
        root->log_theta = 0;
        return root;
    }
    else
    {
        compute_subtree_log_exp(root->left, Y, obs, select);
        compute_subtree_log_exp(root->right, Y, obs, select);
        
        double res = 0;
        
        if (root->gamma)
        {
            Col<unsigned int> Y_l = get_data_subset(obs, root->left);
            Col<unsigned int> Y_r = get_data_subset(obs, root->right);
        
            double res1 = compute_node_log_exp(Y_l, Y_r, root->theta_0, tau_vec, nu_vec, theta_vec, which);
            double res2 = compute_node_log_ml(Y_l, Y_r, root->theta_0, tau_vec, nu_vec, theta_vec);
            res = res1 - res2;
            //std::cout << root->key << ":" << res1 << " " << res2 << std::endl;
        }
        else
        {
            Col<unsigned int> Y_l = (root->left)->data;
            Col<unsigned int> Y_r = (root->right)->data;
            double res1 = compute_node_log_exp(Y_l, Y_r, root->theta_0, tau_vec, nu_vec, theta_vec, which);
            double res2 = compute_node_log_ml(Y_l, Y_r, root->theta_0, tau_vec, nu_vec, theta_vec);
            res = res1 - res2;
            //std::cout << root->key << ":" << res1 << " " << res2 << std::endl;
        }
        
        
        root->log_theta = res;
        return root;
    }
}





/****/

void class_HDTM::compute_log_exp(Col<unsigned int> obs, bool which)
{
    compute_subtree_log_exp(root, Y, obs, which);
}


/****/

Node* class_HDTM::compute_posterior_centroid_subtree(Node* root, Col<unsigned int> obs, vector<double>& cluster_centroid)
{
    if (root->left == nullptr)
    {
        cluster_centroid.push_back(root->node_prob);
        return root;
    }
    else
    {
        (root->left)->node_prob = root->node_prob * exp(root->log_theta);
        (root->right)->node_prob = root->node_prob * (1 - exp(root->log_theta));
        compute_posterior_centroid_subtree(root->left, obs, cluster_centroid);
        compute_posterior_centroid_subtree(root->right, obs, cluster_centroid);
        //std::cout << root->log_theta << " " ;
    }
    return root;
}


/****/

void class_HDTM::compute_posterior_centroid(Col<unsigned int> obs, vector<double>& cluster_centroid)
{
    compute_posterior_centroid_subtree(root, obs, cluster_centroid);
}


/****/

vector<vector<double> >  class_HDTM::compute_centroid( )
{
    vector<vector<double> > centroid;
    for (auto set_it = unique_label.begin(); set_it != unique_label.end(); ++set_it)
    {
        vector<double> cluster_centroid;
        Col<unsigned int> obs(cluster_label[*set_it]);
        compute_log_exp(obs, 1);
        compute_posterior_centroid(obs, cluster_centroid);
        centroid.push_back(cluster_centroid);
    }
    return centroid;
}




/****/

Node* class_HDTM::compute_posterior_diffdisp_subtree(Node* root, Col<unsigned int> obs, vector<double>& cluster_diff_disp)
{
    if (root->left == nullptr)
    {
        return root;
    }
    else
    {
        compute_posterior_diffdisp_subtree(root->left, obs, cluster_diff_disp);
        compute_posterior_diffdisp_subtree(root->right, obs, cluster_diff_disp);
        //root->node_prob = exp(root->log_theta) - exp((root->left)->log_theta) - exp((root->right)->log_theta);
        //root->node_prob = exp(root->log_theta);
        //std::cout << root->node_prob << " " ;
    }
    return root;
}


/****/

void class_HDTM::compute_posterior_diffdisp(Col<unsigned int> obs, vector<double>& cluster_diff_disp)
{
    compute_posterior_diffdisp_subtree(root, obs, cluster_diff_disp);
}



/****/

vector<vector<double> >  class_HDTM::compute_diff_disp( )
{
    vector<vector<double> > diff_disp;
    for (auto set_it = unique_label.begin(); set_it != unique_label.end(); ++set_it)
    {
        vector<double> cluster_diff_disp;
        Col<unsigned int> obs(cluster_label[*set_it]);
        compute_log_exp(obs, 0);
        compute_posterior_diffdisp(obs, cluster_diff_disp);
        diff_disp.push_back(cluster_diff_disp);
    }
    return diff_disp;
}





/***********************************************************************************************************************/
//miscellaneous functions
/****/

void class_HDTM::traverse(Node* root, int type)
{
    if (root == nullptr)
    {
        return ;
    }
    traverse(root->left, type);
    traverse(root->right, type);
    if(type == 1){
        std::cout << root->key << "  ";
    }
    else if(type == 2){ cout << root->key <<":" << root->gamma << "  ";
    }
   
}


/****/

Node* class_HDTM::get_transformed_data(Node* root, mat &transformed_data)
//Node* class_HDTM::get_transformed_data(Node* root)
{
    
    if (root->left == nullptr)
    {
        return root;
    }
    else
    {
        get_transformed_data(root->left, transformed_data);
        get_transformed_data(root->right, transformed_data);
        Col<unsigned int> v_num = (root->left)->data;
        Col<unsigned int> v_den = root->data;
        
        for (int i = 0; i < n; ++i)
        {
            transformed_data(i, root->key - p - 1) = (double)v_num(i) / v_den(i);
            //std::cout << transformed_data(i, root->key - p - 1) << std::endl;
        }
        return root;
    }
}









