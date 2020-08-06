
#ifndef HDTM_hpp
#define HDTM_hpp

using namespace Rcpp;
using namespace arma;
using namespace std;

class class_HDTM
{

public:

    //constructor
    class_HDTM(Mat<unsigned int> Y, Mat<unsigned int> edge, vec tau_vec, vec nu_vec, vec theta_vec, Col<unsigned int> c, Col<unsigned int> gamma_sample, double alpha, bool select);

    //destructor
    ~class_HDTM();

    void traverse(Node* root, int type);
    mat transformed_data;
    Node* get_transformed_data(Node* root, mat &transformed_data);
    //Node* get_transformed_data(Node* root);


    double compute_log_ml(Col<unsigned int> obs, bool select);

    double lambda_global = 0.5;
    Mat<unsigned int>  post_sample;
    vector<double> post_sample_alpha;
    vector<double> post_sample_lambda_global;

    void sample(Col<unsigned int> &c, map<unsigned int, vector<unsigned int> > &cluster_label, set<unsigned int> &unique_label, vector<unsigned int> &available_label);
    //Mat<unsigned int>  sample_mcmc(unsigned int mcmc_iter);

    vector<Mat<unsigned int> > sample_mcmc(unsigned int mcmc_iter, vector<double> &post_sample_alpha, vector<double> &post_sample_lambda_global);

    //functions to summary cluster centroids
    vector<vector<double> > compute_centroid();
    vector<vector<double> > compute_diff_disp();

private:
    Mat<unsigned int> Y;  //observations: the OTU counts
    Mat<unsigned int> edge;   //the edge matrix
    vec tau_vec;
    vec nu_vec;
    vec theta_vec;
    vector<double> precision_grid;

    int n;   //num of observations
    int p;   //num of OTUs
    Node* root;   //pointer to the root of the tree

    Col<unsigned int> c;   ///cluster indicator vector for the n observations
    Col<unsigned int> gamma_sample;   ///coupling indicator vector for the (p-1) nodes
    double alpha;
    bool select; //whether to perform global node select

    //functions to initiate the tree
    Node* get_new_node(unsigned int key);  //create a new node
    map<unsigned int, vector<unsigned int> > get_edge_map(Mat<unsigned int> edge);  //transform the edge matrix to a map used to create the tree
    Node* construct_subtree(Node* root, map<unsigned int, vector<unsigned int> > edge_map);  //construct the tree recursively
    Node* aggregate_data(Node* root, Mat<unsigned int> Y);  //add data to the tree
    Node* get_theta_0(Node* root);
    Node* get_gamma(Node* root);

    void init();
    void clear_node(Node* root);

    //functions to compute the ML
    Col<unsigned int> get_data_subset(Col<unsigned int> obs, Node* root);
    double compute_node_log_ml(Col<unsigned int> Y_l, Col<unsigned int> Y_r, double theta_0, vec tau_vec, vec nu_vec, vec theta_vec);
    Node* compute_subtree_log_ml(Node* root, Mat<unsigned int> Y, Col<unsigned int> obs, bool select);

    //functions to sample gamma
    Node* reset_L1(Node* root);
    Node* compute_lambda_L1(map<unsigned int, vector<unsigned int> > &cluster_label);
    Node* compute_subtree_lambda_L1(Node* root, Col<unsigned int> obs);
    Node* compute_lambda_L0();
    Node* compute_subtree_lambda_L0(Node* root);

    void sample_gamma(map<unsigned int, vector<unsigned int> > &cluster_label, Col<unsigned int> &gamma_sample);
    Node* sample_subtree_gamma(Node* root, Col<unsigned int> &gamma_sample);

    //functions to sample DP precision
    void sample_precision(double &alpha, unsigned int num_cluster, vector<double> precision_grid);
    void get_precision_grid(vector<double> &precision_grid, double start, double step);

    //functions to sample gamma
    void sample_lambda_global(double &lambda_global, unsigned int num_nodes);

    //functions to sample clustering labels
    map<unsigned int, vector<unsigned int> > cluster_label;
    set<unsigned int> unique_label;
    vector<unsigned int> available_label;

    map<unsigned int, vector<unsigned int> > init_label(Col<unsigned> c);
    set<unsigned int> get_unique_label(map<unsigned int, vector<unsigned int> > cluster_label);
    vector<unsigned int> get_available_label(set<unsigned int> unique_label);

    //functions to summary cluster centroids

    double compute_node_log_exp(Col<unsigned int> Y_l, Col<unsigned int> Y_r, double theta_0, vec tau_vec, vec nu_vec, vec theta_vec, bool which);
    Node* compute_subtree_log_exp(Node* root, Mat<unsigned int> Y, Col<unsigned int> obs, bool which);
    void compute_log_exp(Col<unsigned int> obs, bool which);
    Node* compute_posterior_centroid_subtree(Node* root, Col<unsigned int> obs, vector<double>& cluster_centroid);
    void compute_posterior_centroid(Col<unsigned int> obs, vector<double>& cluster_centroid);

    Node* compute_posterior_diffdisp_subtree(Node* root, Col<unsigned int> obs, vector<double>& cluster_diff_disp);
    void compute_posterior_diffdisp(Col<unsigned int> obs, vector<double>& cluster_diff_disp);


};


#endif /* HDTM_hpp */














