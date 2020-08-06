library(ape)

################################################################################################

#' @title Dirichlet-tree Multinomial Mixtures
#' @description Fit the Dirichlet-tree Multinomial Mixtures for clustering microbiome compositions.
#' @param Y a \eqn{n * K} matrix containing the OTU/ASV counts of each sample. Each row represents a sample, each column represents an OTU/ASV.
#' @param tree an object of class \code{"phylo"} in the package \code{"ape"}. A phylogenetic tree of the OTUs/ASVs.
#' @param tau_vec a vector containing the grid of values to numerically integrate out \eqn{\tau}, the cluster-specific dispersion parameter at each node. The values are ordered.
#' @param nu_vec a vector containing the grid of values to numerically integrate out \eqn{\nu}, the base dispersion parameter at each node. The values are ordered.
#' @param theta_vec a vector containing the grid of values to numerically integrate out \eqn{\theta}, the mean parameter at each node. The values are ordered and are between 0 and 1.
#' @param init_c an optional vector of positive integers of length n specifying the initial clustering labels of the samples. If not provided, the labels are initiated by running the k-means algorithm with k = 3.
#' @param beta an optional positive value specifying the initial value for the concentration parameter of the Dirichlet process. If not provided, the concentration parameter is initialized from the prior in the paper.
#' @param mcmc_iter a positive integer indicating the number of iterations to run the Gibbs sampler.
#' @param select logical. If TRUE fit DTMM with the node selection procedure. If FALSE all nodes are active in the clustering process.
#' @return \code{DTMM} returns a list containing the following components:
#'  \itemize{
#'   \item \code{post_c} - a \eqn{n * T} matrix containing the posterior samples of the clustering indicators, where \eqn{T} is the number of iterations. The values in the matrix only have nominal meanings.
#'   \item \code{post_gamma} - a \eqn{n * T} matrix containing the posterior samples of the node selection indicators. The rows are ordered based on the labels of the nodes in the edge matrix of the phylogenetic tree object.
#'   \item \code{post_beta} - a vector of length \eqn{T} containing the posterior samples of the Dirichlet process concentration parameter.
#'   \item \code{post_lambda} - a vector of length \eqn{T} containing the posterior samples of the prior node selection probability.
#' }
#'
#'@examples
#'
#'#--load the illustating dataset
#'data(example)
#'#--the dataset contains an OTU table, a phylogenetic tree, and the result of fitting DTMM (500 iterations).
#'str(example)
#'
#'#--fit DTMM with the default settings.
#'result <- DTMM(example$Y, example$tree, mcmc_iter = 500, select = TRUE)
#'#--get the clustering labels in the last iteration.
#'result$post_c[, 500]
#'
DTMM <- function(Y,
                 tree,
                 tau_vec = 10 ^ seq(-1, 4, 0.5),
                 nu_vec = 1,
                 theta_vec = seq(0.01, 0.99, 0.08),
                 init_c = "default",
                 beta = "default",
                 mcmc_iter = 2000,
                 select = TRUE){

  if(init_c == "default"){
    sum_counts = sweep(Y, 1, apply(Y, 1, sum), "/")
    cc = kmeans(Y, 3)
    c = cc$cluster
  }else{
    c = init_c
  }

  if(beta == "default"){
    temp = runif(1)
    alpha = temp/(1 - temp)
  }

  tree <- reorder(tree, order = "postorder")
  edge <- apply(tree$edge, 2, rev)
  p = dim(Y)[2]
  gamma_sample = rbinom(p - 1, 1, 0.5)

  HDTMcpp(Y, edge, tau_vec, nu_vec, theta_vec,
          c, gamma_sample, alpha, mcmc_iter, select)
}

################################################################################################

#' @title Getting the cluster centroids
#' @description Get the cluster centroids after fitting DTMM
#' @param Y a \eqn{n * K} matrix containing the OTU/ASV counts of each sample. Each row represents a sample, each column represents an OTU/ASV.
#' @param tree an object of class \code{"phylo"} in the package \code{"ape"}. A phylogenetic tree of the OTUs/ASVs.
#' @param tau_vec a vector containing the grid of values to numerically integrate out \eqn{\tau}, the cluster-specific dispersion parameter at each node. The values are ordered.
#' @param nu_vec a vector containing the grid of values to numerically integrate out \eqn{\nu}, the base dispersion parameter at each node. The values are ordered.
#' @param theta_vec a vector containing the grid of values to numerically integrate out \eqn{\theta}, the mean parameter at each node. The values are ordered and are between 0 and 1.
#' @param c a vector specifying the clustering labels of the samples that used to estimate the clustering centroids.
#' @param gamma a vector specifying the node selection indicators. gamma and c should be set to values in a specific iteration of the Gibbs sampler when fitting DTMM.
#' @return \code{DTMM_centroid} returns a list of length \eqn{K} containing the cluster centroids, where \eqn{K} is the number of distinct values in \eqn{c}.
#'

DTMM_centroid <- function(Y,
                          tree,
                          tau_vec = 10 ^ seq(-1, 4, 0.5),
                          nu_vec = 1,
                          theta_vec = seq(0.01, 0.99, 0.08),
                          c,
                          gamma
){

  tree <- reorder(tree, order = "postorder")
  edge <- apply(tree$edge, 2, rev)

  HDTMcpp_centroid(Y, edge, tau_vec, nu_vec, theta_vec, c, gamma)

}

