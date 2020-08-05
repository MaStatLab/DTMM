

library(ape)

################################################################################################

DTMM <- function(Y,
                 tree,
                 tau_vec = 10 ^ seq(-1, 4, 0.5),
                 nu_vec = 1,
                 theta_vec = seq(0.01, 0.99, 0.08),
                 init_c = "default",
                 init_gamma = "default",
                 alpha = "default",
                 mcmc_iter = 10,
                 select = TRUE){

  if(init_c == "default"){
    sum_counts = sweep(Y, 1, apply(Y, 1, sum), "/")
    cc = kmeans(Y, 3)
    c = cc$cluster
  }else{
    c = init_c
  }

  if(init_gamma == "default"){
    p = dim(Y)[2]
    gamma_sample = rbinom(p - 1, 1, 0.5)
  }else{
    gamma_sample = init_gamma
  }

  if(alpha == "default"){
    temp = runif(1)
    alpha = temp/(1 - temp)
  }

  tree <- reorder(tree, order = "postorder")
  edge <- apply(tree$edge, 2, rev)

  DTMMcpp(Y, edge, tau_vec, nu_vec, theta_vec,
          c, gamma_sample, alpha, mcmc_iter, select)
}

################################################################################################

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

  DTMMcpp_centroid(Y, edge, tau_vec, nu_vec, theta_vec, c, gamma)

}

