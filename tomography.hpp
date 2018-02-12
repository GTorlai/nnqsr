#include <iostream>
#include <Eigen/Dense>
#include <iomanip>
#include <fstream>
#ifndef QST_TOMOGRAPHY_HPP
#define QST_TOMOGRAPHY_HPP

namespace qst{

// Quantum State Tomography Class
class Tomography {

    Rbm & rbm_;                 // Rbm

    int npar_;                  // Number of variational parameters
    int bs_;                    // Batch size
    int cd_;                    // Number of Gibbs steps in contrastive divergence
    int epochs_;                // Number of training iterations
    double lr_;                 // Learning rate
    double l2_;                 // L2 regularization constant

    Eigen::VectorXd grad_;      // Gradients
    
    std::mt19937 rgen_;         // Random number generator

public:
    // Contructor 
    Tomography(Rbm &rbm,Parameters &par): rbm_(rbm){
        npar_=rbm_.Npar();
        bs_ = par.bs_;
        cd_ = par.cd_;
        lr_ = par.lr_;
        l2_ = par.l2_;
        epochs_ = par.ep_;
        grad_.resize(npar_);
    }

    // Compute gradient of Negative Log-Likelihood on a batch 
    void Gradient(const Eigen::MatrixXd &batch){ 
        grad_.setZero();
        
        // Positive Phase driven by the data
        for(int s=0;s<bs_;s++){
            grad_ -= rbm_.DerLog(batch.row(s))/double(bs_);
        }
        // Negative Phase driven by the model
        rbm_.Sample(cd_);
        for(int s=0;s<rbm_.Nchains();s++){
            grad_ += rbm_.DerLog(rbm_.VisibleStateRow(s))/double(rbm_.Nchains());
        }
    }

    // Update rbm parameters
    void UpdateParameters(){
        auto pars=rbm_.GetParameters();
        for(int i=0;i<npar_;i++){
            pars(i) -= lr_*(grad_(i)+l2_*pars(i));
        }
        rbm_.SetParameters(pars);
    }

    // Run the tomography
    void Run(Eigen::MatrixXd &trainSet){
        // Initialization
        int index;
        Eigen::MatrixXd batch_bin;
        std::uniform_int_distribution<int> distribution(0,trainSet.rows()-1);
        
        // Training
        for(int i=0;i<epochs_;i++){
            std::cout<<"Epoch "<<i<<std::endl;
            // Initialize the visible layer to random data samples
            batch_bin.resize(rbm_.Nchains(),rbm_.Nvisible());
            for(int k=0;k<rbm_.Nchains();k++){
                index = distribution(rgen_);
                batch_bin.row(k) = trainSet.row(index);
            }
            rbm_.SetVisibleLayer(batch_bin);
            
            // Build the batch of data at random
            batch_bin.resize(bs_,rbm_.Nvisible()); 
            for(int k=0;k<bs_;k++){
                index = distribution(rgen_);
                batch_bin.row(k) = trainSet.row(index);
            }
            
            // Perform one step of optimization
            Gradient(batch_bin);
            UpdateParameters();
        }
    }
};
}

#endif
