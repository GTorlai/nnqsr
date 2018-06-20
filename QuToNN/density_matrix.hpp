#ifndef QST_DENSITYMATRIX_HPP
#define QST_DENSITYMATRIX_HPP

#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <vector>
#include <iomanip>
#include <fstream>
#include <bitset>
#include <complex>


namespace qst{

class DensityMatrix{


    int N_;            // Number of degrees of freedom (visible units)
    int na_;           // Number of purification units
    int npar_;         // Number of parameters
    int nparLambda_;   // Number of amplitude parameters
    int nparMu_;       // Number of phase parameters
    int nparSys_;      // Number of system parameters
    int nparEnv_;      // Number of environment parameters
    //int nh_;    
    Rbm rbmSysAm_;        // RBM for the system amplitude
    Rbm rbmSysPh_;        // RBM for the system phases
    Rbm rbmEnvAm_;        // RBM for the environment amplitude
    Rbm rbmEnvPh_;        // RBM for the environment phases
    
    const std::complex<double> I_; // Imaginary unit
    Eigen::VectorXcd gamma_complex_;
    //Random number generator 
    std::mt19937 rgen_;

public:
    
    DensityMatrix(Parameters &par):rbmSysAm_(par),
                                   rbmSysPh_(par),
                                   rbmEnvAm_(par),
                                   rbmEnvPh_(par),
                                   I_(0,1){
        nparLambda_ = rbmSysAm_.Npar() + rbmEnvAm_.Npar();
        nparMu_ = rbmSysPh_.Npar() + rbmEnvPh_.Npar();
        nparSys_ = rbmSysAm_.Npar() + rbmSysPh_.Npar();
        nparEnv_ = rbmEnvAm_.Npar() + rbmEnvPh_.Npar();
        npar_ = nparLambda_ + nparMu_;
        N_ = rbmSysAm_.Nvisible(); 
        na_ = rbmEnvAm_.Nhidden();
        std::random_device rd;
        gamma_complex_.resize(na_);
        //rgen_.seed(rd());
        rgen_.seed(13579);
    }
    // Private members access functions
    inline int N()const{
        return N_;
    }
    inline int Npar()const{
        return npar_;
    }
    inline int NparLambda()const{
        return nparLambda_;
    }
    inline int Nchains(){
        return rbmSysAm_.Nchains();
    }
    inline Eigen::VectorXd VisibleStateRow(int s){
        return rbmSysAm_.VisibleStateRow(s);
    }
    // Set the state of the wavefunction's degrees of freedom
    inline void SetVisibleLayer(Eigen::MatrixXd v){
        rbmSysAm_.SetVisibleLayer(v);
    }
    // Initialize the wavefunction parameters    
    void InitRandomPars(int seed,double sigma){
        rbmSysAm_.InitRandomPars(seed,sigma);
        rbmSysPh_.InitRandomPars(seed,sigma);
        rbmEnvAm_.InitRandomPars(seed,sigma);
        rbmEnvPh_.InitRandomPars(seed,sigma);
    }

    std::complex<double> rho(const Eigen::VectorXd & v,const Eigen::VectorXd & vp){

        std::complex<double> logrho = 0.0;
        logrho -= rbmSysAm_.EffectiveEnergy(v)-rbmSysAm_.EffectiveEnergy(vp);
        logrho -= I_*(rbmSysAm_.EffectiveEnergy(v)+rbmSysAm_.EffectiveEnergy(vp));
        ln1pexpComplex(0.5*rbmEnvAm_.Weights()*(v+vp)+0.5*rbmEnvPh_.Weights()*(v-vp)+rbmEnvAm_.HiddenBias(),gamma_complex_);
        logrho += gamma_complex_.sum();
        return std::exp(logrho);

    }

    //---- SAMPLING ----//
    void Sample(int steps){
        //rbmSysAm_.Sample(steps);
    }

    //---- DERIVATIVES ----//

    // Lambda gradient of the purification term
    Eigen::VectorXcd PurificationLambdaGrad(const Eigen::VectorXd &v,const Eigen::VectorXd &vp){
        Eigen::VectorXcd der(nparEnv_/2);
        der.setZero();
        logisticComplex(0.5*rbmEnvAm_.Weights()*(v+vp)+0.5*rbmEnvPh_.Weights()*(v-vp)+rbmEnvAm_.HiddenBias(),gamma_complex_);
        int p=0;
        for(int k=0;k<na_;k++){
            for(int j=0;j<N_;j++){
                der(p)=gamma_complex_(k)*(v(j)+vp(j));
                p++;
            }
        }
        for(int j=0;j<N_;j++){
            p++;
        }
        for(int k=0;k<na_;k++){
            der(p) = 2.0*gamma_complex_(k);
            p++;
        }
        return -0.5*der;
    }
    // Mu gradient of the purification term
    Eigen::VectorXcd PurificationMuGrad(const Eigen::VectorXd &v,const Eigen::VectorXd &vp){
        Eigen::VectorXcd der(nparEnv_);
        der.setZero();
        logisticComplex(0.5*rbmEnvAm_.Weights()*(v+vp)+0.5*rbmEnvPh_.Weights()*(v-vp)+rbmEnvAm_.HiddenBias(),gamma_complex_);
        int p=0;
        for(int k=0;k<na_;k++){
            for(int j=0;j<N_;j++){
                der(p)=I_*gamma_complex_(k)*(v(j)-vp(j));
                p++;
            }
        }
        return -0.5*der;
    }
    // Full Lambda Gradient
    Eigen::VectorXcd LambdaGrad(const Eigen::VectorXd &v,const Eigen::VectorXd &vp){
        Eigen::VectorXcd der(nparLambda_);
        der.head(int(nparSys_/2)) = 0.5*(rbmSysAm_.VisEnergyGrad(v)+rbmSysAm_.VisEnergyGrad(vp));
        der.tail(int(nparEnv_/2)) = PurificationLambdaGrad(v,vp);
        return der;
    } 
    // Full Mu Gradient
    Eigen::VectorXcd MuGrad(const Eigen::VectorXd &v,const Eigen::VectorXd &vp){
        Eigen::VectorXcd der(nparMu_);
        der.head(int(nparSys_/2)) = 0.5*I_*(rbmSysPh_.VisEnergyGrad(v)-rbmSysPh_.VisEnergyGrad(vp));
        der.tail(int(nparEnv_/2)) = PurificationMuGrad(v,vp);

        return der;
    } 
    // Full Gradient
    Eigen::VectorXcd Grad(const Eigen::VectorXd &v,const Eigen::VectorXd &vp){
    
        Eigen::VectorXcd der(npar_);
        der << LambdaGrad(v,vp),MuGrad(v,vp);
        return der;
    }
   
    //Compute the gradient of the effective energy in an arbitrary basis given by U
    void rotatedGrad(const std::vector<std::string> & basis,
                            const Eigen::VectorXd & state,
                            std::map<std::string,Eigen::MatrixXcd> & Unitaries,
                            Eigen::VectorXd &rotated_gradient )
    {
 
        int t=0,counter=0,counter_p=0;
        std::complex<double> U=1.0;
        std::complex<double> rotated_rho = 0.0;
        std::bitset<16> bit;
        std::bitset<16> st;
        std::bitset<16> bit_p;
        std::bitset<16> st_p;
        std::vector<int> basisIndex;
        Eigen::VectorXd v(N_);
        Eigen::VectorXd vp(N_);
        Eigen::VectorXcd rotated_rhoGrad(npar_);
        rotated_rhoGrad.setZero();
        basisIndex.clear();

        // Extract the sites where the rotation is non-trivial
        for(int j=0;j<N_;j++){
            if (basis[j]!="Z"){
                t++;
                basisIndex.push_back(j);
            }
        }

        // Loop over the states of the local Hilbert space
        for(int i=0;i<1<<t;i++){
            counter = 0;
            bit = i;
            v = state;
            for(int j=0;j<N_;j++){
                if (basis[j] != "Z"){
                    v(j) = bit[counter];
                    counter++;
                }
            }
            for(int ip=0;ip<1<<t;ip++){
                counter_p =0;
                bit_p = ip;
                vp=state;
                for(int j=0;j<N_;j++){
                    if (basis[j] != "Z"){
                        vp(j) = bit_p[counter_p];
                        counter_p++;
                    }
                }
                U=1.0;
                for(int ii=0;ii<t;ii++){
                    U = U * Unitaries[basis[basisIndex[ii]]](int(state(basisIndex[ii])),int(v(basisIndex[ii])));
                    U = U * conj(Unitaries[basis[basisIndex[ii]]](int(state(basisIndex[ii])),int(vp(basisIndex[ii]))));
                }
                rotated_rhoGrad += U*Grad(v,vp)*rho(v,vp);
                rotated_rho += U*rho(v,vp);
            }
        }
        rotated_gradient = (rotated_rhoGrad/rotated_rho).real();
    }
  
    //---- UTILITIES ----//

    //Get RBM parameters
    Eigen::VectorXd GetParameters(){
        Eigen::VectorXd pars(npar_);
        pars<<rbmSysAm_.GetParameters(),rbmEnvAm_.GetParameters(),rbmSysPh_.GetParameters(),rbmEnvPh_.GetParameters();
        return pars;
    }
    // Set RBM parameters
    void SetParameters(const Eigen::VectorXd & pars){
        rbmSysAm_.SetParameters(pars.head(int(nparSys_/2)));
        rbmEnvAm_.SetParameters(pars.segment(int(nparSys_/2),nparLambda_));
        rbmSysPh_.SetParameters(pars.segment(nparLambda_,nparLambda_+int(nparSys_/2)));
        rbmEnvPh_.SetParameters(pars.tail(int(nparEnv_/2)));
    }
    // Complex Algebra Functions
    inline std::complex<double> ln1pexpComplex(std::complex<double> x)const{
        if(abs(x)>30){
            return x;
        }
        return std::log(1.+std::exp(x));
    }
    void ln1pexpComplex(const Eigen::VectorXcd & x,Eigen::VectorXcd & y){
        for(int i=0;i<x.size();i++){
            y(i)=ln1pexpComplex(x(i));
        }
    }
    inline std::complex<double> logisticComplex(std::complex<double> x)const{
        return 1./(1.+std::exp(-x));
    }
    void logisticComplex(const Eigen::VectorXcd & x,Eigen::VectorXcd & y){
        for(int i=0;i<x.size();i++){
            y(i)=logisticComplex(x(i));
        }
    }
};
}

#endif
