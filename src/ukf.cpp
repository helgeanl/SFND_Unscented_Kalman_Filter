#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = false;

    // If false, initialize state with measurement
    is_initialized_ = false;

    // Init dimensions
    n_x_ = 5;
    n_aug_ = n_x_ + 2;
    n_z_lidar_ = 2;
    n_z_radar_ = 3;
    lambda_ = 3 - n_aug_;

    // Initial state vector
    x_ = VectorXd(n_x_);
    x_.setZero();

    // Initial covariance matrix
    P_ = MatrixXd(n_x_, n_x_);
    P_.setIdentity();
    

    // Set weights
    weights_ = VectorXd(2*n_aug_+1);
    weights_.fill(0.5* 1 / (lambda_ + n_aug_));
    weights_(0) *= 2.0*lambda_;

    // Create sigma point matrices
    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
    Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);


    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 3.0;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 2.0;

    /**
     * DO NOT MODIFY measurement noise values below.
     * These are provided by the sensor manufacturer.
     */

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    /**
     * End DO NOT MODIFY section for measurement noise values 
     */

    // Initialize process and measurement covariance
    R_lidar_ = MatrixXd(n_z_lidar_, n_z_lidar_);
    R_lidar_ << std::pow(std_laspx_,2), 0,
                0, std::pow(std_laspy_,2);
    R_radar_ = MatrixXd(n_z_radar_, n_z_radar_);
    R_radar_ << std::pow(std_radr_,2), 0, 0,
        0, std::pow(std_radphi_,2), 0,
        0, 0, std::pow(std_radrd_,2);
    
    Q_ = MatrixXd(2, 2);
    Q_ << std::pow(std_a_,2), 0, 0, std::pow(std_yawdd_,2);

    // Start time
    time_us_ = 0;

}

UKF::~UKF() {}


void UKF::GenerateSigmaPoints(){
    // create augmented mean state
    VectorXd x_aug = VectorXd(n_aug_);
    x_aug.setZero();
    x_aug.head(n_x_) = x_;

    // create augmented covariance matrix
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
    P_aug.setZero();
    P_aug.topLeftCorner(n_x_, n_x_) = P_;
    P_aug.bottomRightCorner(2,2) = Q_;

    // create square root matrix
    MatrixXd L = P_aug.llt().matrixL();
    
    // create augmented sigma points
    Xsig_aug_.col(0) = x_aug;
    for(size_t i=0; i < n_aug_; i++){
        Xsig_aug_.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_)*L.col(i);
        Xsig_aug_.col(n_aug_ + i + 1) = x_aug - sqrt(lambda_ + n_aug_)*L.col(i);
    }

}

void UKF::PredictSigmaPoints(double delta_t){
    GenerateSigmaPoints();
    // predicted state values
    VectorXd x_pred = VectorXd(n_x_);

    for (int i = 0; i< 2*n_aug_+1; ++i) {
        double p_x = Xsig_aug_(0,i);
        double p_y = Xsig_aug_(1,i);
        double v = Xsig_aug_(2,i);
        double yaw = Xsig_aug_(3,i);
        double yaw_dot = Xsig_aug_(4,i);
        double nu_a = Xsig_aug_(5,i);
        double nu_yaw_dotdot = Xsig_aug_(6,i);
        
        // avoid division by zero
        if (std::abs(yaw_dot) > 1e-5) {
            x_pred[0] = p_x + v/yaw_dot * ( sin (yaw + yaw_dot*delta_t) - sin(yaw));
            x_pred[1] = p_y + v/yaw_dot * ( cos(yaw) - cos(yaw + yaw_dot*delta_t) );
        } else {
            x_pred[0] = p_x + v*delta_t*cos(yaw);
            x_pred[1] = p_y + v*delta_t*sin(yaw);
        }
        x_pred[2] = v;
        x_pred[3] = yaw + yaw_dot*delta_t;
        x_pred[4] = yaw_dot;

        // add noise
        VectorXd x_noise = VectorXd(n_x_);
        x_noise[0] = 0.5*nu_a*std::pow(delta_t,2) * cos(yaw);
        x_noise[1] = 0.5*nu_a*std::pow(delta_t,2) * sin(yaw);
        x_noise[2] = nu_a*delta_t;
        x_noise[3] = 0.5*nu_yaw_dotdot*std::pow(delta_t,2);
        x_noise[4] = nu_yaw_dotdot*delta_t;

        Xsig_pred_.col(i) = x_pred + x_noise;
    }
}

void UKF::Prediction(double delta_t) {
    /**
     * TODO: Complete this function! Estimate the object's location. 
     * Modify the state vector, x_. Predict sigma points, the state, 
     * and the state covariance matrix.
     */

    PredictSigmaPoints(delta_t);

    // predict state mean
    x_.setZero();
    for(size_t i=0; i < weights_.size(); i++){
        x_ += weights_[i] * Xsig_pred_.col(i);
    }

    // predict state covariance matrix
    P_.setZero();
    for(size_t i=0; i < weights_.size(); i++){
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        x_diff[3] = wrap_to_pi(x_diff[3]);
        P_ += weights_[i] * x_diff * x_diff.transpose();
    }
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    /**
     * TODO: Complete this function! Make sure you switch between lidar and radar
     * measurements.
     */

    if(!is_initialized_){
        if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
            double rho = meas_package.raw_measurements_(0);
            double phi = meas_package.raw_measurements_(1);
            double rhodot = meas_package.raw_measurements_(2);
            double x = rho * cos(phi);
            double y = rho * sin(phi);
            double vx = rhodot * cos(phi);
            double vy = rhodot * sin(phi);
            double v = sqrt(vx * vx + vy * vy);
            x_ << x, y, v, rho, rhodot;
        }else{
            x_<< meas_package.raw_measurements_[0],
                 meas_package.raw_measurements_[1], 0, 0, 0;
        }
        is_initialized_ = true;
        time_us_ = meas_package.timestamp_;
    }

    double delta_t = static_cast<double>(meas_package.timestamp_ - time_us_) * 1e-6;
    time_us_ = meas_package.timestamp_;
    Prediction(delta_t);
    
    if(meas_package.sensor_type_ == MeasurementPackage::LASER){
        UpdateLidar(meas_package);
    }else{
        UpdateRadar(meas_package);
    }
}


void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**
     * TODO: Complete this function! Use lidar data to update the belief 
     * about the object's position. Modify the state vector, x_, and 
     * covariance, P_.
     * You can also calculate the lidar NIS, if desired.
     */

    if(!use_laser_) return;


    int n_z = n_z_lidar_;
    // create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    // transform sigma points into measurement space
    for (int i = 0; i< 2*n_aug_+1; ++i) {
        Zsig(0,i) = Xsig_pred_(0,i);
        Zsig(1,i) = Xsig_pred_(1,i);
    }
    
    // calculate mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.setZero();
    for(size_t i=0; i < weights_.size(); i++){
        z_pred += weights_[i] * Zsig.col(i);
    }
 
    // measurement covariance matrix S
    MatrixXd S = R_lidar_;
    for(size_t i=0; i < weights_.size(); i++){
        VectorXd z_diff = Zsig.col(i) - z_pred;
        S += weights_[i] * z_diff * z_diff.transpose();
    }

    // Cross-correlation Matrix
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    Tc.setZero();
    for(size_t i=0; i < weights_.size(); i++){
        VectorXd z_diff = Zsig.col(i) - z_pred;
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        x_diff[3] = wrap_to_pi(x_diff[3]);
        Tc += weights_[i] * x_diff*z_diff.transpose();
    }

     // Kalman gain K
    MatrixXd K = MatrixXd(n_x_, n_z);
    K = Tc * S.inverse();

    // Update State
    VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
    x_ = x_ + K*z_diff;
    
    // Covariance Matrix Update
    P_ = P_ - K*S*K.transpose();

    // Normalised innovation squared
    NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
    std::cout << "NIS_LASER: " << NIS_laser_ << std::endl;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**
     * TODO: Complete this function! Use radar data to update the belief 
     * about the object's position. Modify the state vector, x_, and 
     * covariance, P_.
     * You can also calculate the radar NIS, if desired.
     */

    if(not use_radar_) return;


    int n_z = n_z_radar_;
    // create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    // transform sigma points into measurement space
    for (int i = 0; i< 2*n_aug_+1; ++i) {
        double p_x = Xsig_pred_(0,i);
        double p_y = Xsig_pred_(1,i);
        double v = Xsig_pred_(2,i);
        double yaw = Xsig_pred_(3,i);
        double yawd = Xsig_pred_(4,i);
        
        // predicted state values
        double rho = std::sqrt(p_x*p_x + p_y*p_y);
        double phi = std::atan2(p_y, p_x);
        double rho_dot = 0;
        if(std::abs(rho) > 1e-5){
            rho_dot = (p_x*cos(yaw)*v + p_y*sin(yaw)*v)/rho;
        } 

        // write predicted sigma point into right column
        Zsig(0,i) = rho;
        Zsig(1,i) = phi;
        Zsig(2,i) = rho_dot;
    }
    
    // calculate mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.setZero();
    for(size_t i=0; i < weights_.size(); i++){
        z_pred += weights_[i] * Zsig.col(i);
    }

    // measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);
    S = R_radar_;
    for(size_t i=0; i < weights_.size(); i++){
        VectorXd z_diff = Zsig.col(i) - z_pred;
        z_diff[1] = wrap_to_pi(z_diff[1]);
        S =  S + weights_[i] * z_diff* z_diff.transpose();
    }

    // Cross-correlation Matrix
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    Tc.setZero();
    for(size_t i=0; i < weights_.size(); i++){
        VectorXd z_diff = Zsig.col(i) - z_pred;
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        z_diff[1] = wrap_to_pi(z_diff[1]);
        x_diff[3] = wrap_to_pi(x_diff[3]);
        Tc += weights_[i] * x_diff*z_diff.transpose();
    }
   
    // Kalman gain K
    MatrixXd K = MatrixXd(n_x_, n_z);
    K = Tc * S.inverse();
  
    // Update State
    VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
    z_diff[1] = wrap_to_pi(z_diff[1]);
    x_ = x_ + K*z_diff;
  
    // Covariance Matrix Update
    P_ = P_ - K*S*K.transpose();

    // Normalised innovation squared
    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
    std::cout << "NIS_RADAR: " << NIS_radar_ << std::endl;
}



double UKF::wrap_to_pi(double angle){
    angle = fmod(angle + M_PI, 2.0 * M_PI);
    if (angle < 0)
        angle += 2.0 * M_PI;
    return angle - M_PI;
}