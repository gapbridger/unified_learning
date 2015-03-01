#ifndef _TRANSFORM_H
#define _TRANSFORM_H
#define GRADIENT_SCALE 100000

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/types_c.h"
#include <opencv2/flann.hpp>
#include "../inc/fio.h"

#include <random>
#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/correspondence.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/registration.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/visualization/pcl_visualizer.h>

class Transform
{
private:
	
	// transformation structures		
	cv::Mat transform_inv_;
	cv::Mat transform_;
	cv::Mat prev_transform_inv_;
	cv::Mat prev_transform_;
	cv::Mat transform_elements_;
	cv::Mat e_;

	// weights
	cv::Mat w_;
	cv::Mat w_grad_;
	cv::Mat w_grad_vec_;
	cv::Mat w_grad_batch_;
	cv::Mat w_grad_vector_; // long vector containing all w gradients
	cv::Mat fisher_inv_;
	cv::Mat tmp_;
	cv::Mat natural_w_grad_;
	cv::Mat natural_w_grad_vec_;


	Eigen::Matrix4d eigen_transform_inv_;
	// dimensions...
	int feature_dim_;	
	int transform_dim_;
	int num_weights_;
	int gradient_iter_;

	double w_rate_;
	double w_natural_rate_;
	double average_norm_;
	double lambda_;
	double ini_norm_;
	double average_norm_fisher_;	
	double ini_norm_fisher_;
	double rejection_threshold_;
	
	// std::vector<double> w_rate_; // including natural gradient rate...	

	pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ> estimation_;
	pcl::Correspondences corr_;
	pcl::PointCloud<pcl::PointXYZ>::Ptr prev_cloud_;
	pcl::PointCloud<pcl::PointXYZ>::Ptr curr_cloud_;
	pcl::PointCloud<pcl::PointXYZ>::Ptr prev_home_cloud_;
	pcl::PointCloud<pcl::PointXYZ>::Ptr curr_home_cloud_;

public:
	Transform(int transform_dim_, int feature_dim_);	
	void CalcTransformInv(cv::Mat& feature);
	void TransformDataInv(cv::Mat& input_cloud, cv::Mat& output_cloud, int curr_flag);	
	void TransformData(cv::Mat& input_cloud, cv::Mat& output_cloud, int curr_flag);
	void CalcGradient(cv::Mat& target_cloud, cv::Mat& query_cloud, cv::Mat prediction_cloud, cv::Mat& feature, int iter);
	void cv2eigen(const cv::Mat& input, Eigen::Matrix4d& output);
	// gradients		
	void Update(int iter);
	void CopyTransformToPrev();
	// calculate inverse transformation			
	
	void GetNearestNeighborMatches(cv::Mat& target_cloud, cv::Mat& prediction_cloud, cv::Mat& matched_target_cloud, cv::Mat& cost, int cost_flag);
	cv::Mat TransformDataPointInv(cv::Mat& point, int curr_flag);
	cv::Mat TransformToPreviousFrame(cv::Mat& curr_img_point);
	cv::Mat TransformToNextFrame(cv::Mat& prev_img_point);
	void Rejection(cv::Mat& diff, cv::Mat& filtered_diff, cv::Mat& query_cloud, cv::Mat& filtered_query_cloud, double threshold);
	void set_w_rate(double w_rate);
	void set_w_natural_rate(double natural_rate);
	// void SetLearningRate(double rate)
	// copy to previous transformation
	void CopyToPrev();
	// helper functions
	cv::Mat w();
	cv::Mat fisher_inv();
	cv::Mat natural_w_grad();
	cv::Mat transform_inv();
	cv::Mat prev_transform_inv();
	void set_w(cv::Mat& w);	
	void set_fisher_inv();
	cv::Mat w_grad();
	// check gradient
	void CheckInvGradient();
};

#endif