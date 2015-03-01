#ifndef _EXPLORER_H
#define _EXPLORER_H


#define GRADIENT_SCALE 100000

#include <iostream>
#include <random>
#include <time.h>
#include <queue>
#include <deque>
#include <array>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/nonfree/features2d.hpp"

#include "../inc/fio.h"
#include "../inc/loader.h"
#include "../inc/transform.h"

#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>

typedef struct {
    double r,g,b;
} COLOUR;

// typedef std::vector<double> fL;

class Explorer{
private:
       
	int dir_id_; 	
	int dim_feature_;
	int num_joints_;	
	int num_trend_;
	int num_train_data_;
	int num_test_data_;
	long num_iteration_;	
	int path_count_;	
	int dim_transform_;
	int num_weights_;
	char dir_[40];
	
    double max_exploration_range_;
    double starting_exploration_range_;	
	double range_expanding_period_;
	double avg_cost_;	
	
	std::random_device rd_;

	std::vector<double> targets_;
	std::vector<double> prev_targets_;
	std::vector<std::vector<double>> path_;
	std::vector<double> kernel_list_;

	cv::Mat action_;
    cv::Mat train_prop_;
	cv::Mat train_target_idx_;    
	cv::Mat test_prop_;
	cv::Mat test_target_idx_;   
    cv::Mat home_prop_;
	cv::Mat curr_prop_;
	cv::Mat curr_prop_matrix_;
	cv::Mat prop_diff_;
	cv::Mat prop_dist_;
	cv::Mat aim_idx_matrix_;
	cv::Mat feature_;

	// pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
	// pcl::PointCloud<pcl::PointXYZ>::Ptr prev_cloud_;
	// pcl::PointCloud<pcl::PointXYZ>::Ptr home_cloud_;
	// pcl::PointCloud<pcl::PointXYZ>::Ptr prev_home_cloud_;
	// pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_cloud_;	

	cv::Mat cloud_;
	cv::Mat prev_cloud_;
	cv::Mat home_cloud_;
	cv::Mat prev_home_cloud_;
	cv::Mat tmp_cloud_;

	Transform transform_;

	int cloud_scale_;
    
public:
    // initialization
    Explorer(int dir_id, int num_iteration, int expanding_period, char* dir, int dim_transform, int dim_feature, int num_joints);
    ~Explorer();	
	void GenerateLinePath(std::vector<std::vector<double>>& path, std::vector<double>& targets, std::vector<double>& prev_targets);
	int GenerateAimIndexLinePath(std::mt19937& engine, int current_iteration);
	void LearningFromPointCloud();
	void TestTransformation(int single_frame_flag, int display_flag, int test_idx);
	void ShowTransformationGrid(int num_grid);
	void Explorer::LearningFromPointCloudTest();
	void DownSamplingPointCloud(double voxel_size, pcl::VoxelGrid<pcl::PointXYZ>& voxel_grid, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr down_sampled_cloud);
	void DepthFiltering(float depth, pcl::PassThrough<pcl::PointXYZ>& pass, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud);
	void ShowCloudSequence();
	void SetFeature(cv::Mat& feature, int aim_idx, cv::Mat& prop, cv::Mat& home_prop);
	void SetKernel(std::vector<double>& kernel_list, cv::Mat& data, double* p_current_data, int dim_left, int curr_pos, int data_length, int kernel_dim, int value_flag);
	void PreprocessingAndSavePointCloud();
	void PCD2Mat(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, cv::Mat& cloud_mat);
	void Mat2PCD(cv::Mat& cloud_mat, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
	void ReOrder(cv::Mat& input, cv::Mat& output, cv::Mat& input_indices);
	void RecordingTrend(Transform& transform, Loader& loader, std::vector<std::vector<double>>& trend_array, int iter, int write_trend_interval, double aim_idx);
};

struct DistCompare{
    // the fifth component is the distance...
    inline bool operator()(const cv::Mat& a, const cv::Mat& b){
        return a.at<double>(0, 0) < b.at<double>(0, 0);
    }
};


#endif
