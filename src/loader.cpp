#include "../inc/loader.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"


Loader::Loader(int num_weights, int feature_dim, int trend_number, int dir_id, char* dir)
{
	// initialize weights
	num_weights_ = num_weights;
	feature_dim_ = feature_dim;
	// diagnosis_number_ = weight_number;
	trend_number_ = trend_number;  
	dir_id_ = dir_id;
	
	// just check...
	if(num_weights_ != 12 || trend_number_ != 15)
		std::cout << "directory number incorrect..." << std::endl;

	int len = 400;
	test_weights_dir_ = new char[len];  
	diagnosis_weights_dir_ = new char[len];  
	trend_dir_ = new char*[trend_number_];  
		
	for(int i = 0; i < trend_number_; i++)
		trend_dir_[i] = new char[len]; 	

	sprintf(common_output_prefix_, "D:/Document/HKUST/Year 5/Research/Solutions/unified_learning/output/para_%d/", dir_id_);
	sprintf(common_diagnosis_prefix_, "D:/Document/HKUST/Year 5/Research/Solutions/unified_learning/output/diagnosis_%d/", dir_id_);
	sprintf(common_data_prefix_, "D:/Document/HKUST/Year 5/Research/Data/PointClouds/");
	strcat(common_data_prefix_, dir);
	strcat(common_data_prefix_, "/"); // march 10 2014/"); // feb 23

}

void Loader::LoadWeightsForTest(Transform& transform)
{
	cv::Mat current_weight = cv::Mat::zeros(num_weights_, feature_dim_, CV_64F);	
	FileIO::ReadMatDouble(current_weight, num_weights_, feature_dim_, test_weights_dir_); 
	transform.set_w(current_weight);
}

void Loader::SaveWeightsForTest(Transform& transform)
{
	cv::Mat current_weight = cv::Mat::zeros(num_weights_, feature_dim_, CV_64F);
	current_weight = transform.w();
	FileIO::WriteMatDouble(current_weight, num_weights_, feature_dim_, test_weights_dir_);	
}

void Loader::SaveWeightsForDiagnosis(Transform& transform, int diagnosis_idx)
{
	char tmp_dir[20];	
	FormatWeightsForDiagnosisDirectory();
	sprintf(tmp_dir, "_%d.bin", diagnosis_idx);	
	strcat(diagnosis_weights_dir_, tmp_dir);	
	cv::Mat current_weight = cv::Mat::zeros(num_weights_, feature_dim_, CV_64F);
	current_weight = transform.w();
	FileIO::WriteMatDouble(current_weight, num_weights_, feature_dim_, diagnosis_weights_dir_);

}

void Loader::LoadWeightsForDiagnosis(Transform& transform, int diagnosis_idx)
{
	char tmp_dir[20];	
	FormatWeightsForDiagnosisDirectory();
	sprintf(tmp_dir, "_%d.bin", diagnosis_idx);	
	strcat(diagnosis_weights_dir_, tmp_dir);	

	cv::Mat current_weight = cv::Mat::zeros(num_weights_, feature_dim_, CV_64F);
	FileIO::ReadMatDouble(current_weight, num_weights_, feature_dim_, diagnosis_weights_dir_); 
	transform.set_w(current_weight);	
}

// save value trend: either output average value or weight norm...
void Loader::SaveTrend(std::vector<std::vector<double>>& trend_array, int trend_number, int append_flag)
{
  // FormatValueDirectory(dir_idx, true); // should be called in constructor
	int data_len = 0;
	cv::Mat data;
	for(int i = 0; i < trend_number_; i++){
		data_len = trend_array[i].size();
		data = cv::Mat::zeros(data_len, 1, CV_64F);
		for(int j = 0; j < data_len; j++)
			data.at<double>(j, 0) = trend_array[i][j];
		FileIO::RecordMatDouble(data, data_len, 1, trend_dir_[i], append_flag); 
	}
}
// format test weight directory (final weight loaded for test)
void Loader::FormatWeightsForTestDirectory()
{
	memset(test_weights_dir_, 0, sizeof(test_weights_dir_));
	strcpy(test_weights_dir_, common_output_prefix_);	
	strcat(test_weights_dir_, "w.bin");	
}

void Loader::FormatWeightsForDiagnosisDirectory()
{
	memset(diagnosis_weights_dir_, 0, sizeof(diagnosis_weights_dir_));
	strcpy(diagnosis_weights_dir_, common_diagnosis_prefix_);		
	strcat(diagnosis_weights_dir_, "w");		
}

// format trend directory
void Loader::FormatTrendDirectory()
{
	// char dir_idx_str[5];
	// sprintf(dir_idx_str, "%d/", dir_id_);
	for(int i = 0; i < trend_number_; i++)
	{
		memset(&trend_dir_[i][0], 0, sizeof(trend_dir_[i]));
		strcpy(trend_dir_[i], common_output_prefix_); // "D:/Document/HKUST/Year 5/Research/Solutions/expansion/output/para_"
		// strcat(trend_dir_[i], dir_idx_str);
	}
	AppendTrendName(trend_dir_);
	for(int i = 0; i < trend_number_; i++)	
		strcat(trend_dir_[i], "_trend.bin");
		
}

void Loader::AppendTrendName(char** trend_dir_str_array)
{
	if(trend_dir_str_array != NULL)
	{
		strcat(trend_dir_str_array[0], "w_0_0");
		strcat(trend_dir_str_array[1], "w_0_1");
		strcat(trend_dir_str_array[2], "w_0_2");
		strcat(trend_dir_str_array[3], "w_0_3");
		strcat(trend_dir_str_array[4], "w_1_0");
		strcat(trend_dir_str_array[5], "w_1_1");
		strcat(trend_dir_str_array[6], "w_1_2");
		strcat(trend_dir_str_array[7], "w_1_3");
		strcat(trend_dir_str_array[8], "w_2_0");
		strcat(trend_dir_str_array[9], "w_2_1");
		strcat(trend_dir_str_array[10], "w_2_2");
		strcat(trend_dir_str_array[11], "w_2_3");
		strcat(trend_dir_str_array[12], "idx");
		strcat(trend_dir_str_array[13], "fisher_inv");
		strcat(trend_dir_str_array[14], "natural_grad");
	}
}

void Loader::LoadLearningRates(Transform& transform) // empty parameter, need to be specialized
{
	char input_dir[400];	
	int n_w = 1;
	cv::Mat rates = cv::Mat::zeros(2, 1, CV_64F);
	sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_learning/input/rate_%d.bin", dir_id_);
	FileIO::ReadMatDouble(rates, 2, 1, input_dir);
	// need to write set learning rates routine here
	transform.set_w_rate(rates.at<double>(0, 0));
	transform.set_w_natural_rate(rates.at<double>(1, 0));
	std::cout << "learning rates: ";
	for(int i = 0; i < 2; i++)
		std::cout << rates.at<double>(i, 0) << " ";
	std::cout << std::endl;	
}


// load explained variance
void Loader::LoadProprioception(int num_data, cv::Mat& prop, cv::Mat& prop_idx, cv::Mat& home_prop_)
{
	char input_dir[400];	
	// need to be re-factored later...
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, "train_p0.bin");	
	FileIO::ReadFloatMatToDouble(prop, num_data, 1, input_dir);

	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, "train_prop_idx.bin");	
	FileIO::ReadFloatMatToDouble(prop_idx, num_data, 1, input_dir);
	
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, "prop_home.bin");	
	FileIO::ReadFloatMatToDouble(home_prop_, 1, 1, input_dir);	
}

void Loader::LoadProprioception(int num_train_data, int num_test_data, cv::Mat& train_prop, cv::Mat& test_prop, cv::Mat& home_prop, cv::Mat& train_target_idx, cv::Mat& test_target_idx)
{
	char input_dir[400];
	cv::Mat p_tmp_train = cv::Mat::zeros(num_train_data, 1, CV_64F);
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, "train_p0.bin");	
	// strcat(input_dir, "p0.bin");	
	FileIO::ReadFloatMatToDouble(p_tmp_train, num_train_data, 1, input_dir);
	// FileIO::ReadMatFloat(p_tmp_train, num_train_data, 1, "D:/Document/HKUST/Year 5/Research/Data/Arm Images/feb 22 2014/train_p0.bin");
	p_tmp_train.copyTo(train_prop.colRange(0, 1));
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, "train_p3.bin");
	// strcat(input_dir, "p3.bin");
	FileIO::ReadFloatMatToDouble(p_tmp_train, num_train_data, 1, input_dir);
	// FileIO::ReadMatFloat(p_tmp_train, num_train_data, 1, "D:/Document/HKUST/Year 5/Research/Data/Arm Images/feb 22 2014/train_p3.bin");
	p_tmp_train.copyTo(train_prop.colRange(1, 2));

	cv::Mat p_tmp_test = cv::Mat::zeros(num_test_data, 1, CV_64F);
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, "test_p0.bin");	
	FileIO::ReadFloatMatToDouble(p_tmp_test, num_test_data, 1, input_dir);
	// FileIO::ReadMatFloat(p_tmp_test, num_test_data, 1, "D:/Document/HKUST/Year 5/Research/Data/Arm Images/feb 22 2014/test_p0.bin");
	p_tmp_test.copyTo(test_prop.colRange(0, 1));
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, "test_p3.bin");
	FileIO::ReadFloatMatToDouble(p_tmp_test, num_test_data, 1, input_dir);
	// FileIO::ReadMatFloat(p_tmp_test, num_test_data, 1, "D:/Document/HKUST/Year 5/Research/Data/Arm Images/feb 22 2014/test_p3.bin");
	p_tmp_test.copyTo(test_prop.colRange(1, 2));
  
	cv::Mat p_tmp_home = cv::Mat::zeros(2, 1, CV_64F);
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, "prop_home.bin");	
	FileIO::ReadFloatMatToDouble(p_tmp_home, 2, 1, input_dir);
	// FileIO::ReadMatFloat(p_tmp_home, 2, 1, "D:/Document/HKUST/Year 5/Research/Data/Arm Images/feb 22 2014/prop_home.bin");
	home_prop = p_tmp_home.t();
	std::cout << "home prop: " << home_prop.at<double>(0, 0) << " " << home_prop.at<double>(0, 1) << std::endl;
	
	// train frame index
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, "train_prop_idx.bin");	
	FileIO::ReadFloatMatToDouble(train_target_idx, num_train_data, 1, input_dir);
	// FileIO::ReadMatFloat(train_target_idx, num_train_data, 1, "D:/Document/HKUST/Year 5/Research/Data/Arm Images/feb 22 2014/train_prop_idx.bin");
	// test frame index
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, "test_prop_idx.bin");	
	FileIO::ReadFloatMatToDouble(test_target_idx, num_test_data, 1, input_dir);
	// FileIO::ReadMatFloat(test_target_idx, num_test_data, 1, "D:/Document/HKUST/Year 5/Research/Data/Arm Images/feb 22 2014/test_prop_idx.bin");
}

void Loader::LoadPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PCDReader& reader, int idx)
{	
	char tmp_dir[40];
	char input_dir[400];
	sprintf(tmp_dir, "pcd/%d.pcd", idx);
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, tmp_dir);
	reader.read(input_dir, *cloud);	
}

void Loader::LoadBinaryPointCloud(cv::Mat& cloud, int idx)
{	
	char tmp_dir[40];
	char input_dir[400];
	sprintf(tmp_dir, "binary/size_%d.bin", idx);
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, tmp_dir);
	cv::Mat size_mat = cv::Mat::zeros(1, 1, CV_64F);
	FileIO::ReadMatDouble(size_mat, 1, 1, input_dir);
	int cloud_size = size_mat.at<double>(0, 0);	
	int dim = 4;
	cloud = cv::Mat::ones(cloud_size, dim, CV_64F);	
	sprintf(tmp_dir, "binary/%d.bin", idx);
	strcpy(input_dir, common_data_prefix_);
	strcat(input_dir, tmp_dir);
	FileIO::ReadMatDouble(cloud.colRange(0, dim - 1), cloud_size, dim - 1, input_dir);		
}

void Loader::SavePointCloudAsBinaryMat(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int idx)
{	
	char tmp_dir[40];
	char output_dir[400];
	int dim = 3;	
	int cloud_size = cloud->points.size();
	cv::Mat cloud_mat = cv::Mat::zeros(cloud_size, dim, CV_64F);
	for(int i = 0; i < cloud_size; i++)
	{
		cloud_mat.at<double>(i, 0) = cloud->points[i].x;
		cloud_mat.at<double>(i, 1) = cloud->points[i].y;
		cloud_mat.at<double>(i, 2) = cloud->points[i].z;
	}
	sprintf(tmp_dir, "binary/%d.bin", idx);
	strcpy(output_dir, common_data_prefix_);
	strcat(output_dir, tmp_dir);
	FileIO::WriteMatDouble(cloud_mat, cloud_size, dim, output_dir);
	// save size...
	sprintf(tmp_dir, "binary/size_%d.bin", idx);
	strcpy(output_dir, common_data_prefix_);
	strcat(output_dir, tmp_dir);
	cv::Mat size_mat = cv::Mat::zeros(1, 1, CV_64F);
	size_mat.at<double>(0, 0) = cloud_size;
	FileIO::WriteMatDouble(size_mat, 1, 1, output_dir);
	
}


//cv::Mat elips_home_mu = cv::Mat::zeros(2, 1, CV_64F);
//elips_home_mu.at<double>(0, 0) = x;
//elips_home_mu.at<double>(1, 0) = y;
//FileIO::WriteMatDouble(elips_home_mu, 2, 1, diagnosis_weights_dir_[5]);
//cv::Mat elips_home_cov = cv::Mat::zeros(3, 1, CV_64F);
//elips_home_cov.at<double>(0, 0) = phi;
//elips_home_cov.at<double>(1, 0) = lx;
//elips_home_cov.at<double>(2, 0) = sx;
//FileIO::WriteMatDouble(elips_home_cov, 3, 1, diagnosis_weights_dir_[6]);

//cv::Mat elips_home_mu = cv::Mat::zeros(2, 1, CV_64F);
//FileIO::ReadMatDouble(elips_home_mu, 2, 1, diagnosis_weights_dir_[5]);
//// ellipse.set_home_mu(elips_home_mu);
//*p_x = elips_home_mu.at<double>(0, 0);
//*p_y = elips_home_mu.at<double>(1, 0);
//	
//cv::Mat elips_home_cov = cv::Mat::zeros(3, 1, CV_64F);
//FileIO::ReadMatDouble(elips_home_cov, 3, 1, diagnosis_weights_dir_[6]);
//// ellipse.set_home_cov(elips_home_cov);	
//*p_phi = elips_home_cov.at<double>(0, 0);
//*p_lx = elips_home_cov.at<double>(1, 0);
//*p_sx = elips_home_cov.at<double>(2, 0);