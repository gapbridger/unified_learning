#include "../inc/transform.h"

Transform::Transform(int transform_dim, int feature_dim)
{
	// 
	feature_dim_ = feature_dim;	
	transform_dim_ = transform_dim;
	num_weights_ = transform_dim_ * (transform_dim_ - 1);
	// learning rates
	w_rate_ = 0.1; // cv::Mat::zeros(num_weights_ + 1, 1, CV_64F);
	/*for(int i = 0; i < num_weights_; i++)
		w_rate_[i] = 0;*/	
	// only work for the situation where output dim equal to 1
	w_ = cv::Mat::zeros(num_weights_, feature_dim_, CV_64F);
	w_grad_ = cv::Mat::zeros(num_weights_, feature_dim_, CV_64F);
	w_grad_vec_ = cv::Mat::zeros(feature_dim_ * num_weights_, 1, CV_64F);
	natural_w_grad_vec_ = cv::Mat::zeros(feature_dim_ * num_weights_, 1, CV_64F);
	natural_w_grad_ = cv::Mat::zeros(num_weights_, feature_dim_, CV_64F);
	fisher_inv_ = cv::Mat::eye(feature_dim_ * num_weights_, feature_dim_ * num_weights_, CV_64F);	
	// transformations
	transform_inv_ = cv::Mat::eye(transform_dim_, transform_dim_, CV_64F);
	prev_transform_inv_ = cv::Mat::eye(transform_dim_, transform_dim_, CV_64F);	
	transform_ = cv::Mat::eye(transform_dim_, transform_dim_, CV_64F);
	prev_transform_ = cv::Mat::eye(transform_dim_, transform_dim_, CV_64F);	
	e_ = cv::Mat::eye(transform_dim_, transform_dim_, CV_64F);
	// initialize weights...
	std::random_device rd;
	std::mt19937 engine_uniform_rand(rd());
	double weight_range = 0.0001;
	std::uniform_real_distribution<double> uniform_rand(-weight_range, weight_range);

	// initialize all the weights
	for(int p = 0; p < num_weights_; p++)	
		for(int q = 0; q < feature_dim_; q++)		
			w_.at<double>(p, q) = uniform_rand(engine_uniform_rand);	

	average_norm_ = 1;
	lambda_ = 0.2;
	rejection_threshold_ = 0.10; // 0.15
}

void Transform::CalcTransformInv(cv::Mat& feature)
{
	transform_elements_ = w_ * feature;	
	// std::cout << "calc transform inv..." << std::endl;
	transform_inv_.rowRange(0, transform_dim_ - 1) = e_.rowRange(0, transform_dim_ - 1) + transform_elements_.reshape(1, transform_dim_ - 1);
	cv::invert(transform_inv_, transform_);	
}


void Transform::TransformDataInv(cv::Mat& input_cloud, cv::Mat& output_cloud, int curr_flag)
{
	// input cloud is 4 by n... output cloud is also 4 by n...
	// output_cloud = cv::Mat::zeros(input_cloud.rows, input_cloud.cols, CV_64F);
	if(curr_flag)
		output_cloud = input_cloud * transform_inv_.t();		
	else
		output_cloud = input_cloud * prev_transform_inv_.t();	
}

void Transform::TransformData(cv::Mat& input_cloud, cv::Mat& output_cloud, int curr_flag)
{	
	if(curr_flag)
		output_cloud = input_cloud * transform_.t();		
	else
		output_cloud = input_cloud * prev_transform_.t();	
}

void Transform::GetNearestNeighborMatches(cv::Mat& target_cloud, cv::Mat& prediction_cloud, cv::Mat& matched_target_cloud, cv::Mat& cost, int cost_flag)
{
	int query_size = prediction_cloud.rows;
	cv::Mat target_cloud_float, prediction_cloud_float; // suppose all the input clouds are double...
	cv::Mat indices = cv::Mat::zeros(query_size, 1, CV_32S);
	cv::Mat min_dists = cv::Mat::zeros(query_size, 1, CV_32F);
	cv::Mat cost_float = cv::Mat::zeros(1, 1, CV_32F);
	target_cloud.convertTo(target_cloud_float, CV_32F);
	prediction_cloud.convertTo(prediction_cloud_float, CV_32F);
	cv::flann::Index flann_index(target_cloud_float, cv::flann::KDTreeIndexParams(1), cvflann::FLANN_DIST_EUCLIDEAN);
	flann_index.knnSearch(prediction_cloud_float, indices, min_dists, 1);
	if(cost_flag)
	{
		// cv::reduce(diff.mul(diff) / 2, diff, 1, CV_REDUCE_SUM);
		cv::reduce(min_dists, cost_float, 0, CV_REDUCE_AVG);	
		cost.at<double>(0, 0) = cost_float.at<float>(0, 0);
	}
	matched_target_cloud = cv::Mat::zeros(query_size, transform_dim_, CV_64F);
	// i think the indices are for the target cloud...
	for(int i = 0; i < prediction_cloud.rows; i++)
		for(int j = 0; j < transform_dim_; j++)
			matched_target_cloud.at<double>(i, j) = target_cloud.at<double>(indices.at<int>(i, 0), j);
}

void Transform::CalcGradient(cv::Mat& matched_target_cloud, cv::Mat& query_cloud, cv::Mat prediction_cloud, cv::Mat& feature, int iter)
{
	// define the kd tree search index...
	
	// the cloud should be n by 4...			
	cv::Mat diff, transform_grad, filtered_diff, filtered_query_cloud;
	// double rejection_threshold = 0.0; // 5 mm...
	double query_size = 0;		
	diff = prediction_cloud - matched_target_cloud;
	Rejection(diff, filtered_diff, query_cloud, filtered_query_cloud, rejection_threshold_);
	query_size = filtered_query_cloud.rows;							
	// reshape this into a gradient versus f...
	transform_grad = 1.0 / query_size * filtered_diff.colRange(0, transform_dim_ - 1).t() * filtered_query_cloud;
	// reshaping is fine here...
	// std::cout << "gradient" << std::endl;
	transform_grad = transform_grad.reshape(1, num_weights_);
	w_grad_ = transform_grad * feature.t();

	double epsilon = w_natural_rate_; // * ini_norm_fisher_ / average_norm_fisher_; // 1e-4; // 2e-5 is good... slow update though...	
	w_grad_ = w_grad_.reshape(1, num_weights_ * feature_dim_);		
	cv::Mat tmp = w_grad_.t() * fisher_inv_ * w_grad_;	
	cv::Mat tmp_1 = fisher_inv_ * w_grad_;
	fisher_inv_ = 1 / (1 - epsilon)* fisher_inv_ - epsilon / ((1- epsilon) * (1 - epsilon + epsilon * tmp.at<double>(0, 0))) * (tmp_1 * tmp_1.t());   	
	natural_w_grad_ = fisher_inv_ * w_grad_;
	natural_w_grad_ = natural_w_grad_.reshape(1, num_weights_);
	
}

void Transform::Rejection(cv::Mat& diff, cv::Mat& filtered_diff, cv::Mat& query_cloud, cv::Mat& filtered_query_cloud, double threshold)
{
	if(threshold != 0)
	{
		cv::Mat dist, idx, tmp_diff;
		tmp_diff = diff.mul(diff);
		cv::reduce(tmp_diff, dist, 1, CV_REDUCE_SUM);
		cv::sqrt(dist, dist);
		cv::sortIdx(dist, idx, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
		int count = (int)(dist.rows * (1 - threshold));
		/*while(count < dist.rows && dist.at<double>(idx.at<int>(count, 0), 0) < threshold)
			count++;*/
		// std::cout << "original: " << diff.rows << " filtered: " << count << std::endl;
		filtered_diff = cv::Mat::zeros(count, diff.cols, CV_64F);
		filtered_query_cloud = cv::Mat::zeros(count, query_cloud.cols, CV_64F);
		
		for(int i = 0; i < count; i++)		
		{
			// diff
			for(int m = 0; m < diff.cols; m++)
				filtered_diff.at<double>(i, m) = diff.at<double>(idx.at<int>(i, 0), m);
			// query_cloud		
			for(int n = 0; n < query_cloud.cols; n++)
				filtered_query_cloud.at<double>(i, n) = query_cloud.at<double>(idx.at<int>(i, 0), n);		
		}
	}
	else
	{
		filtered_diff = cv::Mat::zeros(diff.rows, diff.cols, CV_64F);
		diff.copyTo(filtered_diff);
		filtered_query_cloud = cv::Mat::zeros(query_cloud.rows, query_cloud.cols, CV_64F);
		query_cloud.copyTo(filtered_query_cloud);
	}


}


void Transform::Update(int iter)
{
	// double eta = 0.004;
	// w_ = w_ - w_rate_ * w_grad_;
	// double rate = 1e-20;
	double curr_norm = cv::norm(natural_w_grad_, cv::NORM_L2);
	if(iter == 1)
	{
		ini_norm_ = curr_norm;
		average_norm_ = curr_norm;		
	}
	else	
		average_norm_ = (1 - lambda_) * average_norm_ + lambda_ * curr_norm;	
	w_ = w_ - (w_rate_ * ini_norm_ / average_norm_) * natural_w_grad_;
	// w_ = w_ - w_rate_ * natural_w_grad_;
}

// copy to previous transformation
void Transform::CopyTransformToPrev()
{
	transform_inv_.copyTo(prev_transform_inv_);
}
cv::Mat Transform::fisher_inv()
{
	return fisher_inv_;
}

cv::Mat Transform::natural_w_grad()
{
	return natural_w_grad_;
}

cv::Mat Transform::w_grad()
{
	return w_grad_;
}

cv::Mat Transform::w()
{
	return w_;
}

void Transform::set_w(cv::Mat& w)
{
	 w.copyTo(w_);
}

void Transform::set_w_rate(double w_rate)
{
	w_rate_ = w_rate;		
}

void Transform::set_w_natural_rate(double natural_rate)
{
	w_natural_rate_ = natural_rate;
}

void Transform::set_fisher_inv()
{
	fisher_inv_ = cv::Mat::eye(feature_dim_ * num_weights_, feature_dim_ * num_weights_, CV_64F);
}

void Transform::CheckInvGradient()
{
	char input_dir[400];
	cv::Mat feature = cv::Mat::zeros(feature_dim_ + 1, 1, CV_64F);
	pcl::PCDReader reader;
	sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_learning/test/feature.bin");
	FileIO::ReadMatDouble(feature, feature_dim_ + 1, 1, input_dir);
	feature = feature.rowRange(1, feature.rows);

	pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr original_cloud(new pcl::PointCloud<pcl::PointXYZ>);	
	sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_learning/test/target_cloud.pcd");
	reader.read(input_dir, *target_cloud);
	sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_learning/test/original_cloud.pcd");
	reader.read(input_dir, *original_cloud);	
	cv::Mat target_cloud_mat = cv::Mat::zeros(target_cloud->points.size(), transform_dim_, CV_64F);
	cv::Mat target_cloud_mat_float = cv::Mat::zeros(target_cloud->points.size(), transform_dim_, CV_32F);
	cv::Mat original_cloud_mat = cv::Mat::zeros(original_cloud->points.size(), transform_dim_, CV_64F);
	
	for(int i = 0; i < target_cloud_mat.rows; i++)
	{
		target_cloud_mat.at<double>(i, 0) = target_cloud->points[i].x;
		target_cloud_mat.at<double>(i, 1) = target_cloud->points[i].y;
		target_cloud_mat.at<double>(i, 2) = target_cloud->points[i].z;
		target_cloud_mat.at<double>(i, 3) = 1;
	}
	for(int i = 0; i < original_cloud_mat.rows; i++)
	{
		original_cloud_mat.at<double>(i, 0) = original_cloud->points[i].x;
		original_cloud_mat.at<double>(i, 1) = original_cloud->points[i].y;
		original_cloud_mat.at<double>(i, 2) = original_cloud->points[i].z;
		original_cloud_mat.at<double>(i, 3) = 1;
	}

	
	cv::Mat disturb = cv::Mat::zeros(num_weights_, feature_dim_, CV_64F);
	cv::Mat cost = cv::Mat::zeros(1, 1, CV_64F);
	cv::Mat dist = cv::Mat::zeros(1, 1, CV_64F);
	cv::Mat tmp_w;	
	// cv::Mat prediction_cloud, matched_target_cloud;
	double e_1 = 0;
	double e_2 = 0;
	double disturb_value = 0.00001;
	double numerical_gradient = 0;
	double analytical_gradient = 0;		

	// define the kd tree search index...	
	CalcTransformInv(feature);
	// the cloud should be n by 4...
	cv::Mat prediction_cloud, matched_target_cloud, diff, filtered_diff, filtered_query_cloud;
	TransformDataInv(original_cloud_mat, prediction_cloud, 1);
	GetNearestNeighborMatches(target_cloud_mat, prediction_cloud, matched_target_cloud, cost, 0);	
	CalcGradient(matched_target_cloud, original_cloud_mat, prediction_cloud, feature, 0);		
	for(int i = 0; i < num_weights_; i++)
	{
		for(int j = 0; j < feature_dim_; j++)
		{
			disturb = cv::Mat::zeros(num_weights_, feature_dim_, CV_64F);
			disturb.at<double>(i, j) = disturb_value;
			w_ = w_ + disturb;
			transform_elements_ = w_ * feature;	
			transform_inv_.rowRange(0, transform_dim_ - 1) = e_.rowRange(0, transform_dim_ - 1) + transform_elements_.reshape(1, transform_dim_ - 1);
			// the cloud should be n by 4...
			prediction_cloud = original_cloud_mat * transform_inv_.t();	
			diff = prediction_cloud - matched_target_cloud;
			Rejection(diff, filtered_diff, original_cloud_mat, filtered_query_cloud, rejection_threshold_);
			cv::reduce(filtered_diff.mul(filtered_diff) / 2, filtered_diff, 1, CV_REDUCE_SUM);
			cv::reduce(filtered_diff, dist, 0, CV_REDUCE_AVG);
			/*cv::reduce(diff.mul(diff) / 2, diff, 1, CV_REDUCE_SUM);
			cv::reduce(diff, dist, 0, CV_REDUCE_AVG);*/
			e_1 = dist.at<double>(0, 0);

			w_ = w_ - 2 * disturb;
			transform_elements_ = w_ * feature;	
			transform_inv_.rowRange(0, transform_dim_ - 1) = e_.rowRange(0, transform_dim_ - 1) + transform_elements_.reshape(1, transform_dim_ - 1);
			// the cloud should be n by 4...
			prediction_cloud = original_cloud_mat * transform_inv_.t();	
			diff = prediction_cloud - matched_target_cloud;
			Rejection(diff, filtered_diff, original_cloud_mat, filtered_query_cloud, rejection_threshold_);
			cv::reduce(filtered_diff.mul(filtered_diff) / 2, filtered_diff, 1, CV_REDUCE_SUM);
			cv::reduce(filtered_diff, dist, 0, CV_REDUCE_AVG);
			/*cv::reduce(diff.mul(diff) / 2, diff, 1, CV_REDUCE_SUM);
			cv::reduce(diff, dist, 0, CV_REDUCE_AVG);*/
			e_2 = dist.at<double>(0, 0);

			w_ = w_ + disturb;
			numerical_gradient = (e_1 - e_2) / (2 * disturb_value);
			analytical_gradient = w_grad_.at<double>(i, j);
			std::cout << i << " " << j << ": analytical gradient: " << analytical_gradient << " " << "numerical gradient: " << numerical_gradient << std::endl;
		}
	}
	std::cout << "gradient check finished..." << endl;
}

// natural_w_grad_ = fisher_inv_ * w_grad_.rowRange(;
	// natural_w_grad_ = natural_w_grad_.reshape(1, num_weights_);

	//double epsilon = 2e-5; // 2e-5 is good... slow update though...
	//w_grad_ = w_grad_.reshape(1, num_weights_ * feature_dim_);		
 //   cv::Mat tmp = tmp_ = w_grad_.t() * fisher_inv_ * w_grad_;
	//double tmp_value = tmp.at<double>(0, 0);
	//fisher_inv_ = (1 / (1 - epsilon)) * (fisher_inv_ - epsilon / (1- epsilon + epsilon * tmp_value) * fisher_inv_ * (w_grad_ * w_grad_.t()) * fisher_inv_);    
	//natural_w_grad_ = fisher_inv_ * w_grad_;
	//natural_w_grad_ = natural_w_grad_.reshape(1, num_weights_);

	/*for(int i = 0; i < num_weights_; i++)
	{
		cv::Mat tmp_vec = w_grad_.rowRange(i, i + 1).t();
		tmp_vec.copyTo(w_grad_vec_.rowRange(i * feature_dim_, (i + 1) * feature_dim_));
	}
	for(int i = 0; i < num_weights_; i++)
	{
		cv::Mat tmp_vec = w_grad_.rowRange(i, i + 1).t();
		tmp_vec.copyTo(w_grad_vec_.rowRange(i * feature_dim_, (i + 1) * feature_dim_));
	}

	for(int i = 0; i < num_weights_; i++)
	{
		cv::Mat tmp_vec = natural_w_grad_vec_.rowRange(i * feature_dim_, (i + 1) * feature_dim_).t();		
		tmp_vec.copyTo(natural_w_grad_.rowRange(i, i + 1));
	}

	*/
	// fisher_inv_ = (1 / (1 - epsilon)) * (fisher_inv_ - (epsilon / (1- epsilon + epsilon * tmp.at<double>(0, 0))) * ((fisher_inv_ * w_grad_vec_) * (fisher_inv_ * w_grad_vec_).t()));
	// fisher_inv_ = (1 + epsilon) * fisher_inv_ - epsilon * fisher_inv_ * w_grad_ * w_grad_.t() * fisher_inv_;
	/*for(int i = 0; i < num_weights_; i++)
	{
		cv::Mat tmp_vec = natural_w_grad_vec_.rowRange(i * feature_dim_, (i + 1) * feature_dim_).t();		
		tmp_vec.copyTo(natural_w_grad_.rowRange(i, i + 1));
	}*/

     
     // w_ = w - eta .* natural_w_grad; % normal gradient


//cv::Mat phi = cv::repeat(feature.t(), query_size, 1);
//cv::Mat f = 1.0 / query_size * filtered_query_cloud.t() * phi;	
//f = f.reshape(1, transform_dim_ * feature_dim_);	
//// cv::Mat tmp_1 = fisher_inv_ * f;
//// cv::Mat tmp_2 = f.t() * fisher_inv_ * f;
//// huge estimation equation of fisher information matrix
//// fisher_inv_ = 1 / (1 - epsilon) * fisher_inv_ - epsilon / pow(1 - epsilon, 2) * tmp * ((cv::Mat::eye(transform_dim_ - 1, transform_dim_ - 1, CV_64F) + epsilon / (1 - epsilon) * f.t() * fisher_inv_ * f).inv()) * tmp.t();
//// fisher_inv_ = 1 / (1 - epsilon) * fisher_inv_ - epsilon / ((1 - epsilon) *  (1 - epsilon + epsilon * tmp_2.at<double>(0, 0))) * tmp_1 * tmp_1.t(); // ((cv::Mat::eye(transform_dim_ - 1, transform_dim_ - 1, CV_64F) + epsilon / (1 - epsilon) * f.t() * fisher_inv_ * f).inv()) * tmp.t();
//fisher_inv_ = (1 + epsilon) * fisher_inv_ - epsilon * fisher_inv_ * f * f.t() * fisher_inv_;
//w_grad_ = w_grad_.reshape(1, num_weights_ * feature_dim_);		
//for(int i = 0; i < transform_dim_ - 1; i++)
//{
//	cv::Mat tmp = fisher_inv_ * w_grad_.rowRange(i * transform_dim_ * feature_dim_, (i + 1) * transform_dim_ * feature_dim_);
//	tmp = tmp.reshape(1, transform_dim_);
//	tmp.copyTo(natural_w_grad_.rowRange(i * transform_dim_, (i + 1) * transform_dim_)); //  * transform_dim_ * feature_dim_, (i + 1) * transform_dim_ * feature_dim_));
//}