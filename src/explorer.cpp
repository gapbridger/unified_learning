// With google c++ coding style here
#include "../inc/explorer.h"

COLOUR GetColour(double v, double vmin, double vmax)
{
	COLOUR c = {1.0,1.0,1.0}; // white
	double dv;

	if (v < vmin)
		v = vmin;
	if (v > vmax)
		v = vmax;

	dv = vmax - vmin;

	if (v < (vmin + 0.25 * dv)) 
	{
		c.r = 0;
		c.g = 4 * (v - vmin) / dv;
	}
	else if (v < (vmin + 0.5 * dv)) 
	{
		c.r = 0;
		c.b = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
	}
	else if (v < (vmin + 0.75 * dv)) 
	{
		c.r = 4 * (v - vmin - 0.5 * dv) / dv;
		c.b = 0;
	}
	else
	{
		c.g = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
		c.b = 0;
	}

	return c;
}

// constructor
Explorer::Explorer(int dir_id, int num_iteration, int expanding_period, char* dir, int dim_transform, int dim_feature, int num_joints) 
	: transform_(dim_transform, dim_feature)	  
{    	
	
	sprintf(dir_, dir);		

	dir_id_ = dir_id; 	
	dim_feature_ = dim_feature;
	dim_transform_ = dim_transform;
	num_weights_ = dim_transform_ * (dim_transform_ - 1);
	num_trend_ = num_weights_ + 3;		
	num_joints_ = num_joints;	

	num_train_data_ = 17900 / 20 * 19;
	num_test_data_ = 17900 / 20 * 1;
	//num_train_data_ = 8210; // 17900 / 20 * 19;
	//num_test_data_ = 0; // 17900 / 20 * 1;
	num_iteration_ = num_iteration;	
	path_count_ = 0;		
	
    max_exploration_range_ = 1;
    starting_exploration_range_ = 0.02;	
	range_expanding_period_ = expanding_period;
	avg_cost_ = 0;	
	
	
	targets_ = std::vector<double>(num_joints_);
	prev_targets_ = std::vector<double>(num_joints_);
	train_prop_ = cv::Mat::zeros(num_train_data_, num_joints_, CV_64F);    
	train_target_idx_ = cv::Mat::zeros(num_train_data_, 1, CV_64F);    
	test_prop_ = cv::Mat::zeros(num_test_data_, num_joints_, CV_64F);    
	test_target_idx_ = cv::Mat::zeros(num_test_data_, 1, CV_64F);    
    home_prop_ = cv::Mat::zeros(1, num_joints_, CV_64F);
	curr_prop_ = cv::Mat::zeros(1, num_joints_, CV_64F);
	curr_prop_matrix_ = cv::Mat::zeros(num_train_data_, num_joints_, CV_64F);
	prop_diff_ = cv::Mat::zeros(num_train_data_, num_joints_, CV_64F);
	prop_dist_ = cv::Mat::zeros(num_train_data_, 1, CV_64F);
	aim_idx_matrix_ = cv::Mat::zeros(num_train_data_, 1, CV_64F);
	feature_ = cv::Mat::zeros(dim_feature_, 1, CV_64F);	

	cloud_scale_ = 1000;

}

Explorer::~Explorer(){
}




void Explorer::LearningFromPointCloud()
{
	char input_dir[400];
	char output_dir[400];	
	int aim_idx = 0;	
	int aim_frame_idx = 0;
	int num_gradient_iteration = 1;
	unsigned long iteration_count = 0;
	double depth_threshold = 0.8;
	double voxel_grid_size = 0.008;
	int cost_flag = 0;
	int write_trend_interval = 1000;
	int write_diagnosis_interval = 1000;
	cv::Mat target_cloud, matched_target_cloud, transformed_query_cloud, indices, min_dists;
	cv::Mat cost = cv::Mat::zeros(1, 1, CV_32F);
	std::mt19937 engine(rd_());		
	std::vector<std::vector<double>> trend_array(num_trend_, std::vector<double>(0));

	Loader loader(num_weights_, dim_feature_, num_trend_, dir_id_, dir_);
	loader.FormatWeightsForTestDirectory();
	loader.FormatTrendDirectory();
	loader.LoadLearningRates(transform_);			
	// loader.LoadProprioception(num_train_data_, train_prop_, train_target_idx_, home_prop_);		
	loader.LoadProprioception(num_train_data_, num_test_data_, train_prop_, test_prop_, home_prop_, train_target_idx_, test_target_idx_);	
	// loader.LoadLearningRates(transform_);
	cv::Mat size = cv::Mat::zeros(1, 1, CV_64F);
	
	for(iteration_count = 0; iteration_count < num_iteration_; iteration_count++)
	{		
		aim_idx = GenerateAimIndexLinePath(engine, iteration_count);			
		aim_frame_idx = train_target_idx_.at<double>(aim_idx, 0);
		loader.LoadBinaryPointCloud(cloud_, aim_frame_idx);
		cloud_ = cloud_ * cloud_scale_;
		// cloud: query_cloud, home_cloud: prediction_cloud, prev_home_cloud: target_cloud
		SetFeature(feature_, aim_idx, train_prop_, home_prop_);
		// feature_ = feature_.rowRange(1, feature_.rows);
		transform_.CalcTransformInv(feature_);
		transform_.TransformDataInv(cloud_, home_cloud_, 1);		
		
		if(iteration_count != 0)
		{
			prev_home_cloud_.convertTo(target_cloud, CV_32F);	
			cv::flann::Index kd_trees(target_cloud, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN); // build kd tree
			//prev_home_cloud_.convertTo(target_cloud, CV_32F);				
			home_cloud_.convertTo(transformed_query_cloud, CV_32F); 
			indices = cv::Mat::zeros(transformed_query_cloud.rows, 1, CV_32S);
			min_dists = cv::Mat::zeros(transformed_query_cloud.rows, 1, CV_32F);			
			kd_trees.knnSearch(transformed_query_cloud, indices, min_dists, 1, cv::flann::SearchParams(64)); // kd tree search
			cv::reduce(min_dists, cost, 0, CV_REDUCE_AVG);
			if(iteration_count % 200 == 1)
			{
				std::cout << "iteration: " << iteration_count << " average distance: " << cost.at<float>(0, 0) << std::endl;
			}
			ReOrder(prev_home_cloud_, matched_target_cloud, indices);		
			transform_.CalcGradient(matched_target_cloud, cloud_, home_cloud_, feature_, iteration_count);				
			transform_.Update(iteration_count);			
			if(iteration_count % write_diagnosis_interval == 1)
				loader.SaveWeightsForDiagnosis(transform_, iteration_count / write_diagnosis_interval);
			
			
		}
		// copy to previous
		RecordingTrend(transform_, loader, trend_array, iteration_count, write_trend_interval, aim_idx);		
		// if(iteration_count == 0)
		home_cloud_.copyTo(prev_home_cloud_);
		// transform_.CopyTransformToPrev();
		
	}	

	pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_pcd(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud_pcd(new pcl::PointCloud<pcl::PointXYZ>);

	Mat2PCD(prev_home_cloud_, target_cloud_pcd);
	Mat2PCD(home_cloud_, transformed_cloud_pcd);

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer((new pcl::visualization::PCLVisualizer("cloud Viewer")));
	viewer->setBackgroundColor(0, 0, 0);	
	viewer->initCameraParameters();	

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_cloud_color(target_cloud_pcd, 0, 0, 255);			
	viewer->addPointCloud<pcl::PointXYZ>(target_cloud_pcd, target_cloud_color, "target_cloud");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> transformed_cloud_color(transformed_cloud_pcd, 0, 255, 0);
	viewer->addPointCloud<pcl::PointXYZ>(transformed_cloud_pcd, transformed_cloud_color, "transformed_cloud");	
	// viewer->updatePointCloud(home_cloud_, cloud_color, "original_cloud");
	viewer->spin();
}

void Explorer::TestTransformation(int single_frame_flag, int display_flag, int test_idx)
{
	char input_dir[400];
	char output_dir[400];	
	int aim_idx = 0;	
	int aim_frame_idx = 0;
	int num_gradient_iteration = 10;
	unsigned long iteration_count = 0;
	double depth_threshold = 0.8;
	double voxel_grid_size = 0.008;
	int cost_flag = 0;
	int write_trend_interval = 2000;
	int cloud_scale = 1;
	cv::Mat target_cloud, matched_target_cloud, transformed_cloud, indices, min_dists;
	cv::Mat cost = cv::Mat::zeros(1, 1, CV_32F);
	std::mt19937 engine(rd_());		
	std::vector<std::vector<double>> trend_array(num_trend_, std::vector<double>(0));

	Loader loader(num_weights_, dim_feature_, num_trend_, dir_id_, dir_);
	loader.FormatWeightsForTestDirectory();
	loader.FormatTrendDirectory();
	loader.LoadLearningRates(transform_);			
	// loader.LoadProprioception(num_train_data_, train_prop_, train_target_idx_, home_prop_);	
	loader.LoadProprioception(num_train_data_, num_test_data_, train_prop_, test_prop_, home_prop_, train_target_idx_, test_target_idx_);	
	loader.LoadWeightsForTest(transform_);
	cv::Mat size = cv::Mat::zeros(1, 1, CV_64F);

	int start_idx = 0;
	int end_idx = num_train_data_;
	if(single_frame_flag == 1)
	{
		start_idx = test_idx;
		end_idx = test_idx + 1;
	}

	int home_frame_idx = train_target_idx_.at<double>(0, 0);
	loader.LoadBinaryPointCloud(cloud_, home_frame_idx);
	cloud_ = cloud_ * cloud_scale_;
	SetFeature(feature_, aim_idx, train_prop_, home_prop_);
	transform_.CalcTransformInv(feature_);
	transform_.TransformDataInv(cloud_, home_cloud_, 1);		

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	if(display_flag == 1)
	{		
		viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer("cloud Viewer"));
		viewer->setBackgroundColor(0, 0, 0);	
		viewer->initCameraParameters();	
	}

	for(aim_idx = start_idx; aim_idx < end_idx; aim_idx++)
	{
		aim_frame_idx = train_target_idx_.at<double>(aim_idx, 0);
		loader.LoadBinaryPointCloud(cloud_, aim_frame_idx);
		cloud_ = cloud_ * cloud_scale_;
		SetFeature(feature_, aim_idx, train_prop_, home_prop_);
		transform_.CalcTransformInv(feature_);
		transform_.TransformData(home_cloud_, transformed_cloud, 1);
		if(display_flag == 1)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_pcd(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud_pcd(new pcl::PointCloud<pcl::PointXYZ>);

			cloud_ = cloud_ / cloud_scale_;
			transformed_cloud = transformed_cloud / cloud_scale_;

			Mat2PCD(cloud_, target_cloud_pcd);
			Mat2PCD(transformed_cloud, transformed_cloud_pcd);

			
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_cloud_color(target_cloud_pcd, 0, 0, 255);			
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> transformed_cloud_color(transformed_cloud_pcd, 0, 255, 0);
			if(aim_idx - start_idx == 0)
			{			
				viewer->addPointCloud<pcl::PointXYZ>(target_cloud_pcd, target_cloud_color, "target_cloud");			
				viewer->addPointCloud<pcl::PointXYZ>(transformed_cloud_pcd, transformed_cloud_color, "transformed_cloud");									
			}
			else
			{
				viewer->updatePointCloud<pcl::PointXYZ>(target_cloud_pcd, target_cloud_color, "target_cloud");
				viewer->updatePointCloud<pcl::PointXYZ>(transformed_cloud_pcd, transformed_cloud_color, "transformed_cloud");
			}
			if(end_idx - start_idx == 1)
				viewer->spin();
			else
				viewer->spinOnce(50);
		}
	}
	
}

void Explorer::ShowTransformationGrid(int num_grid)
{
	char input_dir[400];
	char output_dir[400];	
	int aim_idx = 0;	
	int aim_frame_idx = 0;
	int num_gradient_iteration = 10;
	unsigned long iteration_count = 0;
	double depth_threshold = 0.8;
	double voxel_grid_size = 0.008;
	int cost_flag = 0;
	int write_trend_interval = 2000;
	int cloud_scale = 1;
	cv::Mat target_cloud, matched_target_cloud, transformed_cloud, indices, min_dists;
	cv::Mat cost = cv::Mat::zeros(1, 1, CV_32F);
	std::mt19937 engine(rd_());		
	std::vector<std::vector<double>> trend_array(num_trend_, std::vector<double>(0));

	Loader loader(num_weights_, dim_feature_, num_trend_, dir_id_, dir_);
	loader.FormatWeightsForTestDirectory();
	loader.FormatTrendDirectory();
	loader.LoadLearningRates(transform_);			
	// loader.LoadProprioception(num_train_data_, train_prop_, train_target_idx_, home_prop_);		
	loader.LoadProprioception(num_train_data_, num_test_data_, train_prop_, test_prop_, home_prop_, train_target_idx_, test_target_idx_);	
	loader.LoadWeightsForTest(transform_);
	cv::Mat size = cv::Mat::zeros(1, 1, CV_64F);
	
	int home_frame_idx = train_target_idx_.at<double>(0, 0);
	loader.LoadBinaryPointCloud(cloud_, home_frame_idx);
	cloud_ = cloud_ * cloud_scale_;
	SetFeature(feature_, aim_idx, train_prop_, home_prop_);
	transform_.CalcTransformInv(feature_);
	transform_.TransformDataInv(cloud_, home_cloud_, 1);		

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	
	viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer("cloud Viewer"));
	viewer->setBackgroundColor(0, 0, 0);	
	viewer->initCameraParameters();	
	
	for(int i = 0; i < num_grid; i++)
	{
		for(int j = 0; j < num_grid; j++)
		{
			cv::Mat curr_prop = cv::Mat::zeros(1, num_joints_, CV_64F);
			curr_prop.at<double>(0, 0) = -1.0 + (0.5 + i) * 2.0 / num_grid;
			curr_prop.at<double>(0, 1) = -1.0 + (0.5 + j) * 2.0 / num_grid;
			curr_prop_matrix_ = repeat(curr_prop, num_train_data_, 1);
			prop_diff_ = train_prop_ - curr_prop_matrix_;
			prop_diff_ = prop_diff_.mul(prop_diff_);
			reduce(prop_diff_, prop_dist_, 1, CV_REDUCE_SUM);
			sortIdx(prop_dist_, aim_idx_matrix_, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);	
			aim_idx = (int)aim_idx_matrix_.at<int>(0, 0);

			aim_frame_idx = train_target_idx_.at<double>(aim_idx, 0);
			loader.LoadBinaryPointCloud(cloud_, aim_frame_idx);
			cloud_ = cloud_ * cloud_scale_;
			SetFeature(feature_, aim_idx, train_prop_, home_prop_);
			transform_.CalcTransformInv(feature_);
			transform_.TransformDataInv(cloud_, home_cloud_, 1);				
			// transform_.TransformData(home_cloud_, transformed_cloud, 1);
			
			pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud_pcd(new pcl::PointCloud<pcl::PointXYZ>);			
			transformed_cloud = home_cloud_ / cloud_scale_;			
			Mat2PCD(transformed_cloud, transformed_cloud_pcd);						
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> transformed_cloud_color(transformed_cloud_pcd, 0, 255, 0);

			char cloud_name[10];
			sprintf(cloud_name, "%d_%d", i, j);			
			viewer->addPointCloud<pcl::PointXYZ>(transformed_cloud_pcd, transformed_cloud_color, cloud_name);												
		}
	}
	viewer->spin();

}

void Explorer::RecordingTrend(Transform& transform, Loader& loader, std::vector<std::vector<double>>& trend_array, int iter, int write_trend_interval, double aim_idx)
{
	cv::Mat w = transform_.w();
	cv::Mat fisher_inv = transform_.fisher_inv();
	cv::Mat natural_grad = transform_.natural_w_grad();
	cv::Mat grad = transform_.w_grad();
	int trend_number = 15;
	trend_array[0].push_back(cv::norm(w.rowRange(0, 1), cv::NORM_L2)); trend_array[1].push_back(cv::norm(w.rowRange(1, 2), cv::NORM_L2));
	trend_array[2].push_back(cv::norm(w.rowRange(2, 3), cv::NORM_L2)); trend_array[3].push_back(cv::norm(w.rowRange(3, 4), cv::NORM_L2));      
	trend_array[4].push_back(cv::norm(w.rowRange(4, 5), cv::NORM_L2)); trend_array[5].push_back(cv::norm(w.rowRange(5, 6), cv::NORM_L2));		
	trend_array[6].push_back(cv::norm(w.rowRange(6, 7), cv::NORM_L2)); trend_array[7].push_back(cv::norm(w.rowRange(7, 8), cv::NORM_L2));	
	trend_array[8].push_back(cv::norm(w.rowRange(8, 9), cv::NORM_L2)); trend_array[9].push_back(cv::norm(w.rowRange(9, 10), cv::NORM_L2));
	trend_array[10].push_back(cv::norm(w.rowRange(10, 11), cv::NORM_L2)); trend_array[11].push_back(cv::norm(w.rowRange(11, 12), cv::NORM_L2));
	// trend_array[12].push_back(cv::norm(fisher_inv, cv::NORM_L2)); trend_array[13].push_back(cv::norm(natural_grad, cv::NORM_L2));
	// trend_array[12].push_back(cv::norm(grad, cv::NORM_L2));
	trend_array[12].push_back(aim_idx);
	trend_array[13].push_back(cv::norm(fisher_inv, cv::NORM_L2)); trend_array[14].push_back(cv::norm(natural_grad, cv::NORM_L2));
	/*trend_array[10].push_back(cv::norm(w.rowRange(10, 11), cv::NORM_L2)); trend_array[11].push_back(cv::norm(w.rowRange(11, 12), cv::NORM_L2));	
	trend_array[12].push_back(cv::norm(grad, cv::NORM_L2));*/
	if(iter % write_trend_interval == 0)
	{			
		int append_flag = iter == 0 ? 0 : 1;			
		loader.SaveTrend(trend_array, trend_number, append_flag);								
		for(int i = 0; i < trend_number; i++)
			trend_array[i].clear();
	}
	loader.SaveWeightsForTest(transform);
	/*cv::Mat fisher_inv = transform_.fisher_inv();
	cv::Mat natural_grad = transform_.natural_w_grad();*/
}

void Explorer::ReOrder(cv::Mat& input, cv::Mat& output, cv::Mat& input_indices)
{
	output = cv::Mat::zeros(input_indices.rows, input.cols, CV_64F);
	for(int p = 0; p < input_indices.rows; p++)
		for(int q = 0; q < input.cols; q++)
			output.at<double>(p, q) = input.at<double>(input_indices.at<int>(p, 0), q);				
}

void Explorer::PCD2Mat(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, cv::Mat& cloud_mat)
{
	int size = cloud->points.size();
	int dim = 4;
	cloud_mat = cv::Mat::zeros(size, dim, CV_64F);
	for(int i = 0; i < size; i++)
	{
		cloud_mat.at<double>(i, 0) = cloud->points[i].x;
		cloud_mat.at<double>(i, 1) = cloud->points[i].y;
		cloud_mat.at<double>(i, 2) = cloud->points[i].z;
		cloud_mat.at<double>(i, 3) = 1.0;
	}
}

void Explorer::Mat2PCD(cv::Mat& cloud_mat, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
	int size = cloud_mat.rows;
	std::vector<pcl::PointXYZ> points_vec(size);
	cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
	for(int i = 0; i < size; i++)
	{
		pcl::PointXYZ point;
		point.x = cloud_mat.at<double>(i, 0);
		point.y = cloud_mat.at<double>(i, 1);
		point.z = cloud_mat.at<double>(i, 2);
		cloud->push_back(point);
	}	
}

void Explorer::PreprocessingAndSavePointCloud()
{
	char input_dir[400];
	char output_dir[400];		
	unsigned long iteration_count = 0;
	double depth_threshold = 0.8;
	double voxel_grid_size = 0.010;
	int num_clouds = num_train_data_;
	std::mt19937 engine(rd_());		

	Loader loader(num_weights_, dim_feature_, num_trend_, dir_id_, dir_);
	loader.FormatWeightsForTestDirectory();
	loader.FormatTrendDirectory();
	loader.LoadLearningRates(transform_);
	// algorithms
	pcl::PassThrough<pcl::PointXYZ> pass;
	pcl::PCDReader reader;
	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;	
	
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	 /*pcl::PointCloud<pcl::PointXYZ>::Ptr prev_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	 pcl::PointCloud<pcl::PointXYZ>::Ptr home_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	 pcl::PointCloud<pcl::PointXYZ>::Ptr prev_home_cloud(new pcl::PointCloud<pcl::PointXYZ>);*/
	 pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_cloud(new pcl::PointCloud<pcl::PointXYZ>);	
	// point clouds		
	// boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer((new pcl::visualization::PCLVisualizer("cloud Viewer")));
	// viewer->setBackgroundColor(0, 0, 0);	
	// viewer->initCameraParameters();	
	for(iteration_count = 0; iteration_count < num_clouds; iteration_count++)
	{					
		loader.LoadPointCloud(cloud, reader, iteration_count); // load point cloud		
		DepthFiltering(depth_threshold, pass, cloud, tmp_cloud);
		DownSamplingPointCloud(voxel_grid_size, voxel_grid, tmp_cloud, cloud);
		loader.SavePointCloudAsBinaryMat(cloud, iteration_count);
		if(iteration_count % 100 == 1)
			std::cout << "iteration: " << iteration_count << std::endl;
	}
}

void Explorer::DepthFiltering(float depth, pcl::PassThrough<pcl::PointXYZ>& pass, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud)
{
	pass.setInputCloud(cloud);
	pass.setFilterFieldName ("z");
	pass.setFilterLimits(0.0, depth);
	pass.filter(*filtered_cloud);
}

void Explorer::DownSamplingPointCloud(double voxel_size, pcl::VoxelGrid<pcl::PointXYZ>& voxel_grid, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr down_sampled_cloud)
{
	voxel_grid.setInputCloud(cloud);
	voxel_grid.setLeafSize(voxel_size, voxel_size, voxel_size);
	voxel_grid.filter(*down_sampled_cloud);
}

void Explorer::ShowCloudSequence()
{
	int num_cloud = 4000;
	Loader loader(12, 3, 13, dir_id_, dir_);
	pcl::PCDReader reader;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer((new pcl::visualization::PCLVisualizer("cloud Viewer")));
	viewer->setBackgroundColor(0, 0, 0);	
	viewer->initCameraParameters();	
	for(int i = 0; i < num_cloud; i++)
	{
		loader.LoadPointCloud(cloud, reader, i);
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color(cloud, 0, 0, 255);	
		if(i == 0)
			viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_color, "original_cloud");	
		else
			viewer->updatePointCloud(cloud, cloud_color, "original_cloud");
		viewer->spinOnce(1);		
	}
	viewer->close();
}

void Explorer::GenerateLinePath(std::vector<std::vector<double>>& path, std::vector<double>& targets, std::vector<double>& prev_targets)
{		
	double max_speed = 8.0 / 20.0;
	double path_length = 0;
	for(int i = 0; i < num_joints_; i++)	
		path_length += sqrt(pow(targets[i] - prev_targets[i], 2));	
	
	int num_frame_path = (int)(path_length / max_speed) + 1;

	path = std::vector<std::vector<double>>(num_joints_, std::vector<double>(num_frame_path));

	for(int i = 0; i < num_joints_; i++)	
		for(int j = 0; j < num_frame_path; j++)		
			path[i][j] = prev_targets[i] + (targets[i] - prev_targets[i]) * (j + 1) / num_frame_path;
	

}

int Explorer::GenerateAimIndexLinePath(std::mt19937& engine, int current_iteration)
{
	int aim_idx = 0;
	double current_range = 0;
	
	current_range = starting_exploration_range_ + (max_exploration_range_ - starting_exploration_range_) * current_iteration / range_expanding_period_;	
	current_range = current_range > max_exploration_range_ ? max_exploration_range_ : current_range;
	std::uniform_real_distribution<double> uniform(-1.0 * current_range, 1.0 * current_range);	  	
	
	// generate path
	if(path_count_ == 0)
	{
		for(int i = 0; i < num_joints_; i++)
			targets_[i] = uniform(engine);
		
		GenerateLinePath(path_, targets_, prev_targets_);
		for(int i = 0; i < num_joints_; i++)
			prev_targets_[i] = targets_[i];
		
		path_count_ = path_[0].size();
	}

	for(int i = 0; i < num_joints_; i++)
		curr_prop_.at<double>(0, i) = path_[i][path_[0].size() - path_count_];	
	path_count_--;
	
	curr_prop_matrix_ = repeat(curr_prop_, num_train_data_, 1);
	prop_diff_ = train_prop_ - curr_prop_matrix_;
	prop_diff_ = prop_diff_.mul(prop_diff_);
	reduce(prop_diff_, prop_dist_, 1, CV_REDUCE_SUM);
	sortIdx(prop_dist_, aim_idx_matrix_, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);	
	aim_idx = (int)aim_idx_matrix_.at<int>(0, 0);

	return aim_idx;

}

void Explorer::SetFeature(cv::Mat& feature, int aim_idx, cv::Mat& prop, cv::Mat& home_prop)
{
	cv::Mat delta_p = cv::Mat::zeros(num_joints_, 1, CV_64F);
	for(int i = 0; i < num_joints_; i++)
		delta_p.at<double>(i, 0) = prop.at<double>(aim_idx, i) - home_prop.at<double>(0, i);

	double current_data = 1.0;
	double* p_current_data = &current_data;
	int kernel_dim = 3;
	int curr_pos = 0;	
	kernel_list_.clear();	
	SetKernel(kernel_list_, delta_p, p_current_data, kernel_dim, curr_pos, num_joints_, kernel_dim, 0); 
	double sum = 0;
	// switch off the bias...
	for(int i = 1; i < kernel_list_.size(); i++)
		feature.at<double>(i - 1, 0) = kernel_list_[i];
}

void Explorer::SetKernel(std::vector<double>& kernel_list, cv::Mat& data, double* p_current_data, int dim_left, int curr_pos, int data_length, int kernel_dim, int value_flag)
{
	int pos = curr_pos;
	for(int dim = 0; dim <= dim_left; dim++)
	{
		if(pos == 0 && dim == 0)
		{			
			kernel_list.push_back(0.0); // bias
			// int next_pos = pos + 1;
			int next_pos = pos < data_length - 1 ? pos + 1 : pos;
			double tmp_data = *p_current_data;
			if(kernel_dim != 0 && next_pos != 0)
				SetKernel(kernel_list, data, p_current_data, kernel_dim , next_pos, data_length, kernel_dim, value_flag);
			*p_current_data = tmp_data;
		}
		else if(dim == 0)
		{
			int next_pos = pos < data_length - 1 ? pos + 1 : pos;
			double tmp_data = *p_current_data;
			int actual_dim_left = dim_left - dim;
			if(kernel_dim != 0 && pos != next_pos)
				SetKernel(kernel_list, data, p_current_data, actual_dim_left , next_pos, data_length, kernel_dim, value_flag);
			*p_current_data = tmp_data;
		}
		else if(dim != 0)
		{
			*p_current_data = (*p_current_data) * data.at<double>(pos, 0); // pow(data[pos], (double)dim);
			kernel_list.push_back(*p_current_data);
			int next_pos = pos < data_length - 1 ? pos + 1 : pos;
			int actual_dim_left = dim_left - dim;
			double tmp_data = *p_current_data;
			if(actual_dim_left != 0 && pos != next_pos)
				SetKernel(kernel_list, data, p_current_data, actual_dim_left, next_pos, data_length, kernel_dim, value_flag);
			*p_current_data = tmp_data;
		}
	}
}

// if(iteration_count % 200 == 1)
//{
//	std::cout << cost.at<double>(0, 0) << std::endl;
//	if(iteration_count == 1)
//	{
//		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> prev_home_cloud_color(prev_home_cloud_, 0, 0, 255);			
//		viewer->addPointCloud<pcl::PointXYZ>(prev_home_cloud_, prev_home_cloud_color, "prev_home_cloud");			
//		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> home_cloud_color(home_cloud_, 0, 255, 0);			
//		viewer->addPointCloud<pcl::PointXYZ>(home_cloud_, home_cloud_color, "home_cloud");	
//		// viewer->updatePointCloud(cloud_, cloud_color, "original_cloud");
//		viewer->spinOnce(500);
//	}
//	else
//	{
//		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> prev_home_cloud_color(prev_home_cloud_, 0, 0, 255);			
//		viewer->updatePointCloud(prev_home_cloud_, prev_home_cloud_color, "prev_home_cloud");			
//		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> home_cloud_color(home_cloud_, 0, 255, 0);			
//		viewer->updatePointCloud(home_cloud_, home_cloud_color, "home_cloud");	
//		viewer->spinOnce(500);
//	}
//}

//if(iteration_count == 1)
//{
//	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color(prev_home_cloud_, 0, 0, 255);			
//	viewer->addPointCloud<pcl::PointXYZ>(prev_home_cloud_, cloud_color, "original_cloud");			
//	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> prev_cloud_color(home_cloud_, 0, 255, 0);			
//	viewer->addPointCloud<pcl::PointXYZ>(home_cloud_, prev_cloud_color, "target_cloud");	
//	// viewer->updatePointCloud(home_cloud_, cloud_color, "original_cloud");
//	viewer->spin();
//}

//aim_frame_idx = train_target_idx_.at<double>(aim_idx, 0); // generate aim index...		
			//loader.LoadBinaryPointCloud(home_cloud_mat_, aim_frame_idx);			
			//std::cout << aim_frame_idx << std::endl;

			//Mat2PCD(cloud_mat_, cloud_);
			//Mat2PCD(home_cloud_mat_, home_cloud_);

			//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer((new pcl::visualization::PCLVisualizer("cloud Viewer")));
			//viewer->setBackgroundColor(0, 0, 0);	
			//viewer->initCameraParameters();	

			//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color(cloud_, 0, 0, 255);			
			//viewer->addPointCloud<pcl::PointXYZ>(cloud_, cloud_color, "original_cloud");
			//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> prev_cloud_color(home_cloud_, 0, 255, 0);			
			//viewer->addPointCloud<pcl::PointXYZ>(home_cloud_, prev_cloud_color, "target_cloud");	
			//// viewer->updatePointCloud(home_cloud_, cloud_color, "original_cloud");
			//viewer->spin();


//void Explorer::LearningFromPointCloudTest()
//{
//	char input_dir[400];
//	int feature_dim = 4;
//	int transform_dim = 4;
//	int cloud_size = 0;
//	int iteration = 100;
//	int gradient_iteration = 100;
//	int write_trend_interval = 10000;
//	int trend_number = 15;
//	std::vector<std::vector<double>> trend_array(trend_number, std::vector<double>(0));
//	pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_pcd(new pcl::PointCloud<pcl::PointXYZ>);
//	pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud_pcd(new pcl::PointCloud<pcl::PointXYZ>);
//	cv::Mat feature = cv::Mat::zeros(feature_dim, 1, CV_64F);
//	cv::Mat target_cloud, matched_target_cloud, query_cloud, transformed_query_cloud, target_cloud_f, transformed_query_cloud_f, prediction_cloud;
//	cv::Mat size, indices, min_dists_f, min_dists, dist;	
//	size = cv::Mat::zeros(1, 1, CV_64F);
//	dist = cv::Mat::zeros(1, 1, CV_64F);
//
//	Loader loader(num_weights_, dim_feature_, trend_number, dir_id_, dir_);
//	loader.FormatWeightsForTestDirectory();
//	loader.FormatTrendDirectory();
//
//	sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_learning/test/feature.bin");
//	FileIO::ReadMatDouble(feature, feature_dim, 1, input_dir);
//		
//	// load target cloud
//	sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_learning/test/target_cloud_size.bin");
//	FileIO::ReadMatDouble(size, 1, 1, input_dir);
//	cloud_size = size.at<double>(0, 0);
//	target_cloud = cv::Mat::zeros(cloud_size, transform_dim, CV_64F);
//	sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_learning/test/target_cloud.bin");
//	FileIO::ReadMatDouble(target_cloud, cloud_size, transform_dim, input_dir);
//
//	// load query cloud
//	sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_learning/test/query_cloud_size.bin");
//	FileIO::ReadMatDouble(size, 1, 1, input_dir);
//	cloud_size = size.at<double>(0, 0);
//	query_cloud = cv::Mat::zeros(cloud_size, transform_dim, CV_64F);
//	matched_target_cloud = cv::Mat::zeros(cloud_size, transform_dim, CV_64F);
//	sprintf(input_dir, "D:/Document/HKUST/Year 5/Research/Solutions/unified_learning/test/query_cloud.bin");
//	FileIO::ReadMatDouble(query_cloud, cloud_size, transform_dim, input_dir);
//
//	indices = cv::Mat::zeros(query_cloud.rows, 1, CV_32S);
//	min_dists_f = cv::Mat::zeros(query_cloud.rows, 1, CV_32F);
//	min_dists = cv::Mat::zeros(query_cloud.rows, 1, CV_64F);
//	target_cloud.convertTo(target_cloud_f, CV_32F);	
//	cv::flann::Index kd_trees(target_cloud_f, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN);
//
//	query_cloud.copyTo(prediction_cloud);
//	for(int i = 0; i < iteration; i++)
//	{
//		prediction_cloud.convertTo(transformed_query_cloud_f, CV_32F);
//		kd_trees.knnSearch(transformed_query_cloud_f, indices, min_dists_f, 1);
//		min_dists_f.convertTo(min_dists, CV_64F);	
//		cv::reduce(min_dists, dist, 0, CV_REDUCE_AVG);
//		if(i % 2 == 0)
//			std::cout << "average distance: " << dist.at<double>(0, 0) << std::endl;
//		// copy...
//		for(int p = 0; p < query_cloud.rows; p++)
//			for(int q = 0; q < transform_dim; q++)
//				matched_target_cloud.at<double>(p, q) = target_cloud.at<double>(indices.at<int>(p, 0), q);				
//		for(int j = 0; j < gradient_iteration; j++)
//		{
//			transform_.CalcTransformInv(feature);
//			transform_.TransformDataInv(query_cloud, prediction_cloud, 1);
//			transform_.CalcGradient(matched_target_cloud, query_cloud, prediction_cloud, feature);
//			if(i > 8000)
//				transform_.Update();
//
//			cv::Mat w = transform_.w();
//			cv::Mat fisher_inv = transform_.fisher_inv();
//			cv::Mat natural_grad = transform_.natural_w_grad();
//			cv::Mat grad = transform_.w_grad();
//			trend_array[0].push_back(cv::norm(w.rowRange(0, 1), cv::NORM_L2)); trend_array[1].push_back(cv::norm(w.rowRange(1, 2), cv::NORM_L2));      
//			trend_array[2].push_back(cv::norm(w.rowRange(2, 3), cv::NORM_L2)); trend_array[3].push_back(cv::norm(w.rowRange(3, 4), cv::NORM_L2));      
//			trend_array[4].push_back(cv::norm(w.rowRange(4, 5), cv::NORM_L2)); trend_array[5].push_back(cv::norm(w.rowRange(5, 6), cv::NORM_L2));		
//			trend_array[6].push_back(cv::norm(w.rowRange(6, 7), cv::NORM_L2)); trend_array[7].push_back(cv::norm(w.rowRange(7, 8), cv::NORM_L2));	
//			trend_array[8].push_back(cv::norm(w.rowRange(8, 9), cv::NORM_L2)); trend_array[9].push_back(cv::norm(w.rowRange(9, 10), cv::NORM_L2));
//			trend_array[10].push_back(cv::norm(w.rowRange(10, 11), cv::NORM_L2)); trend_array[11].push_back(cv::norm(w.rowRange(11, 12), cv::NORM_L2));
//			trend_array[12].push_back(cv::norm(fisher_inv, cv::NORM_L2)); trend_array[13].push_back(cv::norm(natural_grad, cv::NORM_L2));
//			trend_array[14].push_back(cv::norm(grad, cv::NORM_L2));
//			int k = i * gradient_iteration + j;
//			if(k % write_trend_interval == 0)
//			{			
//				int append_flag = k == 0 ? 0 : 1;			
//				loader.SaveTrend(trend_array, trend_number, append_flag);								
//				for(int i = 0; i < trend_number; i++)
//					trend_array[i].clear();
//			}
//		}
//		// transform_.TransformDataInv(query_cloud, transformed_query_cloud, 1);
//		// transform_.set_fisher_inv();
//	}
//
//	Mat2PCD(target_cloud, target_cloud_pcd);
//	Mat2PCD(prediction_cloud, transformed_cloud_pcd);
//
//	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer((new pcl::visualization::PCLVisualizer("cloud Viewer")));
//	viewer->setBackgroundColor(0, 0, 0);	
//	viewer->initCameraParameters();	
//
//	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_cloud_color(target_cloud_pcd, 0, 0, 255);			
//	viewer->addPointCloud<pcl::PointXYZ>(target_cloud_pcd, target_cloud_color, "target_cloud");
//	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> transformed_cloud_color(transformed_cloud_pcd, 0, 255, 0);
//	viewer->addPointCloud<pcl::PointXYZ>(transformed_cloud_pcd, transformed_cloud_color, "transformed_cloud");	
//	// viewer->updatePointCloud(home_cloud_, cloud_color, "original_cloud");
//	viewer->spin();
//
//	
//	/*min_dists_f.convertTo(min_dists, CV_64F);	
//	cv::reduce(min_dists, dist, 0, CV_REDUCE_AVG);
//	std::cout << "average distance: " << dist.at<double>(0, 0) << std::endl;*/
//
//	// show kd tree result here...
//
//	/*for(int i = 0; i < 100; i++)
//	{
//		std::cout << i << " " << indices.at<int>(i, 0) << " " << query_cloud.at<double>(i, 0) << " " << query_cloud.at<double>(i, 1) << " " << query_cloud.at<double>(i, 2) 
//			<< " " << target_cloud.at<double>(i, 0) << " " << target_cloud.at<double>(i, 1) << " " << target_cloud.at<double>(i, 2) << std::endl;
//	}*/
//
//	std::cout << "test finished..." << std::endl;
//
//}