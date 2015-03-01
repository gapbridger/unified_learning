#include "../inc/explorer.h"
#include "../inc/transform.h"

int main(int argc, char* argv[])
{
	// parameters...
	char input_dir[400];
	int num_iteration = 100000; 
	int single_frame_flag = 0;
	int display_flag = 0;
	int test_flag = 0;
	int test_idx = 0;
	int target_idx = 0;		
	int expanding_period = num_iteration / 2;
	int directory_id = atoi(argv[1]);
	int train_flag = atoi(argv[2]);
	char dir[40];
	sprintf(dir, "january_03_2014");		

	//Transform transform(initial_x, initial_y, initial_long_axis, initial_short_axis, initial_angle);
	//transform.CheckInvGradient();
	
	if(argc == 4)
	{
		if(train_flag == 1)
			num_iteration = atoi(argv[3]);			
		else if(train_flag == 0)
			single_frame_flag = atoi(argv[3]);
		else if(train_flag == 2)
			sprintf(dir, argv[3]);
		else if(train_flag == 3)		
			sprintf(dir, argv[3]);			
		else
			return 0;
	}
	if(argc == 5)
	{
		if(train_flag == 1)
		{
			sprintf(dir, argv[3]);
			num_iteration = atoi(argv[4]);
			// expanding_period = atof(argv[4]);
		}
		else if(train_flag == 0)
		{
			single_frame_flag = atoi(argv[3]);
			display_flag = atoi(argv[4]);
		}
		else if(train_flag == 3)
		{
			sprintf(dir, argv[3]);
			test_idx = atoi(argv[4]);			
		}
		else
			return 0;
		
	}
	if(argc == 6){
		if(train_flag == 1)
		{
			sprintf(dir, argv[3]);
			num_iteration = atoi(argv[4]);
			expanding_period = atof(argv[5]);
		}
		//else if(train_flag == 0)
		//{
		//	sprintf(dir, argv[3]);
		//	single_frame_flag = atoi(argv[3]);
		//	display_flag = atoi(argv[4]);
		//	// test_idx = atoi(argv[5]);
		//}
		else
			return 0;
	}
	if(argc == 7){
		if(train_flag == 0)
		{
			sprintf(dir, argv[3]);
			single_frame_flag = atoi(argv[4]);
			display_flag = atoi(argv[5]);
			// test_flag = atoi(argv[6]);
			test_idx = atoi(argv[6]);
		}
		else if(train_flag == 1)
		{			
			sprintf(dir, argv[3]);
			num_iteration = atoi(argv[4]);
			expanding_period = atof(argv[5]);
			// load_all_flag = atoi(argv[6]);		
		}
		else
			return 0;
	}
	if(argc == 8){
		if(train_flag == 0)
		{
			sprintf(dir, argv[3]);
			single_frame_flag = atoi(argv[4]);
			display_flag = atoi(argv[5]);
			test_flag = atoi(argv[6]);
			test_idx = atoi(argv[7]);
		}
		else
			return 0;
	}
	
	
	// Explorer explorer(directory_id, num_iteration, expanding_period, dir, 4, 3, 1);	
	Explorer explorer(directory_id, num_iteration, expanding_period, dir, 4, 9, 2);
	
	if(train_flag == 1)
		explorer.LearningFromPointCloud();
	else if(train_flag == 0)
	{
		explorer.TestTransformation(single_frame_flag, display_flag, test_idx);
	}
	if(train_flag == 3)
	{
		// explorer.PreprocessingAndSavePointCloud();
		explorer.ShowTransformationGrid(12);
		// Transform transform(4, 3);
		// transform.CheckInvGradient();
		// explorer.ShowCloudSequence();
	
	}

}
