#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <pthread.h>

// Multi Thread -- pthread -- 6 cores, 2 threads per core -> 12 threads?? divide set by 12
#define NUM_THREADS 12

/*
run this command each time this file is changed
cc -fPIC -shared -Ofast -lpthread -o cosq_funct.so cosq_funct.c 

*/
// create function for training/fitting the scalar quantizer

struct partition_thread_data
{
    float* point_arr;
    int    point_length;
    float* centroid_arr;
    int    centroid_length;
    int*   partition_map_arr;
    float  epsilon;
    int    bits;
};

inline float calc_transition_probabilities(unsigned int input, unsigned int output, float epsilon, int bits){
    float transition_prob = 1;

    for (int i=0; i<bits; i++){
        if ((input & 0x1U) == (output & 0x1U))
            transition_prob *= (1-epsilon);
	else
            transition_prob *= epsilon;
	input >>= 1;
	output >>= 1;
    }
    return transition_prob;
}

inline double expected_distortion(float centroid_array[], int array_length, unsigned int index, float point, float epsilon, int bits){
    double distortion = 0;
    float trans_prob = 0;

    for(unsigned int i =0; i<array_length; i++){
        trans_prob = calc_transition_probabilities(index, i, epsilon, bits);
        distortion += trans_prob * (centroid_array[i] - point) * (centroid_array[i] - point);
    }
    return distortion;
}

void* calc_partition_thread(void* thread_data)
{
    // Grab data from pointer
    struct partition_thread_data *data = thread_data;
    //printf("Thread number %ld starting!\n", pthread_self());

    double global_min = 0;
    int index_min_dis = 0;
    float current_exp_dist = 0;
    unsigned int i, j = 0;
    for(i=0; i < data->point_length; i++){
    	global_min = 0;
    	index_min_dis = 0;
        for(j=0; j < data->centroid_length; j++){
            current_exp_dist = expected_distortion(data->centroid_arr, data->centroid_length, j, data->point_arr[i], data->epsilon, data->bits);
            if(j==0){
                global_min = current_exp_dist;
                continue;
            }
            if (current_exp_dist<global_min){
                global_min = current_exp_dist;
                index_min_dis = j;
            }
        }
        data->partition_map_arr[i] = index_min_dis;
    }
    //printf("Thread number %ld ending!\n", pthread_self());
}

void calc_partitions(float* point_array, int point_length, float* centroid_array, int centroid_length, float epsilon, int bits, int* return_array){
    // array that has corresponding to point array, contains centroid index value for the point

    pthread_t thread_ids[NUM_THREADS];
    struct partition_thread_data *thread_data = malloc(NUM_THREADS * sizeof(struct partition_thread_data));

    // Prepare data for threads
    int points_per_thread = point_length / NUM_THREADS; 
    for(int i = 0; i < NUM_THREADS; i++)
    {
	thread_data[i].point_arr = &point_array[points_per_thread * i];
	thread_data[i].partition_map_arr = &return_array[points_per_thread * i]; 
	if (i != NUM_THREADS-1)
	    thread_data[i].point_length = points_per_thread;
	else
	    thread_data[i].point_length = point_length - points_per_thread * i;
	thread_data[i].centroid_arr = centroid_array; 
	thread_data[i].centroid_length = centroid_length; 
	thread_data[i].epsilon = epsilon;
	thread_data[i].bits = bits;
    }

    // Call Threads
    for(int i = 0; i < NUM_THREADS; i++)
        pthread_create(&thread_ids[i], NULL, calc_partition_thread, (void*)&thread_data[i]);

    // Join Threads
    for(int i = 0; i < NUM_THREADS; i++)
        pthread_join(thread_ids[i], NULL);

    free(thread_data);
}

void calc_centroids(int* partition_arr, int partition_arr_len, float* centroid_arr, int centroid_arr_len, float* point_arr, int point_arr_len, float epsilon, int bits, float* return_arr){
    double sum_array[centroid_arr_len];
    int sum_array_count[centroid_arr_len];
    memset(sum_array, 0, centroid_arr_len*sizeof(double));
    memset(sum_array_count, 0, centroid_arr_len*sizeof(int));

    if(partition_arr_len!=point_arr_len){
        printf("uwu i made a fucky");
    }

    for(int i = 0; i<partition_arr_len; i++){
        sum_array[partition_arr[i]]+=point_arr[i];
        sum_array_count[partition_arr[i]]++;
    }

    double numerator, denominator;
    float trans_prob;

    for(int i=0; i<centroid_arr_len; i++){
        numerator = 0;	
        denominator = 0;	
        for(int j=0; j<centroid_arr_len; j++){
            trans_prob = calc_transition_probabilities(j,i, epsilon, bits);
	    numerator += trans_prob*sum_array[j];
	    denominator += trans_prob*sum_array_count[j];
        }
	if(denominator != 0)
            return_arr[i] = numerator/denominator;
	else
	    return_arr[i] = centroid_arr[i];	    
    }            
}

float calc_distortion(float* centroids, int centroid_len, float* training_points, int* partitions, int training_point_len, float epsilon, int bits)
{
    double distortion = 0;
    for(int i =  0; i < centroid_len; i++)
    {
        for(int j = 0; j < centroid_len; j++)
	{
	    float transition_prob = calc_transition_probabilities(i, j, epsilon, bits);
	    for(int k = 0; k < training_point_len; k++)
	    	if (partitions[k] == i) 
		    distortion += transition_prob * (training_points[k] - centroids[j]) * (training_points[k] - centroids[j]);
	}
    }
    return distortion / training_point_len;
}

float * iteration(float* centroids, int centroid_len, float* training_points, int training_point_len, int count, float epsilon, int bits, int print_distortion){
    int* point2centroid = malloc(training_point_len * sizeof(int));
    memset(point2centroid, 0, training_point_len*sizeof(int));

    float* curr_centroids = malloc(centroid_len * sizeof(float));
    float* prev_centroids = malloc(centroid_len * sizeof(float));
    memcpy(curr_centroids, centroids, centroid_len * sizeof(float));

    calc_partitions(training_points, training_point_len, curr_centroids, centroid_len, epsilon, bits, point2centroid);
    float curr_distortion = calc_distortion(curr_centroids, centroid_len, training_points, point2centroid, training_point_len, epsilon, bits);
    float distortion = 0;
    int i = 0;
    do
    {
        memcpy(prev_centroids, curr_centroids, centroid_len * sizeof(float));
	distortion = curr_distortion;
    	calc_centroids(point2centroid, training_point_len, prev_centroids, centroid_len, training_points, training_point_len, epsilon, bits, curr_centroids);
    	calc_partitions(training_points, training_point_len, curr_centroids, centroid_len, epsilon, bits, point2centroid);
	curr_distortion = calc_distortion(curr_centroids, centroid_len, training_points, point2centroid, training_point_len, epsilon, bits); 
	if (i % 10 == 0)
	    printf("Iteration: %d, distortion: %f, current distortion: %f\n", i, distortion, curr_distortion);
    } while((fabs((distortion - curr_distortion)/distortion) > 0.0001) && (++i < 125));
    //} while((distortion > curr_distortion) && (++i < 200));

    free(point2centroid);
    free(curr_centroids);

    printf("Distortion bits: %d, Epsilon: %f, distortion: %f\n", bits, epsilon, distortion);
    // applying mini batch gradient decent in python code
    return prev_centroids;
}

