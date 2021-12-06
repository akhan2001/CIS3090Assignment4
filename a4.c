#define PROGRAM_FILE "a4.cl"
#define KERNEL_FUNC "a4"
#define ARRAY_SIZE 10

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

void setConf(int conf, int size, int numKernels, int *coords, char **gol);
void playGoL(int size, int numKernels, int *coords, char **gol);
void splitWork(int lineLength, int numKernels, int **arr);
void printGoL(int size, char **gol);
char displayNum (int *coords, int yIndex, int numCoords);
void destroy_gol(char **gol, int size);

/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device() {

   cl_platform_id platform;
   cl_device_id dev;
   int err;

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
      perror("Couldn't identify a platform");
      exit(1);
   }

   /* Access a device */
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
   }
   if(err < 0) {
      perror("Couldn't access any devices");
      exit(1);
   }

   return dev;
}

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   int err;

   /* Read program file and place content into buffer */
   program_handle = fopen(filename, "r");
   if(program_handle == NULL) {
      perror("Couldn't find the program file");
      exit(1);
   }
   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char*)malloc(program_size + 1);
   program_buffer[program_size] = '\0';
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);

   /* Create program from file */
   program = clCreateProgramWithSource(ctx, 1,
      (const char**)&program_buffer, &program_size, &err);
   if(err < 0) {
      perror("Couldn't create the program");
      exit(1);
   }
   free(program_buffer);

   /* Build program */
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if(err < 0) {

      /* Find size of log and print to std output */
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
            0, NULL, &log_size);
      program_log = (char*) malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
            log_size + 1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }

   return program;
}

int main(int argc, char *argv[]) {

	int numWorkers = 1; //num kernels
	int size = 20; //size
	int conf = 0; //init config

	int k,m;

	/* Initializing for command-line arguments */
	if(argc > 1){
		for(k = 1; k < argc - 1; k++){
			if(strcmp(argv[k], "-n") == 0){
				numWorkers = atoi(argv[k + 1]);
			}
			else if(strcmp(argv[k], "-s") == 0){
				size = atoi(argv[k + 1]);
			}
			else if(strcmp(argv[k], "-i") == 0){
				conf = atoi(argv[k + 1]);
			}
		}
	}
	/* Error-check if all arguments are in expected range */
	if (numWorkers < 1 || size < 1 || conf < 0 || conf > 4)
	{
		printf("Usage - oclgrind ./a4 <-n #> <-s #> <-i #>\n");
		printf("Where i must be from 0 to 4 inclusive.\n");
		exit(0);
	}

	/* set the coordinates that each worker will work at */
	int *coords = (int *)malloc(sizeof(int) * numWorkers);
	splitWork(size, numWorkers, &coords);

	/* set the display array and set each character to space */
	char **gol = (char **)malloc(sizeof(char *) * size);
	for(k = 0; k < size; k++){
		gol[k] = (char *)malloc(sizeof(char) * size);
		for(m = 0; m < size; m++){
			gol[k][m] = ' ';
		}
	}

	/* Set the first line of the display array according to the configuration */
	setConf(conf, size, numWorkers, coords, gol);

   /* OpenCL structures */
   /* ---- Don't Touch ---- */
   cl_device_id device;
   cl_context context;
   cl_program program;
   cl_kernel kernel;
   cl_command_queue queue;
   cl_int err;
   /* ---- ____________ ---- */

   size_t local_size, global_size;

   /* Create device and context */
   /* ---- Don't Touch ---- */
   device = create_device();
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a context");
      exit(1);
   }
   /* Build program */
   program = build_program(context, device, PROGRAM_FILE);
   /* ---- __________ ---- */
/* ---- Don't Touch ---- */
   /* Create a command queue */
   queue = clCreateCommandQueue(context, device, 0, &err);
   if(err < 0) {
      perror("Couldn't create a command queue");
      exit(1);
   };
   /* Create a kernel */
   kernel = clCreateKernel(program, KERNEL_FUNC, &err);
   if(err < 0) {
      perror("Couldn't create a kernel");
      exit(1);
   };
/* ---- ____________ ---- */

	   /* Create data buffer */
//    Edit this to just put in the array size and output the start,end values of each work_item

   /* Data and buffers */
   cl_mem input_buffer;
   cl_mem output_buffer;
   global_size = numWorkers;
   local_size = numWorkers;

   cl_int clsize = size;

	for (k = 1; k < size; k++)
	{
		char *row_above = gol[k-1];
		input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY |
				CL_MEM_COPY_HOST_PTR, size * sizeof(char), row_above, &err);
		if(err < 0) {
			perror("Couldn't create a buffer");
			exit(1);
		};
		char *curr_row = gol[k];
		output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
				CL_MEM_COPY_HOST_PTR, size * sizeof(char), curr_row, &err);
		if(err < 0) {
			perror("Couldn't create a buffer");
			exit(1);
		};

		/* Create kernel arguments */
		err = clSetKernelArg(kernel, 0, sizeof(cl_int), &clsize);
		err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_buffer);
		err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_buffer);
		if(err < 0) {
			perror("Couldn't create a kernel argument");
			exit(1);
		}

		/* Enqueue kernel */
		err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size,
				&local_size, 0, NULL, NULL);
		if(err < 0) {
			perror("Couldn't enqueue the kernel");
			exit(1);
		}

		/* Read the kernel's output */
		err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0,
				sizeof(char)*size, curr_row, 0, NULL, NULL);
		if(err < 0) {
			perror("Couldn't read the buffer");
			exit(1);
		}

		memcpy(gol[k], curr_row, sizeof(char)*(size));
		clReleaseMemObject(input_buffer);
		clReleaseMemObject(output_buffer);
	}

	printGoL(size, gol);

	/* Deallocate resources */
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseContext(context);
	destroy_gol(gol, size);

   return 0;
}

void setConf(int conf, int size, int numKernels, int *coords, char **gol){
	int i = 0;
	time_t t;

	int x = 0;
	int y = 0;
	int z = 0;

	srand((unsigned) time(&t));

	if(conf == 0){
		int temp;
		for(i = 0; i < size; i++){ //random
			temp = rand() % 2;
			if(temp == 0){
				gol[0][i] = displayNum(coords, i, numKernels);
			}
			else{
				gol[0][i] = ' ';
			}
		}
	}
	else if(conf == 1){
		x = 1;
		y = 0;
		for(i = size / 2 - 2; i < size; i++){ //flip flop
			if(i < 0){
				i = 0;
			}
			if(x == 1 && y == 0){
				gol[0][i] = displayNum(coords, i, numKernels);
				x = 0;
				y = 1;
			}
			else if(x == 0 && y == 1){
				gol[0][i] = ' ';
				x = 0;
				y = 0;
			}
			else if(x == 0 && y == 0){
				gol[0][i] = displayNum(coords, i, numKernels);
				x = 1;
				y = 1;
			}
			else if(x == 1 && y == 1){
				gol[0][i] = displayNum(coords, i, numKernels);
				x = 0;
				y = 2;
			}
		}
	}
	else if(conf == 2){
		int count = 0;
		for(i = size / 2 - 3; i < size && count < 6; i++){ //spider
			if(i < 0){
				i = 0;
			}
			count = count + 1;
			gol[0][i] = displayNum(coords, i, numKernels);
		}
	}
	else if(conf == 3){
		x = 1;
		y = 0;
		z = 0;
		for(i = size / 2 - 3; i < size; i++){ //glider
			if(i < 0){
				i = 0;
			}
			if(x == 1 && y == 0 && z == 0){
				gol[0][i] = displayNum(coords, i, numKernels);
				x = 0;
				y = 1;
				z = 0;
			}
			else if(x == 0 && y == 1 && z == 0){
				gol[0][i] = ' ';
				x = 1;
				y = 0;
				z = 1;
			}
			else if(x == 1 && y == 0 && z == 1){
				gol[0][i] = displayNum(coords, i, numKernels);
				x = 1;
				y = 1;
				z = 0;
			}
			else if(x == 1 && y == 1 && z == 0){
				gol[0][i] = displayNum(coords, i, numKernels);
				x = 1;
				y = 1;
				z = 1;
			}
			else if(x == 1 && y == 1 && z == 1){
				gol[0][i] = displayNum(coords, i, numKernels);
				x = 0;
				y = 1;
				z = 2;
			}
		}
	}
	else if(conf == 4){
		int count = 0;
		for(i = size / 2 - 3; i < size && count < 7; i++){ //face
			if(i < 0){
				i = 0;
			}
			count = count + 1;
			gol[0][i] = displayNum(coords, i, numKernels);
		}
	}
}

void playGoL(int size, int numKernels, int *coords, char **gol){
	int col = 0;
  	int level = 1;
	char out = (char) 48;
	int ncount = 0;

  for(level = 1; level < size; level ++){
    for(col = 0; col < size; col++){
			ncount = 0;
			if(col - 2 >= 0){
				if(gol[level - 1][col - 2] != ' '){
					ncount = ncount + 1;
				}
			}
			if(col - 1 >= 0){
				if(gol[level - 1][col - 1] != ' '){
					ncount = ncount + 1;
				}
			}
			if(col + 1 < size){
				if(gol[level - 1][col + 1] != ' '){
					ncount = ncount + 1;
				}
			}
			if(col + 2 < size){
				if(gol[level - 1][col + 2] != ' '){
					ncount = ncount + 1;
				}
			}

      if(gol[level - 1][col] != ' '){
        if(ncount == 2 || ncount == 4){
          gol[level][col] = out;
        }
        else{
          gol[level][col] = ' ';
        }
      }
      else{
        if(ncount == 2 || ncount == 3){
          gol[level][col] = out;
        }
        else{
          gol[level][col] = ' ';
        }
			}
		}
  }
}

void splitWork(int lineLength, int numKernels, int **arr){
	int *retArr;
	int i = 0;

	retArr = *arr;

	for(i = 0; i < numKernels; i++){
		if(i < numKernels - 1){
			retArr[i] = (lineLength / numKernels) * (i + 1);
		}
		else{
			retArr[i] = (lineLength / numKernels) * (i + 1) + (lineLength % numKernels);
		}
	}
}

void printGoL(int size, char **gol){
	int i = 0;
	int j = 0;

	for(i = 0; i < size; i++){
		for(j = 0; j < size; j++){
			printf(" %c",gol[i][j]);
		}
		printf("\n");
	}
}

char displayNum (int *coords, int yIndex, int numCoords){
	int i = 0;
	char out = ' ';
	for(i = 0; i < numCoords; i++){
		if(yIndex < coords[i]){
			if(numCoords > 10){
				out = 'X';
			}
			else{
				out = (char)(i + 48);
			}
			return out;
		}
	}
	return out;
}

void destroy_gol(char **gol, int size){
	for (int i = 0; i < size; i++)
		free(gol[i]);
	free(gol);
}
