#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
# define PI 3.14159265358979323846

using namespace cv;
using namespace std;
using namespace ml;

#ifndef DENSITYTREE_H
#define DENSITYTREE_H

class Decision{
public:
	double p_x,p_y;
	double n_x,n_y;
	double information;
	Mat set;
	Mat cov;
	Mat leftSubset;
	Mat leftCov;
	Mat rightSubset;
	Mat rightCov;

	Decision(){information = -1;};
	Decision(double min_x, double max_x, double min_y, double max_y,double n_x, double n_y, Mat &inputSet, Mat &inputCov);
	bool Decide(double x, double y);
	void CalculateSubsets();
	void CalcInformation();


};

class MyNode{
public:
	bool isRoot;
	MyNode* parent;
	MyNode* childrenL;
	MyNode* childrenR;
	bool isLeaf;
	//bool isLeftChild;
	Decision decision;
	Mat inputSet;
	Mat inputCov;
	MyNode(){
		isRoot = false;
		isLeaf = false;
		//isLeftChild = false;
	}
};

class DensityTree 
{
public:
    DensityTree();
    DensityTree(unsigned int D, unsigned int R, Mat X);
    void train();
    Mat densityXY();

private:
    unsigned int D;
    unsigned int n_thresholds;
    unsigned int N;
    unsigned int iN;
    //vector<int >asd ;
    std::vector<MyNode> nodes;
    Mat X;
    Mat cov;
    Mat mean;
    double min_x,max_x,min_y,max_y;
};

#endif /* DENSITYTREE_H */

