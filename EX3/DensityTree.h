#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
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

#ifndef DENSITYTREE_H
#define DENSITYTREE_H

class Decision{
public:
	double p_x,p_y;
	double n_x,n_y;
	double information;
	Decision(){};
	Decision(double min_x, double max_x, double min_y, double max_y, Mat X);
	bool decide(double x, double y);
	void calcInformation(Mat X);


};

class MyNode{
public:
	bool isRoot;
	MyNode* parent;
	MyNode* childrenL;
	MyNode* childrenR;
	bool isLeaf;
	Decision decision;
	MyNode(){
		isRoot = false;
		isLeaf = false;
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
    double min_x,max_x,min_y,max_y;
};

#endif /* DENSITYTREE_H */

