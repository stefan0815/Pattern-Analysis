/*
 *
 * Compilation line:
 g++ -o main main.cpp DensityTree.cpp -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml
 https://www.nowpublishers.com/article/Details/CGV-035 <-density forest paper
*
*/

#include "DensityTree.h"


Decision::Decision(double min_x, double max_x, double min_y, double max_y, Mat X){
	p_x = min_x + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(max_x-min_x)));
	p_y = min_y + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(max_y-min_y)));
	double length = 0;
	do{
		n_x = min_x + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(max_x-min_x)));
		n_y = min_y + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(max_y-min_y)));
		n_x = n_x - p_x;
		n_y = n_y - p_y;
		length = sqrt(n_x*n_x + n_y*n_y);
	}
	while (length == 0);
	n_x /= length;
	n_y /= length;
	calcInformation(X);
}
bool Decision::decide(double x, double y){
	return (n_x * (x - p_x) + n_y * (y - p_y)) >= 0;
}
void Decision::calcInformation(Mat X){
	information = 0;//finish me
}

DensityTree::DensityTree(unsigned int D, unsigned int n_thresholds, Mat X) 
{
    /*
     * D is the depht of the tree. If D=2, the tree will have 3 nodes, the root and its 2 children.
     * This is a binari tree. Once you know D, you know the number of total nodes is pow(2,D)-1, the number of leaves or terminal nodes are pow(2,(D-1)).
     * The left child of the i-th node is the (i*2+1)-th node and the right one is the (i*2+2).
     * Having this information, you can use simple arrays as a structure to save information of each node. For example, you can save in a boolean vector wheather the
     * node is a leave or not like this:
     *
     * bool *isLeaf=new bool[numOfNodes];
     * for(int i=0;i<numOfInternal;i++)
     *     isLeaf[i]=false;
     * for(int i=numOfInternal;i<numOfNodes;i++)
     *  isLeaf[i]=true;
     */
    this-> D=D;
    this-> X=X;
    this-> n_thresholds=n_thresholds;
    this-> N = pow(2,D)-1;
    this-> iN = N - pow(2,(D-1));
    this-> nodes = vector<MyNode>(N);
    nodes[0].isRoot = true;
    for(int i =iN; i < N;i++){
    	nodes[i].isLeaf = true;
    }
    srand (time(NULL));
    Mat col1 = X.col(0);
    Mat col2 = X.col(0);
    minMaxLoc(col1, &min_x, &max_x);
    minMaxLoc(col2, &min_y, &max_y);
}




void DensityTree::train()
{    

	for(int k = 0; k < iN;k++){
		nodes[k].childrenL = &nodes[(k*2+1)];
		nodes[(k*2+1)].parent = &nodes[k];
		nodes[k].childrenR = &nodes[(k*2+2)];
		nodes[(k*2+2)].parent = &nodes[k];
		for(int i = 0; i <n_thresholds;i++){

			Decision dec = Decision(min_x,max_x,min_y,max_y,X);
			if(nodes[k].decision.information < dec.information){
				nodes[k].decision = dec;
			}
		}
	}
	cout << D << endl;
	cout << iN << endl;
	cout << N << endl;
	cout << X.rows << endl;
    cout << "it is not implemented yet" << endl;//Temporal
}
Mat DensityTree::densityXY()
{

    /*
    *
    if X=
    [x1_1,x2_1;
     x1_2,x2_2;
     ....
     x1_N,x2_N]

    then you return
    M=
    [Px1,Px2]

    Px1 and Px2 are column vectors of size N (X and M have the same size)
    They are the marginals distributions.
    Check https://en.wikipedia.org/wiki/Marginal_distribution
    Feel free to delete this comments
    Tip: you can use cv::ml::EM::predict2 to estimate the probs of a sample.

    *
    */
    return X;//Temporal, only to not generate an error when compiling
}



