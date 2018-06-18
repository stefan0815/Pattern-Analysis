/*
 *
 * Compilation line:
 g++ -o main main.cpp DensityTree.cpp -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml
 https://www.nowpublishers.com/article/Details/CGV-035 <-density forest paper
*
*/

#include "DensityTree.h"

Mat ComputeCovariance(Mat &matrix)
{
	Mat mean;
	reduce(matrix, mean, 0, CV_REDUCE_AVG);
	const int nElem = matrix.rows;
	const int dim = matrix.cols;
	Mat sum = Mat::zeros(dim, dim, CV_64F);
	for (int i = 0; i < nElem; i++){
		Mat x_i = Mat(matrix.row(i) - mean);
		Mat x_i_t;
		transpose(x_i, x_i_t);
		Mat cov_i = x_i_t * x_i;
		sum = sum + cov_i;
	}
	return sum / nElem;
}

Mat ComputeCovariance(Mat &matrix,Mat &mean)
{
	const int nElem = matrix.rows;
	const int dim = matrix.cols;
	Mat sum = Mat::zeros(dim, dim, CV_64F);
	for (int i = 0; i < nElem; i++){
		Mat x_i = Mat(matrix.row(i) - mean);
		Mat x_i_t;
		transpose(x_i, x_i_t);
		Mat cov_i = x_i_t * x_i;
		sum = sum + cov_i;
	}
	return sum / nElem;
}



Decision::Decision(double min_x, double max_x, double min_y, double max_y, double n_x, double n_y, Mat &inputSet, Mat &inputCov){
	p_x = min_x + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(max_x-min_x)));
	p_y = min_y + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(max_y-min_y)));
	//double length = 0;
	//do{
	//	n_x = min_x + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(max_x-min_x)));
	//	n_y = min_y + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(max_y-min_y)));
	//	n_x = n_x - p_x;
	//	n_y = n_y - p_y;
	//	length = sqrt(n_x*n_x + n_y*n_y);
	//}
	//while (length == 0);
	//n_x /= length;
	//n_y /= length;
	this->n_x = n_x;
	this->n_y = n_y;
	this->set = inputSet;
	this->cov = inputCov;
	CalculateSubsets();
	CalcInformation();
}
bool Decision::Decide(double x, double y){
	return (n_x * (x - p_x) + n_y * (y - p_y)) >= 0;
}

void Decision::CalculateSubsets(){
	leftSubset = Mat();
	rightSubset = Mat();
	for (int i = 0; i < set.rows; i++){
		double x = set.at<double>(i, 0);
		double y = set.at<double>(i, 1);
		if (Decide(x, y)){
			leftSubset.push_back(set.row(i));
		}
		else{
			rightSubset.push_back(set.row(i));
		}
	}
	//double leftProb = (double)rightSubset.rows / (double)set.rows;
	//double rightProb = (double)rightSubset.rows / (double)set.rows;
	//leftEntropy = -leftProb * log(leftProb);
	//rightEntropy = -rightProb * log(rightProb);
}

void Decision::CalcInformation(){
	double leftRatio = (double)leftSubset.rows / (double)set.rows;
	double rightRatio = (double)rightSubset.rows / (double)set.rows;
	leftCov = ComputeCovariance(leftSubset);
	rightCov = ComputeCovariance(rightSubset);
	if(leftSubset.rows == 0 || rightSubset.rows == 0){
		information = 0;
		return;
	}
	double lDet = determinant(leftCov);
	double rDet = determinant(rightCov);
	if (lDet > 0.00001f && rDet > 0.00001f){
		information = log(determinant(cov)) - leftRatio*log(lDet) - rightRatio*log(rDet);
	}
	else{
		information = 0;
	}
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
    this-> nodes[0].inputSet = X;
    this-> nodes[0].inputCov = ComputeCovariance(nodes[0].inputSet);
    this-> cov = this-> nodes[0].inputCov;
	reduce(X, mean, 0, CV_REDUCE_AVG);
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

		for (int dir = 0; dir < 2; dir++){
			for (int i = 0; i < n_thresholds; i++){
				double n_x = dir;
				double n_y = 1 - dir;
				Decision dec = Decision(min_x, max_x, min_y, max_y, n_x, n_y, nodes[k].inputSet, nodes[k].inputCov);
				if (nodes[k].decision.information < dec.information){
					nodes[k].decision = dec;
				}
			}
		}
		nodes[k].childrenL->inputSet = nodes[k].decision.leftSubset;
		nodes[k].childrenR->inputSet = nodes[k].decision.rightSubset;
		nodes[k].childrenL->inputCov = nodes[k].decision.leftCov;
		nodes[k].childrenR->inputCov = nodes[k].decision.rightCov;
	}
	cout << nodes[0].inputSet.rows <<","<< nodes[0].decision.set.rows << ":::" << nodes[0].decision.leftSubset.rows << "|||" << nodes[0].decision.rightSubset.rows << endl;
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
	Mat M = Mat::zeros(X.rows,X.cols,CV_64F);
	int n_cluster = N-iN;
	Mat weights = Mat::ones(1,n_cluster,CV_64F);
	Ptr<EM> em_model = EM::create();
	em_model->setClustersNumber(n_cluster);
	//Mat covs[n_cluster];
	Mat means = Mat::zeros(n_cluster,X.cols,CV_64F);
	for (int k = iN; k < N; k++){
		//covs[k-iN] = nodes[k].inputCov;
		Mat meank;
		reduce(nodes[k].inputCov, meank, 0, CV_REDUCE_AVG);
		cout << meank << endl;
		meank.copyTo(means.row(k-iN));
	}

	Mat loglikely;
	Mat labels;
	Mat probs;
	cout << means << endl;
	if(em_model->trainE(X,means,noArray(),weights,loglikely,labels,probs)){
		cout << "successfully trained"<<endl;
	}

	/* for each sample in X:
	 * create new sample with:
	 * x.at<double>(0,0) = subset.at<double>(i,0);
	 * x.at<double>(0,1) = subset.at<double>(j,1);
	 * get probability of that point to be in the Gauss distribution of the subset
	 * add these probabilities for all j in subset;
	 * -> Px1[i] = sum over j
	 * Px2: fix second component and iterate over first component
	 * x.at<double>(0,1) = subset.at<double>(i,1);
	 * x.at<double>(0,0) = subset.at<double>(j,0);
	 */

	for (int k = iN; k < N; k++){//iterate over each leaf;
		Mat subset = nodes[k].inputSet;
		for (int i = 0; i < subset.rows; i++){
			Mat x_1 = Mat::zeros(1,2,CV_64F);
			x_1.at<double>(0,0) = subset.at<double>(i,0);

			Mat x_2 = Mat::zeros(1,2,CV_64F);
			x_2.at<double>(0,1) = subset.at<double>(i,1);

			for(int j = 0; j < subset.rows; j++){
				x_1.at<double>(0,1) = subset.at<double>(j,1);
				x_2.at<double>(0,0) = subset.at<double>(j,0);
				Mat probs_1 = Mat::zeros(1,n_cluster,CV_64F);
				Mat probs_2 = Mat::zeros(1,n_cluster,CV_64F);
				em_model->predict(x_1,probs_1);
				em_model->predict(x_2,probs_2);
				M.at<double>(0,i) += probs_1.at<double>(0,N-iN);
				M.at<double>(1,i) += probs_2.at<double>(0,N-iN);
			}
		}
	}
    return M;//Temporal, only to not generate an error when compiling
}



