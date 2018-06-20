/*
 *
 * Compilation line:
 g++ -o main main.cpp DensityTree.cpp -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml
 https://www.nowpublishers.com/article/Details/CGV-035 <-density forest paper
*
*/

#include "DensityTree.h"

void plotData2(Mat dataMatrix, char const * name)
{
    Mat origImage = Mat(1000,1000,CV_8UC3);
    origImage.setTo(0.0);
    double minValX,maxValX;
    minMaxIdx( dataMatrix.col(0), &minValX, &maxValX,NULL,NULL);
    double minValY,maxValY;
    minMaxIdx( dataMatrix.col(1), &minValY, &maxValY,NULL,NULL);
    double v;
    double nmin=100,nmax=900;
    for (int i = 0; i < 1000; i++)
    {
        Point p1;
        v= dataMatrix.at<double>(i,0);
        p1.x = ((v-minValX)/(maxValX-minValX))*(nmax-nmin)+nmin;
        v= dataMatrix.at<double>(i,1);
        p1.y = ((v-minValY)/(maxValY-minValY))*(nmax-nmin)+nmin;
        circle(origImage,p1,3,Scalar( 255, 255, 255 ));
    }

    namedWindow( name, WINDOW_AUTOSIZE );
    imshow( name, origImage );
}

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



Decision::Decision(double min_x, double max_x, double min_y, double max_y, double n_x, double n_y, Mat &inputSet, Mat &inputCov, std::vector<int>inputIdx){
	p_x = min_x + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(max_x-min_x)));
	p_y = min_y + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(max_y-min_y)));
	//double length = 0;
	//do{
	//	this->n_x = min_x + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(max_x-min_x)));
	//	this->n_y = min_y + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(max_y-min_y)));
	//	this->n_x = this->n_x - p_x;
	//	this->n_y = this->n_y - p_y;
	//	length = sqrt(this->n_x*this->n_x + this->n_y*this->n_y);
	//}
	//while (length == 0);
	//this->n_x /= length;
	//this->n_y /= length;
	this->n_x = n_x;
	this->n_y = n_y;
	this->set = inputSet;
	this->cov = inputCov;
	this->idx = inputIdx;
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
			leftIdx.push_back(idx[i]);
		}
		else{
			rightSubset.push_back(set.row(i));
			rightIdx.push_back(idx[i]);
		}
	}
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
    this-> nodes[0].inputIdx.resize(X.rows);
    std::iota (nodes[0].inputIdx.begin(), nodes[0].inputIdx.end(), 0);
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
	cout << "start training of tree" << endl;
	for(int k = 0; k < iN;k++){
		nodes[k].childrenL = &nodes[(k*2+1)];
		nodes[(k*2+1)].parent = &nodes[k];

		nodes[k].childrenR = &nodes[(k*2+2)];
		nodes[(k*2+2)].parent = &nodes[k];

		for (int dir = 0; dir < 2; dir++){
			for (int i = 0; i < n_thresholds; i++){
				double n_x = dir;
				double n_y = 1 - dir;
				Decision dec = Decision(min_x, max_x, min_y, max_y, n_x, n_y, nodes[k].inputSet, nodes[k].inputCov, nodes[k].inputIdx);
				if (nodes[k].decision.information < dec.information){
					nodes[k].decision = dec;
				}
			}
		}
		nodes[k].childrenL->inputSet = nodes[k].decision.leftSubset;
		nodes[k].childrenR->inputSet = nodes[k].decision.rightSubset;
		nodes[k].childrenL->inputCov = nodes[k].decision.leftCov;
		nodes[k].childrenR->inputCov = nodes[k].decision.rightCov;
		nodes[k].childrenL->inputIdx = nodes[k].decision.leftIdx;
		nodes[k].childrenR->inputIdx = nodes[k].decision.rightIdx;
	}
	cout << "end training of tree" << endl;
}
Mat DensityTree::densityXY()
{

	cout << "Start densityXY" << endl;
	Mat M = Mat::zeros(X.rows,X.cols,CV_64F);
	int n_cluster = 1;
	Mat means = Mat::zeros(n_cluster,X.cols,CV_64F);
	for (int k = iN; k < N; k++){
		for(int i = 0; i < nodes[k].inputIdx.size();i++){
			int idx = nodes[k].inputIdx[i];
			Mat point = nodes[k].inputSet.row(i);

			if(!nodes[k].hasPDF){

				Mat meank;
				reduce(nodes[k].inputSet, meank, 0, CV_REDUCE_AVG);
				std::vector<Mat> covArray;
				covArray.push_back(nodes[k].inputCov);
				nodes[k].pdf = EM::create();
				nodes[k].pdf->setClustersNumber(n_cluster);
				if(nodes[k].pdf->trainE(nodes[k].inputSet,meank,covArray, noArray(),noArray(),noArray(),noArray())){
					cout << "Em successfully trained for leaf #" << k - iN + 1<<endl;
					nodes[k].hasPDF = true;
				}else
					cout << "training failed" << endl;
			}

			for(int j = 0; j < nodes[k].inputIdx.size(); j++)
			{
				// X
				Mat maginalHelperX 			= Mat::zeros( 1, 2, CV_64F );
				maginalHelperX.at<double>(0) = point.at<double>(0);
				maginalHelperX.at<double>(1) = nodes[k].inputSet.row(j).at<double>(1);

				// Y
				Mat maginalHelperY 			= Mat::zeros( 1, 2, CV_64F );
				maginalHelperY.at<double>(0) = nodes[k].inputSet.row(j).at<double>(0);
				maginalHelperY.at<double>(1) = point.at<double>(1);

				Mat a;
				M.row(idx).at<double>(0) += (float) -exp(nodes[k].pdf->predict2(maginalHelperX, a).val[0]);
				M.row(idx).at<double>(1) += (float) -exp(nodes[k].pdf->predict2(maginalHelperY, a).val[0]);
			}
		}
	}
	cout << "End densityXY" <<endl;
	cout << endl;
	return M;
}



