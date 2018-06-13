#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <map>

using namespace cv;
using namespace std;

// functions for drawing
void Draw3DManifold(Mat &dataMatrix, char const * name, int nSamplesI, int nSamplesJ);
void Draw2DManifold(Mat &dataMatrix, char const * name, int nSamplesI, int nSamplesJ);

//myFunctions
Mat ComputeCovarianc(Mat &matrix, Mat mean);
double findMaxL2(Mat &matrix);

// functions for dimensionality reduction
// first parameter: features saved in rows
// second parameter: desired # of dimensions ( here: 2)
Mat reducePCA(Mat &dataMatrix, unsigned int dim);
Mat reduceIsomap(Mat &dataMatrix, unsigned int dim);


int main(int argc, char** argv)
{
	// generate Data Matrix
	unsigned int nSamplesI = 10;
	unsigned int nSamplesJ = 10;
	Mat dataMatrix =  Mat(nSamplesI*nSamplesJ, 3, CV_64F);
	// noise in the data
	double noiseScaling = 1000.0;
	
	for (int i = 0; i < nSamplesI; i++)
	{
		for (int j = 0; j < nSamplesJ; j++)
		{
			dataMatrix.at<double>(i*nSamplesJ+j,0) = (i/(double)nSamplesI * 2.0 * 3.14 + 3.14) * cos(i/(double)nSamplesI * 2.0 * 3.14) + (rand() % 100)/noiseScaling;
			dataMatrix.at<double>(i*nSamplesJ+j,1) = (i/(double)nSamplesI * 2.0 * 3.14 + 3.14) * sin(i/(double)nSamplesI * 2.0 * 3.14) + (rand() % 100)/noiseScaling;
            dataMatrix.at<double>(i*nSamplesJ+j,2) = 10.0*j/(double)nSamplesJ + (rand() % 100)/noiseScaling;
		}
	}
	
	// Draw 3D Manifold
	Draw3DManifold(dataMatrix, "3D Points",nSamplesI,nSamplesJ);
	
	// PCA

	Mat dataPCA = reducePCA(dataMatrix,2);
	double scale = findMaxL2(dataPCA);
	//cout << dataPCA << endl;
	dataPCA = dataPCA / (2*scale);
	//cout << dataPCA << endl;
	Draw2DManifold(dataPCA,"PCA",nSamplesI,nSamplesJ);

	// Isomap
	Mat dataIsomap = reduceIsomap(dataMatrix,2);
	Draw2DManifold(dataIsomap, "ISOMAP",nSamplesI,nSamplesJ);
	
	waitKey(0);


	return 0;
}

Mat reducePCA(Mat &dataMatrix, unsigned int dim)
{

	Mat meanVec;
	reduce (dataMatrix,meanVec,0,CV_REDUCE_AVG);
	// 1. Compute Covariance Matrix of transformed mean vectors
	Mat Cov = ComputeCovarianc(dataMatrix, meanVec);
	// 2. Compute the 2 eigenvectors of the covariance matrix belong into the largest eigenvalues.
	Mat U, S, vT;
	SVD::compute(Cov, S, U, vT);
	return dataMatrix * U.colRange(0,dim);
}


Mat ComputeCovarianc(Mat &matrix, Mat mean)
{
	Mat Cov = Mat();
	const int nElem = matrix.rows;
	const int dim = matrix.cols;
	Mat sum = Mat::zeros(3, 3, CV_64F);
	for (int i = 0; i < nElem; i++){
		Mat x_i = matrix.row(i)-mean;
		Mat x_i_t;
		transpose (x_i, x_i_t);
		Mat cov_i = x_i_t * x_i;
		sum += cov_i;
	}
	return sum/nElem;
}


double findMaxL2(Mat &matrix){
    const int nElem = matrix.rows;
    double max = 0.0;
    for(int i = 0; i < nElem; i++){
        Mat x_i = matrix.row(i);
        Mat x_i_t;
        transpose (x_i, x_i_t);
        Mat scalar = (x_i * x_i_t);
        double value = sqrt(scalar.at<double>(0,0));
        if(value > max){
            max = value;
        }
    }
    cout <<"Maximum L2 Norm: "<< max << endl;
    return max;
}


map<double, int> KNN(Mat &dataMatrix, int index, int K){
	K = K + 1; //Add itself to KNN list
	int nElem = dataMatrix.rows;
	map<double, int> distanceMap;
	Mat x = dataMatrix.row(index);
	for(int i = 0; i < nElem; i++)
	{
		//get iths datapoint
		Mat x_i 	= dataMatrix.row(i);
		//compute difference to given index
		Mat diff 	= x - x_i;
		//compute the L2 norm result in distance
		Mat diff_t;
		transpose (diff, diff_t);
		Mat scalar = diff * diff_t;
		double distance = sqrt(scalar.at<double>(0,0));
		//add index and distance to the map
		distanceMap.insert(pair<double, int>(distance,i));
	}
	//take the K nearest distances
	map<double, int> ret;
	int k = 0;
	for (auto iter = distanceMap.begin(); k < K; k++, iter++){
	    ret.insert(pair<double, int>(iter->first,iter->second));
	}
	//cout << ret.size() << endl;
	return ret;
}

Mat floydWarshall (Mat &distanceMap){
	int nElem = distanceMap.rows;
<<<<<<< HEAD
=======

>>>>>>> 88e409eabab03ff12b722a022c73dc96c9109402
	for (int k = 0; k < nElem; k++){
		for (int i = 0; i < nElem; i++){
			for (int j = 0; j < nElem; j++){
				double shortCut = distanceMap.at<double>(i,k) + distanceMap.at<double>(k,j);
				if (shortCut < distanceMap.at<double>(i,j)){
					distanceMap.at<double>(i,j) = shortCut;
					distanceMap.at<double>(j,i) = shortCut;
				}
			}
		}
	}
    return distanceMap;
}

Mat reduceIsomap(Mat &dataMatrix, unsigned int dim)
{
	int K = 10;
	int nElem = dataMatrix.rows;

	Mat meanVec;
	reduce(dataMatrix,meanVec,0,CV_REDUCE_AVG);
	for(int i = 0 ; i < nElem ; i++){
		dataMatrix.row(i) -= meanVec;
	}


	Mat distanceMap = Mat::ones(nElem, nElem, CV_64F) * 10000;
	for(int i = 0; i < nElem; i++){
		map<double, int> kNNResult = KNN(dataMatrix,i,K);
		int k = 0;
		for (auto iter = kNNResult.begin(); k < K + 1; k++, iter++){ //K+1 because kNN contains node itself
			distanceMap.at<double>(i, iter->second) = iter->first;
		}
	}

	distanceMap = floydWarshall(distanceMap);
	Mat D = distanceMap.mul(distanceMap);
	Mat C = Mat::eye(nElem,nElem,CV_64F);
	C -= (Mat::ones(nElem,nElem,CV_64F) * 1.0/(double)nElem);
	cout << C << endl;
	Mat SVDInput= -0.5 * C * D * C; //X^T * X
    Mat U, S, vT;
    SVD::compute(SVDInput, S, U, vT);
	return U.colRange(0,dim);
}

void Draw3DManifold(Mat &dataMatrix, char const * name, int nSamplesI, int nSamplesJ)
{
	Mat origImage = Mat(1000,1000,CV_8UC3);
	origImage.setTo(0.0);
	for (int i = 0; i < nSamplesI; i++)
	{
		for (int j = 0; j < nSamplesJ; j++)
		{
			Point p1;
			p1.x = dataMatrix.at<double>(i*nSamplesJ+j,0)*50.0 +500.0 - dataMatrix.at<double>(i*nSamplesJ+j,2)*10;
			p1.y = dataMatrix.at<double>(i*nSamplesJ+j,1) *50.0 + 500.0 - dataMatrix.at<double>(i*nSamplesJ+j,2)*10;
			circle(origImage,p1,3,Scalar( 255, 255, 255 ));
			
			Point p2;
			if(i < nSamplesI-1)
			{
				p2.x = dataMatrix.at<double>((i+1)*nSamplesJ+j,0)*50.0 +500.0 - dataMatrix.at<double>((i+1)*nSamplesJ+(j),2)*10;
				p2.y = dataMatrix.at<double>((i+1)*nSamplesJ+j,1) *50.0 + 500.0 - dataMatrix.at<double>((i+1)*nSamplesJ+(j),2)*10;
			
				line( origImage, p1, p2, Scalar( 255, 255, 255 ), 1, 8 );
			}
			if(j < nSamplesJ-1)
			{
				p2.x = dataMatrix.at<double>((i)*nSamplesJ+j+1,0)*50.0 +500.0 - dataMatrix.at<double>((i)*nSamplesJ+(j+1),2)*10;
				p2.y = dataMatrix.at<double>((i)*nSamplesJ+j+1,1) *50.0 + 500.0 - dataMatrix.at<double>((i)*nSamplesJ+(j+1),2)*10;
			
				line( origImage, p1, p2, Scalar( 255, 255, 255 ), 1, 8 );
			}
		}
	}
	

	namedWindow( name, WINDOW_AUTOSIZE );
	imshow( name, origImage ); 
}

void Draw2DManifold(Mat &dataMatrix, char const * name, int nSamplesI, int nSamplesJ)
{
	Mat origImage = Mat(1000,1000,CV_8UC3);
	origImage.setTo(0.0);
	for (int i = 0; i < nSamplesI; i++)
	{
		for (int j = 0; j < nSamplesJ; j++)
		{
			Point p1;
			p1.x = dataMatrix.at<double>(i*nSamplesJ+j,0)*1000.0 +500.0;
			p1.y = dataMatrix.at<double>(i*nSamplesJ+j,1) *1000.0 + 500.0;
			//circle(origImage,p1,3,Scalar( 255, 255, 255 ));
			
			Point p2;
			if(i < nSamplesI-1)
			{
				p2.x = dataMatrix.at<double>((i+1)*nSamplesJ+j,0)*1000.0 +500.0;
				p2.y = dataMatrix.at<double>((i+1)*nSamplesJ+j,1) *1000.0 + 500.0;
				line( origImage, p1, p2, Scalar( 255, 255, 255 ), 1, 8 );
			}
			if(j < nSamplesJ-1)
			{
				p2.x = dataMatrix.at<double>((i)*nSamplesJ+j+1,0)*1000.0 +500.0;
				p2.y = dataMatrix.at<double>((i)*nSamplesJ+j+1,1) *1000.0 + 500.0;
				line( origImage, p1, p2, Scalar( 255, 255, 255 ), 1, 8 );
			}
			
		}
	}
	

	namedWindow( name, WINDOW_AUTOSIZE );
	imshow( name, origImage ); 
	imwrite( (String(name) + ".png").c_str(),origImage);
}
