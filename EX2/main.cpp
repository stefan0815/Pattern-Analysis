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
Mat reduceLLE(Mat &dataMatrix, unsigned int dim);

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
	//Mat dataIsomap = reduceIsomap(dataMatrix,2);
	//Draw2DManifold(dataIsomap,"ISOMAP",nSamplesI,nSamplesJ);
	
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
        Mat reduced = U.rowRange(0,2);
        Mat reducedT;
        transpose(reduced, reducedT);
        return dataMatrix * reducedT;
}


Mat ComputeCovarianc(Mat &matrix, Mat mean)
{
	Mat Cov = Mat();
        const int nElem = matrix.rows;
        const int dim = matrix.cols;
	Mat sum = Mat::zeros(3, 3, CV_64F);
        for (int i = 0; i < nElem; i++)
	{
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

Mat reduceIsomap(Mat &dataMatrix, unsigned int dim)
{
	return dataMatrix;
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
