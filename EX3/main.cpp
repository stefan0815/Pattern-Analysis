/*
 *
 * Compilation line:
 g++ -o main main.cpp DensityTree.cpp -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml
*
*/

#include "DensityTree.h"

using namespace cv;
using namespace std;

Mat densityForest(unsigned int D, unsigned int numberOfThresholds,unsigned int numberOfTrees, Mat X);
Mat generateData();
void plotData(Mat dataMatrix, char const * name);
void plotDensities(Mat data, vector<Mat> densityXY,int dimension,char const * name);
int main(int argc, char** argv)
{
    int n_levels=2; //This is a parameter, it can change
    int n_thresholds=50;
    
    Mat dataMatrix = generateData();
    plotData(dataMatrix,"Data Matrix Plot");

    vector<Mat> densityXY;
    
    densityXY.push_back(densityForest(n_levels,n_thresholds,1,dataMatrix));
    densityXY.push_back(densityForest(n_levels,n_thresholds,10,dataMatrix));
    densityXY.push_back(densityForest(n_levels,n_thresholds,20,dataMatrix));
    
    plotDensities(dataMatrix,densityXY,0,"Densities X");
    plotDensities(dataMatrix,densityXY,1,"Densities Y");

    waitKey(0);
    return 0;
}

Mat generateData()
{

    unsigned int dim = 2;
    Mat A =  Mat(500, dim, CV_64F);;
    cv::theRNG().fill(A, cv::RNG::NORMAL, 5, 3);	
    Mat B =  Mat(500, dim, CV_64F);
    cv::theRNG().fill(B, cv::RNG::NORMAL, -9, 3);
    Mat dataMatrix;
    vconcat(A, B, dataMatrix);

    return dataMatrix;
}


Mat densityForest(unsigned int D, unsigned int numberOfThresholds,unsigned int numberOfTrees, Mat X)
{

    Mat density=Mat::zeros(1000,2,CV_64F);
    for (int i=0;i<numberOfTrees;i++)
    {
        DensityTree T(D,numberOfThresholds,X);
        T.train();
        density+=T.densityXY();
        cout << i + 1 << "/" << numberOfTrees << " trees processed" << endl;
    }
    return density/(double)numberOfTrees;
}

void plotData(Mat dataMatrix, char const * name)
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

void plotDensities(Mat data, vector<Mat> density, int dim, char const * name)
{

    int n_densities=density.size();
    Mat origImage = Mat(1000,1000,CV_8UC3);
    origImage.setTo(0.0);
    double minValX,maxValX;
    double minValY,maxValY;
    double tempMinY,tempMaxY;
    minMaxIdx( data.col(dim), &minValX, &maxValX,NULL,NULL);
    minMaxIdx( density[0].col(dim), &minValY, &maxValY,NULL,NULL);
    for(int i=1;i<n_densities;i++)
    {
        minMaxIdx( density[i].col(dim), &tempMinY, &tempMaxY,NULL,NULL);
        if(tempMaxY>maxValY) maxValY=tempMaxY;
        if(tempMinY<minValY) minValY=tempMinY;
    }
    double v;
    int nmin=100;
    int nmax=900;
    int r=0,g=0,b=0;
    for(int j=0; j< n_densities;j++)
    {
        if(j==0)r=250;
        if(j==1)g=250;
        if(j==2)b=250;
        for (int i = 0; i < 1000; i++)
        {
            Point p1;
            v= data.at<double>(i,dim);
            p1.x = ((v-minValX)/(maxValX-minValX))*(nmax-nmin)+nmin;
            v= density[j].at<double>(i,dim);
            p1.y = ((v-minValY)/(maxValY-minValY))*(nmax-nmin)+nmin;
            circle(origImage,p1,1,Scalar(b,g,r));
        }
        if(j==0)r=0;
        if(j==1)g=0;
        if(j==2)b=0;
    }
    namedWindow( name, WINDOW_AUTOSIZE );
    imshow( name, origImage );
    imwrite( (String(name) + ".png").c_str(),origImage);
}
