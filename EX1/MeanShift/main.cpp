#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <iostream>

using namespace cv;
using namespace std;

//g++ -o main main.cpp -lopencv_core -lopencv_highgui -lopencv_imgcodecs

double lambda=0.03;
double h = 1.5;
double tolX=0.001;
int maxIter=1;



string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}


float Kernel(float x){
    return exp(-0.5*x);
}

Vec3f KernelFunc(vector<Vec3f> x_vec, int k){
    Vec3f m_x = Vec3f(0.0,0.0,0.0);
    float scale = 0.0;
    int count = 0;
    for(int i = 0; i < x_vec.size();i++){
        Vec3f diff = x_vec[i] - x_vec[k];
        float length = norm(diff);//sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]);
        float kRes = Kernel(length/(h*h));
        if(kRes < lambda){
            m_x += (kRes * x_vec[i]);
            scale += (kRes);
            count++;
        }
        //cout << y[k] << endl;
    }
    if(count == 0)
        return x_vec[k];
    return m_x/scale;
}


int countInliers(vector<Vec3f> y){
    int count = 0;
    for(int k = 0; k < y.size();k++){
          if(norm(y[k]) < lambda){
              count++;
          }
    }
    return count;
}

//Mat meanshiftFunc2(Mat img){

//    img.convertTo(img,CV_32F);
//    cout << type2str(img.type())<<endl;
//    vector<Vec3f> x;
//    for(int y_Coord = 0; y_Coord < img.cols;y_Coord++){
//        for(int x_Coord = 0; x_Coord < img.rows;x_Coord++){
//            Point3f feature = Vec3f(x_Coord,y_Coord,img.at<float>(x_Coord,y_Coord));
//            x.push_back(feature);
//        }
//    }


//    Vec3f meanShift = Vec3f(0.0);
//    for(int iter = 0; iter < maxIter; iter++){
//        vector<Vec3f> y = vector<Vec3f>(x.size());
//        for(int i = 0; i < x.size();i++){

//            vector<float> y_tmp = KernelFunc(x,i);
//            Vec3f y_point = Vec3f(0.0,0.0,0.0);
//            float sum = 0;
//            for(int k = 0; k < y_tmp.size();k++){
//                y_point += (x[i]*y_tmp[k]);
//                sum += y_tmp[k];
//                cout << sum << endl;
//            }
//            y_point /= sum;
//            y[i] = y_point;
//        }
//        int count = countInliers(y);
//        cout << count << endl;
//        if(count == 0){
//            cout << "converged" <<endl;
//        }
//        else{
//            for(int k = 0; k < y.size();k++){
//                if(norm(y[k]) < lambda){
//                    //cout << (1.0/(float)count) << endl;
//                    meanShift += (1.0/(float)count)*x[k];
//                    meanShift -= x[k];
//                    x[k] += meanShift;
//                }
//            }
//            cout << meanShift<< endl;
//        }
//        cout << meanShift<< endl;
//    }

//    Mat imgOut = Mat(img.rows, img.cols, CV_32F);

//    for(int i = 0; i < x.size();i++){
//        imgOut.at<float>(x[i][0],x[i][1]) = x[i][2];
//    }
//    imgOut.convertTo(imgOut,CV_8U);
//    return imgOut;
//}


Mat meanshiftFunc(Mat img){

    img.convertTo(img,CV_32F);
    cout << type2str(img.type())<<endl;
    vector<Vec3f> x_vec;
    for(int y_Coord = 0; y_Coord < img.cols;y_Coord++){
        for(int x_Coord = 0; x_Coord < img.rows;x_Coord++){
            Point3f feature = Vec3f(x_Coord,y_Coord,img.at<float>(x_Coord,y_Coord));
            x_vec.push_back(feature);
        }
    }


    for(int iter = 0; iter < maxIter; iter++){
        for(int k = 0; k < x_vec.size();k++){
            Vec3f m_x = KernelFunc(x_vec,k);
            Vec3f meanShift = m_x - x_vec[k];
            x_vec[k][2] += meanShift[2];
            if(0 == k % (x_vec.size() / 10)){
                cout << endl;
                cout << (int)((10*k) / (x_vec.size() / 10) ) <<"%";
                cout << " -  Norm meanShift:"<<norm(meanShift) << " <- should be smaller than:"<< tolX << endl;
            }else if (0 == k % (x_vec.size() / 100)){
                cout << ".";
                cout << flush;
            }
        }
    }
    //cout << endl;
    Mat imgOut = Mat(img.rows, img.cols, CV_32F);

    for(int i = 0; i < x_vec.size();i++){
        imgOut.at<float>(x_vec[i][0],x_vec[i][1]) = x_vec[i][2];
    }
    imgOut.convertTo(imgOut,CV_8U);
    return imgOut;
}


int main(int argc, char *argv[])
{

    string fileName = "../cameraman_noisy_original.png";
    Mat image = imread(fileName, IMREAD_GRAYSCALE);

    if(!image.data)                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
    imshow("Display window", image);                   // Show our image inside it.

    Mat meanShiftImage = meanshiftFunc(image.clone());

    namedWindow("Display window1", WINDOW_AUTOSIZE);
    imshow("Display window1", meanShiftImage );
    imwrite( "../cameraman_noisy_denoised.png", meanShiftImage);

    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}
