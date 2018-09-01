// YOUR NAME
// IF NECESSARY: YOUR COMPILATION COMMAND

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <map>
#include <math.h>

#define _USE_MATH_DEFINES

using namespace cv;
using namespace std;

void generateRandomObservations(Mat A, Mat B, Mat P, unsigned int* observations, unsigned int observationCount);
double observationProbabilityForward(Mat A, Mat B, Mat P, unsigned int* observations, unsigned int observationCount);
double bestStateSequence(Mat A, Mat B, Mat P, unsigned int* observations, unsigned int observationCount, unsigned int* bestStates);
double rd() { return (double)rand() / (double)RAND_MAX; } // we suggest to use this to create uniform random values between 0 and 1

int main(int argc, char** argv)
{
	// keep this to produce our results
	srand(42);
	srand (time(NULL));
	// Four states, 3 symbols
	Mat A =  Mat(4, 4, CV_64F);
	Mat B =  Mat(4, 3, CV_64F);
	Mat P = Mat(4,1, CV_64F);
	A.at<double>(0,0) = 0.5;
	A.at<double>(0,1) = 0.2;
	A.at<double>(0,2) = 0.3;
	A.at<double>(0,3) = 0.0;
	
	A.at<double>(1,0) = 0.2;
	A.at<double>(1,1) = 0.4;
	A.at<double>(1,2) = 0.1;
	A.at<double>(1,3) = 0.3;
	
	A.at<double>(2,0) = 0.7;
	A.at<double>(2,1) = 0.1;
	A.at<double>(2,2) = 0.1;
	A.at<double>(2,3) = 0.1;
	
	A.at<double>(3,0) = 0.0;
	A.at<double>(3,1) = 0.1;
	A.at<double>(3,2) = 0.8;
	A.at<double>(3,3) = 0.1;
	
	P.at<double>(0,0) = 0.7;
	P.at<double>(0,1) = 0.2;
	P.at<double>(0,2) = 0.1;
	P.at<double>(0,3) = 0.0;
	
	B.at<double>(0,0) = 0.6;
	B.at<double>(0,1) = 0.2;
	B.at<double>(0,2) = 0.2;
	
	B.at<double>(1,0) = 0.4;
	B.at<double>(1,1) = 0.4;
	B.at<double>(1,2) = 0.2;
	
	B.at<double>(2,0) = 0.3;
	B.at<double>(2,1) = 0.3;
	B.at<double>(2,2) = 0.4;
	
	B.at<double>(3,0) = 0.1;
	B.at<double>(3,1) = 0.2;
	B.at<double>(3,2) = 0.7;

	// Length = 2;
	unsigned int cnt = 2;
	unsigned int obs1[cnt];
	unsigned int bestStates1[cnt];
	generateRandomObservations(A,B,P,obs1,cnt);
	cout << "Observation Sequence: "; 
	for( int i = 0; i < cnt; i++)
	{
		cout << obs1[i] << " ";
	}
	cout << endl;
	double prob_all = observationProbabilityForward(A,B,P,obs1,cnt);
	double prob_best = bestStateSequence(A, B, P, obs1, cnt, bestStates1);
	cout << "Probability: " << prob_all << endl;
	cout << "Best Sequence: ";
	for( int i = 0; i < cnt; i++)
	{
		cout << bestStates1[i] << " ";
	}
	cout << "Probability: " << prob_best << endl;
	cout << "Best Prob. / Total Prob: " << prob_best / prob_all << endl << endl;
	
	// Length = 10
	cnt = 10;
	unsigned int obs2[]= {0,0,0,2,0,1,1,1,2,2};
	unsigned int bestStates2[cnt];
	//generateRandomObservations(A,B,P,obs2,cnt);
	cout << "Observation Sequence: "; 
	for( int i = 0; i < cnt; i++)
	{
		cout << obs2[i] << " ";
	}
	cout << endl;
	prob_all = observationProbabilityForward(A,B,P,obs2,cnt);
	prob_best = bestStateSequence(A, B, P, obs2, cnt, bestStates2);
	cout << "Probability: " << prob_all << endl;
	cout << "Best Sequence: ";
	for( int i = 0; i < cnt; i++)
	{
		cout << bestStates2[i] << " ";
	}
	cout << "Probability: " << prob_best << endl;
	cout << "Best Prob. / Total Prob: " << prob_best / prob_all << endl << endl;
	
	return 0;
}

//Generating random observations by random sampling as learned in the lecture
void generateRandomObservations(Mat A, Mat B, Mat P, unsigned int* observations, unsigned int observationCount)
{
	int numStates = A.rows;
	int numSymbols = B.cols;
        
	//cout << A.rows << " -- " << A.cols << endl;
	//cout << B.rows << " -- " << B.cols << endl;
	//cout << A << endl;
	//cout << B << endl;
	//cout << P << endl;
	int state;
        
	double r = rd(); //random number
	double s = 0.0; // probability counter
	// Starting state
        // iterate over all states and set the starting state
	for (int i = 0; i < numStates; i++)
	{
		if(r >= s && r < s + P.at<double>(i,0))
		{
			state = i;
			break;
		}
		s = s + P.at<double>(i,0);
	}

	for(int t = 0; t < observationCount; t++){
		// emit symbol
		r = rd();
		s = 0.0;
		for(int i = 0; i < numSymbols; i++)
		{
			if(r >= s && r < s + B.at<double>(state,i))
			{
				observations[t] = i;
				break;
			}
			s = s + B.at<double>(state,i);
		}
		// switch to next state
		if(t < observationCount-1)
		{
			r = rd();
			s = 0.0;
			for (int i = 0; i < numStates; i++)
			{
				if(r >= s && r < s + A.at<double>(state,i))
				{
					state = i;
					break;
				}
				s = s + A.at<double>(state,i);
			}
		}
	}
}
// Observation probability -> Forward algorithm
double observationProbabilityForward(Mat A, Mat B, Mat P, unsigned int* observations, unsigned int observationCount)
{
	//cout << A << endl;
	//cout << B << endl;
	//cout << P << endl;
	//cout << observationCount << endl;
	int numStates = A.rows;
	int numSymbols = B.cols;
	//Mat alpha = Mat(4, observationCount, CV_64F);;
	double alpha_t[numStates];
	double alpha_t1[numStates];
	for(int i = 0; i < numStates;i++){
		alpha_t[i] = P.at<double>(i,0) * B.at<double>(i,observations[0]);
	}
	for(int t = 1; t < observationCount; t++){
		for(int j = 0; j < numStates;j++){
			double sum = 0;
			for(int i = 0; i < numStates; i++){
				 sum += alpha_t[i] * A.at<double>(i,j);
			}
			alpha_t1[j] = sum * B.at<double>(j,observations[t]);
		}
		for(int i = 0; i < numStates; i++){
			alpha_t[i] = alpha_t1[i];
		}
	}
	double ret = 0;
	for(int i = 0; i < numStates; i++){
		ret += alpha_t[i];
	}
	return ret;
}
// best state sequence and observation probability using this state sequence -> Viterbi algorithm
// check https://en.wikipedia.org/wiki/Viterbi_algorithm for a pseudocode example
double bestStateSequence(Mat A, Mat B, Mat P, unsigned int* observations, unsigned int observationCount, unsigned int* bestStates)
{
	int numStates = A.rows;
	int numSymbols = B.cols;

	vector<vector<double>> delta;
	vector<vector<int>> psi;


	//initialize
	vector<double> delta_0(numStates);
	vector<int> psi_0(numStates);
	for(int i = 0; i < numStates;i++){
		delta_0[i] = P.at<double>(i,0) * B.at<double>(i,observations[0]);
		psi_0[i] = 0;
	}
	delta.push_back(delta_0);
	psi.push_back(psi_0);


	//Recursion
	for(int t = 1; t < observationCount; t++){
		vector<double> delta_t(numStates);
		vector<int> psi_t(numStates);
		for(int j = 0; j < numStates;j++){
			double max_delta = -1.0;
			double max_psi = -1.0;
			int idx_psi = -1;
			for(int i = 0; i < numStates; i++){
				double cur_psi = delta[t-1][i] * A.at<double>(i,j);
				double cur_delta = cur_psi * B.at<double>(j,observations[t]);
				if(cur_delta > max_delta){
					max_delta = cur_delta;
				}
				if(cur_psi > max_psi){
					max_psi = cur_psi;
					idx_psi = i;
				}
			}
			delta_t[j] = max_delta;
			psi_t[j] = idx_psi;
		}
		delta.push_back(delta_t);
		psi.push_back(psi_t);
	}

	double p_star = -1.0;
	for(int i = 0; i < numStates; i++){
		if(delta[delta.size() - 1][i] > p_star){
			p_star = delta[delta.size() - 1][i];
			bestStates[observationCount - 1] = i;
		}
	}
	for(int t = observationCount - 2; t >= 0; t--){
		bestStates[t] = psi[t+1][bestStates[t+1]];
	}
	return p_star;
}



