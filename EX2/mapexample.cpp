/* 
 * File:   main.cpp
 * Author: Rodriguez
 *
 * Created on May 15, 2018, 8:41 PM
 */

#include <map>
#include <iostream>
using namespace std;

int main(int argc, char** argv) {

    map<double, int> distanceMap;
    double vec[4];
    
    distanceMap.insert( pair<double, int>(1.4,1) );
    distanceMap.insert( pair<double, int>(1.2,2) );
    distanceMap.insert( pair<double, int>(1.1,3) );
    distanceMap.insert( pair<double, int>(1.6,4) );
    distanceMap.insert( pair<double, int>(1.3,5) );
    map<double, int>::iterator it = distanceMap.begin();
    for (int k = 0; k < 5; k++)
    {
	cout << "Value -> " << it->first;
        cout << " | "<< it->second << " <- Id"<< endl;
	it++;
    }
    return 0;
}

