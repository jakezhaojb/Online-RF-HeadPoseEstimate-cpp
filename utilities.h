#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#ifndef WIN32
#endif

using namespace std;

//! Returns a random number in [0, 1]
inline double randDouble() {
    
    return rand() / (RAND_MAX + 1.0);
}

//! Returns a random number in [min, max]
inline double randomFromRange(const double &minRange, const double &maxRange) {
    return minRange + (maxRange - minRange) * randDouble();
}

//! Random permutations
void randPerm(const int &inNum, vector<int> &outVect);
void randPerm(const int &inNum, const int inPart, vector<int> &outVect);


inline void fillWithRandomNumbers(const int &length, vector<double> &inVect) {
    inVect.clear();
    for (int i = 0; i < length; i++) {
        inVect.push_back(2.0 * (randDouble() - 0.5));
    }
}


inline double sum(const vector<double> &inVect) {
    double val = 0.0;
    vector<double>::const_iterator itr(inVect.begin()), end(inVect.end());
    while (itr != end) {
        val += *itr;
        ++itr;
    }

    return val;
}

//! Poisson sampling
inline int poisson(double A) {
    int k = 0;
    int maxK = 10;
    while (1) {
        double U_k = randDouble();
        A *= U_k;
        if (k > maxK || A < exp(-1.0)) {
            break;
        }
        k++;
    }
    return k;
}

#endif /* UTILITIES_H_ */
