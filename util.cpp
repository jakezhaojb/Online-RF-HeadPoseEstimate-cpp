#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <cmath>
#ifndef WIN32

#endif

#include "utilities.h"

using namespace std;

void randPerm(const int &inNum, vector<int> &outVect) {
    outVect.resize(inNum);
    int randIndex, tempIndex;
    for (int nFeat = 0; nFeat < inNum; nFeat++) {
        outVect[nFeat] = nFeat;
    }
	bool de=false;
    for (register int nFeat = 0; nFeat < inNum; nFeat++) {
        randIndex = (int) floor(((double) inNum - nFeat) * randDouble()) + nFeat;
        if (randIndex == inNum) {
            randIndex--;
        }
        tempIndex = outVect[nFeat];
        outVect[nFeat] = outVect[randIndex];
        outVect[randIndex] = tempIndex;
		if(outVect[randIndex]>=inNum)
			 de=true;
    }
}

void randPerm(const int &inNum, const int inPart, vector<int> &outVect) {
    outVect.resize(inNum);
    int randIndex, tempIndex;
    for (int nFeat = 0; nFeat < inNum; nFeat++) {
        outVect[nFeat] = nFeat;
    }
    for (register int nFeat = 0; nFeat < inPart; nFeat++) {
        randIndex = (int) floor(((double) inNum - nFeat) * randDouble()) + nFeat;
        if (randIndex == inNum) {
            randIndex--;
        }
        tempIndex = outVect[nFeat];
        outVect[nFeat] = outVect[randIndex];
        outVect[randIndex] = tempIndex;
    }

    outVect.erase(outVect.begin() + inPart, outVect.end());
}