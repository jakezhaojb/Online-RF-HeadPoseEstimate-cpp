#ifndef HYPERPARAMETERS_H_
#define HYPERPARAMETERS_H_

#include <string>
using namespace std;
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <stdio.h>
#include <cmath>
using namespace std;

class Hyperparameters
{
 public:
    Hyperparameters();
    Hyperparameters(const string& confFile){
    cout << "Loading config file: " << confFile << " ... ";

    ifstream in(confFile);
	string dummy;

	if(in.is_open()) {

			
		in >> dummy;
		in >> maxDepth;

		// Number of trees
		in >> dummy;
		in >> numRandomTests;

		in >> dummy;
		in >> counterThreshold;

		in >> dummy;
		in >> numTrees;

		in >> dummy;
		in >>  numEpochs;


	} else {
		cerr << "File not found "  << endl;
		exit(-1);
	}
	in.close();
    cout << "Done." << endl;
}


    // Online node
    int numRandomTests;
    int counterThreshold;
    int maxDepth;

    // Online forest
    int numTrees;
    int numEpochs;

    // Data
    string trainData;
    string testData;

    // Output
    int verbose;
};

#endif /*HYPERPARAMETERS_H_*/
