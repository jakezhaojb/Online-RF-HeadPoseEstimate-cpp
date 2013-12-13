#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <stdio.h>
#include <cmath>
#include <vector>
#include "data.h"
#include <algorithm>
#include "hp.h"
#include "onlinetree.h"
#include "randomerf.h"
using namespace std;

int main()
{

	// Load the hyperparameters
	string confFileName="config.txt";
	Hyperparameters hp(confFileName); // Read config-file 
	fstream fout; 
	fout.open("result.txt",ios::out); // Save the results
	
	// Starts
	DataSet *dataset_train, *dataset_test;
	string trainfile, testfile, basetr0, basetr1, basete0, basete1;
	basete0 = "test\\01samples";
	basete1 = ".txt";
	int sample_groupsnum = 9; // Training exploits 10 samples
	trainfile = "train\\01samples0.txt"; 
	dataset_test = new DataSet;
	(*dataset_train).loadLIBSVM(trainfile);   // Load initial train data to start On-line training
	OnlineRF model(hp, (*dataset_train).m_numClasses);
    model.train(*dataset_train, *dataset_test);
	vector<double> result;

	// Samples arrive one-by-one
	for(int i = 1 ; i < sample_groupsnum ; i++){ 
	
		stringstream s0;
        s0 << i;
        string str0 = s0.str();
		trainfile = basete0 + str0 + basete1; 
		(*dataset_train).loadLIBSVM(trainfile);
		model.train(*dataset_train, *dataset_test);
		
	 }
	
	for(int i = 1 ; i < 5 ; i = i + 2){ // Testing uses 5 samples

        stringstream s0;
        s0 << i;
        string str0 = s0.str();
		testfile = basete0 + str0 + basete1;
		(*dataset_test).loadLIBSVM(testfile);
	    result = model.test(*dataset_test);
		fout<<"prediction: "<<result[0]<<" "<<result[1]<<" "<<result[2]<<" "<<result[3]<<" "<<result[4]<<" "<<result[5]<<" "<<endl;
	}
 
	delete dataset_train;
    delete dataset_test;
	f.close();
	   
}
		
	
	
	