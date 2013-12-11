#ifndef DATA_H_
#define DATA_H_

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <stdlib.h>


using namespace std;

// TYPEDEFS
typedef vector<double> Label;
typedef double Weight;
typedef int mark;
typedef vector<double> SparseVector;

// DATA CLASSES
class Sample {
public:
    SparseVector x;
    Label y;
    Weight w;
	mark ma;
};

class DataSet {
public:
    vector<Sample> m_samples;
    int m_numSamples;
    int m_numFeatures;
    int m_numClasses;

    void loadLIBSVM(string filename)
	{
    ifstream fp(filename.c_str(), ios::binary);
    if (!fp) {
        cout << "Could not open input file " << filename << endl;
        exit(EXIT_FAILURE);
    }

    cout << "Loading data file: " << filename << " ... " << endl;
	
	// Reading the header
    int startIndex;
    string line, tmpStr;
	fp >> m_numSamples;
    fp >> m_numFeatures;
    fp >> m_numClasses;
    fp >> startIndex;
    int prePos, curPos;
    m_samples.clear();

    for (int i = 0; i < m_numSamples; i++) {
        vector<double> x(m_numFeatures);
        Sample sample;
		sample.y.resize(6);
		fp>>sample.ma;
		if(sample.ma>=m_numSamples)
			bool a=true;
        fp>>sample.y[0]; // read labels
		fp>>sample.y[1];
		fp>>sample.y[2];
		fp>>sample.y[3]; 
		fp>>sample.y[4];
		fp>>sample.y[5];
        sample.w = 1.0;  // set weight
		sample.x.resize(m_numFeatures);
        getline(fp, line); // read the rest of the line
        prePos = 0;
        curPos = line.find(' ', 0);
   
		for(int i=0;i<6400;i++)
		{
			 prePos = curPos + 1;
             curPos = line.find(' ', prePos);
			 tmpStr = line.substr(prePos, curPos - prePos);
			 sample.x[i]=atof(tmpStr.c_str());
		}
	
        m_samples.push_back(sample); // push sample into dataset
    }

    fp.close();

    if (m_numSamples != (int) m_samples.size()) {
        cout << "Could not load " << m_numSamples << " samples from " << filename;
        cout << ". There were only " << m_samples.size() << " samples!" << endl;
        exit(EXIT_FAILURE);
    }

    cout << "Loaded " << m_numSamples << " samples with " << m_numFeatures;
    cout << " features and " << m_numClasses << " classes." << endl;
}
};

class Result {
public:
    vector<double> confidence;
    vector<float> prediction;
};

#endif /* DATA_H_ */
