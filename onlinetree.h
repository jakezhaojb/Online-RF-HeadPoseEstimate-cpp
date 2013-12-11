#ifndef ONLINETREE_H_
#define ONLINETREE_H_

#include <cstdlib>
#include <iostream>

#include "classifier.h"
#include "data.h"
#include "hp.h"
#include "onlinenode.h"

using namespace std;

class OnlineTree: public Classifier {
public:
    OnlineTree(const Hyperparameters &hp, const int &numClasses) :
        m_counter(0.0), m_hp(&hp) {
        m_rootNode = new OnlineNode(hp, numClasses, 0);
    }

		~OnlineTree() {
        delete m_rootNode;
    }
		 virtual void update(Sample &sample,DataSet &dataset_tr) {
        m_rootNode->update(sample,dataset_tr);
    }
		 virtual Result eval(Sample &sample) {
        return m_rootNode->eval(sample);
    }
		  
		 virtual void train(DataSet &dataset_tr, DataSet &dataset_ts) {
        vector<int> randIndex;
        int sampRatio = dataset_tr.m_numSamples / 10;
        for (int n = 0; n < m_hp->numEpochs; n++) {
            randPerm(dataset_tr.m_numSamples, randIndex);
            for (int i = 0; i < dataset_tr.m_numSamples; i++) {
                update(dataset_tr.m_samples[randIndex[i]],dataset_tr);
                if (m_hp->verbose >= 3 && (i % sampRatio) == 0) {
                    cout << "--- Online Random Tree training --- Epoch: " << n + 1 << " --- ";
                    cout << (10 * i) / sampRatio << "%" << endl;
                }
            }
        }
    }

		 virtual vector<double> trainAndTest(DataSet &dataset_tr, DataSet &dataset_ts){
			 vector<double> a(6,0.0);
			 return a;
		    
		 }
		 virtual vector<Result> test(DataSet & dataset){
             vector<Result> results;
        for (int i = 0; i < dataset.m_numSamples; i++) {
            results.push_back(eval(dataset.m_samples[i]));
        }

			 return results;
		 }
   
    double m_counter;
    const Hyperparameters *m_hp;

    OnlineNode* m_rootNode;
};


#endif /* ONLINETREE_H_ */
