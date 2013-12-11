#ifndef ONLINENODE_H_
#define ONLINENODE_H_

#include <vector>
#include "data.h"
#include "hp.h"
#include "randomtest.h"
#include "utilities.h"

using namespace std;

class OnlineNode {
public:
    OnlineNode() {
        m_isLeaf = true;
    }

    OnlineNode(const Hyperparameters &hp, const int &numClasses, const int &depth) :
        m_numClasses(&numClasses), m_depth(depth), label(6,0), m_isLeaf(true), m_counter(0.0), m_label(6,-1),
                m_parentCounter(0.0), m_hp(&hp) {
					for (int i = 0; i <numClasses; i++) {
            m_labelStats.push_back(0.0);

        }

        // Creating random tests
       for (int i = 0; i < hp.numRandomTests; i++) {
            HyperplaneFeature test(numClasses);
            m_onlineTests.push_back(test);
        }
    }
    int m_transformlabel;
	
    OnlineNode(const Hyperparameters &hp, const int &numClasses, const int &depth, const vector<double> &parentStats,vector<float> extantlabel) :
        m_numClasses(&numClasses),m_depth(depth), label(6,0.0),m_isLeaf(true), m_counter(0.0), m_label(6,-1),
                m_parentCounter(0.0), m_hp(&hp){
        m_labelStats = parentStats;
        m_label = extantlabel;
	    m_parentCounter = sum(m_labelStats);

        // Creating random tests
       for (int i = 0; i < hp.numRandomTests; i++) {
            HyperplaneFeature test(numClasses);
			m_onlineTests.push_back(test);
        }
    }

    ~OnlineNode() {
        if (!m_isLeaf) {
            delete m_leftChildNode;
            delete m_rightChildNode;
        }
    }
    vector<float> label;
   
	void update(Sample &sample,DataSet &dataset_tr){
    m_counter += sample.w;
    m_labelStats[sample.ma] += sample.w;
	for(int i=0;i<6;i++)
	{
	label[i]+=sample.y[i];
	}

    if (m_isLeaf) {

        // Update online tests
        for (int i = 0; i < m_hp->numRandomTests; i++) {
            m_onlineTests[i].update(sample);
        }

		for(int i=0;i<6;i++){
		m_label[i]=label[i]/m_counter;
		}

	    // Decide for split
        if (shouldISplit()) {
            m_isLeaf = false;

            // Find the best online test
            int maxIndex = 0;
            double maxScore = -1e100, score;
            for (int i = 0; i < m_hp->numRandomTests; i++) {
                score = m_onlineTests[i].score(dataset_tr);
                if (score > maxScore) {
                    maxScore = score;
                    maxIndex = i;
                }
			}
            m_bestTest = m_onlineTests[maxIndex];
            m_onlineTests.clear();

            // Split		
            pair<vector<double> , vector<double> > parentStats = m_bestTest.getStats();
            m_rightChildNode = new OnlineNode(*m_hp, *m_numClasses, m_depth + 1,parentStats.first,m_label);
            m_leftChildNode = new OnlineNode(*m_hp, *m_numClasses,m_depth + 1,parentStats.second,m_label);
		    }
	}
     else {
        if (m_bestTest.eval(sample)) {
            m_rightChildNode->update(sample,dataset_tr);
        } else {
            m_leftChildNode->update(sample,dataset_tr);
        }
    }
}

    Result eval(Sample &sample) {
        if (m_isLeaf) {
            Result result;
            if (m_counter + m_parentCounter) {
                result.confidence = m_labelStats;
                result.prediction = m_label;
            } else {
                for (int i = 0; i < *m_numClasses; i++) {
                    result.confidence.push_back(1.0 / *m_numClasses);
                }
				for(int i=0;i<6;i++)
                  result.prediction.push_back (0.0);
				
            }

            return result;
        } else {
            if (m_bestTest.eval(sample)) {
                return m_rightChildNode->eval(sample);
            } else {
                return m_leftChildNode->eval(sample);
            }
        }
    }

private:
    const int *m_numClasses;
    int m_depth;
    bool m_isLeaf;
    double m_counter;
    vector<float> m_label;
    double m_parentCounter;
    const Hyperparameters *m_hp;
    vector<double> m_labelStats;
    OnlineNode* m_leftChildNode;
    OnlineNode* m_rightChildNode;

    vector<HyperplaneFeature> m_onlineTests;
    HyperplaneFeature m_bestTest;

    bool shouldISplit() {
        bool isPure = false;
        for (int i = 0; i < *m_numClasses; i++) {
            if (m_labelStats[i] == m_counter + m_parentCounter) {
                isPure = true;
                break;
            }
        }

        if (isPure) {
            return false;
        }

        if (m_depth >= m_hp->maxDepth) { // Do not split if the max depth is reached
            return false;
        }

        if (m_counter < m_hp->counterThreshold) { // Do not split if with not enough samples
            return false;
        }

        return true;
    }

};

#endif /* ONLINENODE_H_ */
