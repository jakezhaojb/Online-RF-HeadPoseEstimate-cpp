#ifndef CLASSIFIER_H_
#define CLASSIFIER_H_

#include <vector>

#include "data.h"

using namespace std;

class Classifier {
public:
    virtual void update(Sample &sample,DataSet &dataset) = 0;
    virtual void train(DataSet &dataset_tr, DataSet &dataset_ts) = 0;
    virtual Result eval(Sample &sample) = 0;
    virtual vector<Result> test(DataSet & dataset) = 0;
    virtual vector<double> trainAndTest(DataSet &dataset_tr, DataSet &dataset_ts) = 0;

   
};

#endif /* CLASSIFIER_H_ */
