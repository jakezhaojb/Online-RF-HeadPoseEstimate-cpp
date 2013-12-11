#ifndef RANDOMERF_H_
#define RANDOMERF_H_
#include "classifier.h"
#include "data.h"
#include "hp.h"
#include "onlinetree.h"
#include "utilities.h"
#include <math.h>
#include <vector>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;


class OnlineRF: public Classifier {
public:
	OnlineRF(const Hyperparameters &hp, const int &numClasses) :m_numClasses(&numClasses), m_counter(0.0), m_hp(&hp) {
		OnlineTree *tree;
		for(int i = 0; i < hp.numTrees; i++) {
			tree = new OnlineTree(hp, numClasses);
			m_trees.push_back(tree);
		}
	}

	~OnlineRF() {
		for(int i = 0; i < m_hp->numTrees; i++) {
			delete m_trees[i];
		}
	}

	virtual void update(Sample &sample,DataSet &dataset_tr) {
		m_counter += sample.w;
		Result result, treeResult;
		for(int i = 0; i < *m_numClasses; i++) {
			result.confidence.push_back(0.0);
		}


		int numTries;
		for(int i = 0; i < m_hp->numTrees; i++) {
			numTries = poisson(1.0);
			if (numTries) {
				for(int n = 0; n < numTries; n++) {
					m_trees[i]->update(sample,dataset_tr);
				}
			} else {
				treeResult = m_trees[i]->eval(sample);
			}
		}
	}

	virtual Result eval(Sample &sample) {
		Result result, treeResult;
		for(int i = 0; i < *m_numClasses; i++) {
			result.confidence.push_back(0.0);
		}

		vector<float> sumvote(6,0);
		for(int i = 0; i < m_hp->numTrees; i++) {
			treeResult = m_trees[i]->eval(sample);
			for(int j=0;j<6;j++)
				sumvote[j]+=treeResult.prediction[j];
		}

		vector<float> finale(6,0.0);
		for(int i=0;i<6;i++){
			finale[i]=sumvote[i]/ m_hp->numTrees;
			cout<<"prediction: "<<finale[i]<<" contrast with the standard: "<<sample.y[i]<<endl;
		}
		result.prediction=finale;
		return result;
	}

	virtual vector<double> test(DataSet &dataset) {
		Result result, treeResult;
		vector<vector<float>> estimation;
		for(int i = 0; i < dataset.m_numSamples; i++)
		{

			for(int i = 0; i < m_hp->numTrees; i++) 
			{
				treeResult = m_trees[i]->eval(dataset.m_samples[i]);
				estimation.push_back(treeResult.prediction);
			}
		}

		// Cluster
		vector<vector<vector<float>>> temp_clusters;
		vector<vector<float>> cluster_means;
		for(unsigned int l=0;l<estimation.size();++l){
			bool found = false;
			unsigned int best_cluster = 0;
			// For each cluster
			for(unsigned int c=0; ( c<cluster_means.size() && found==false ); ++c){
				float norm = 0;
				for(int n=0;n<3;++n)
					norm += (estimation[l][n]-cluster_means[c][n])*(estimation[l][n]-cluster_means[c][n]);//
				if( norm < 21830.063 ){
					best_cluster = c;
					found = true;
					temp_clusters[best_cluster].push_back(estimation[l]);

					// Update mean
					for(int i=0;i<6;i++)
					{
						cluster_means[best_cluster][i]=0;
					}
					if ( temp_clusters[best_cluster].size() > 0 ){

						for(int i=0;i<temp_clusters[best_cluster].size();i++)
						{
							cluster_means[best_cluster][0]=cluster_means[best_cluster][0]+ temp_clusters[best_cluster][i][0];
							cluster_means[best_cluster][1]=cluster_means[best_cluster][1]+ temp_clusters[best_cluster][i][1];
							cluster_means[best_cluster][2]=cluster_means[best_cluster][2]+ temp_clusters[best_cluster][i][2];
							cluster_means[best_cluster][3]=cluster_means[best_cluster][3]+ temp_clusters[best_cluster][i][3];
							cluster_means[best_cluster][4]=cluster_means[best_cluster][4]+ temp_clusters[best_cluster][i][4];
							cluster_means[best_cluster][5]=cluster_means[best_cluster][5]+ temp_clusters[best_cluster][i][5];

						}

						float div = float(MAX(1,temp_clusters[best_cluster].size()));

						for(int n=0;n<6;++n)
							cluster_means[best_cluster][n] /= div;
					}

				}

			}


			// Create a new cluster
			if( !found && temp_clusters.size() < 20 ){

				vector<vector< float>> new_cluster;
				new_cluster.push_back(estimation[l]);
				temp_clusters.push_back( new_cluster );
				cluster_means.push_back( estimation[l]);

			}

		}

		vector<vector<vector<float >>> new_clusters; 
		vector<vector<float>> new_means;
		Vec<float,6> temp_mean;
		int count = 0;
		float ms_radius2 = 2235;
		int th = 80;
		for(unsigned int c=0;c<cluster_means.size();++c){
			if ( temp_clusters[c].size() <= th ){
				cout << "skipping cluster " << endl;
				continue;
			}
			vector<vector<float >> new_cluster;
			for(unsigned int it=0; it<10; ++it){
				count = 0;
				temp_mean = 0;
				new_cluster.clear();
				// For each vote in the cluster
				for(unsigned int idx=0; idx < temp_clusters[c].size() ;++idx){

					float norm = 0;
					for(int n=0;n<3;++n)
						norm += (temp_clusters[c][idx][n]-cluster_means[c][n])*(temp_clusters[c][idx][n]-cluster_means[c][n]);

					if( norm < ms_radius2 ){

						temp_mean[0] = temp_mean[0] + temp_clusters[c][idx][0];
						temp_mean[1] = temp_mean[1] + temp_clusters[c][idx][1];
						temp_mean[2] = temp_mean[2] + temp_clusters[c][idx][2];
						temp_mean[3] = temp_mean[3] + temp_clusters[c][idx][3];
						temp_mean[4] = temp_mean[4] + temp_clusters[c][idx][4];
						temp_mean[5] = temp_mean[5] + temp_clusters[c][idx][5];
						new_cluster.push_back( temp_clusters[c][idx] );
						count++;

					}

				}
				for(int n=0;n<6;++n)
					temp_mean[n] /= (float)MAX(1,count);

				float distance_to_previous_mean2 = 0;
				for(int n=0;n<6;++n){
					distance_to_previous_mean2 += (temp_mean[n]-cluster_means[c][n])*(temp_mean[n]-cluster_means[c][n]);
				}

				// Update the mean
				cluster_means[c][0] = temp_mean[0];
				cluster_means[c][1] = temp_mean[1];
				cluster_means[c][2] = temp_mean[2];
				cluster_means[c][3] = temp_mean[3];
				cluster_means[c][4] = temp_mean[4];
				cluster_means[c][5] = temp_mean[5];

				if( distance_to_previous_mean2 < 1 )
					break;
			}

			new_clusters.push_back( new_cluster );
			new_means.push_back( cluster_means[c] );

		}
		vector<vector<float>> means;
		vector<vector<vector<float>>> clusters;
		for(unsigned int c=0; c < new_clusters.size() ;++c){
			if( new_clusters[c].size() < th ) // Discard clusters with insufficient votes
				continue;

			vector< vector<float >> cluster;
			cluster_means[c][0] = 0;
			cluster_means[c][1] = 0;
			cluster_means[c][2] = 0;
			cluster_means[c][3] = 0;
			cluster_means[c][4] = 0;
			cluster_means[c][5] = 0;

			// For each vote in the cluster
			for(unsigned int k=0; k < new_clusters[c].size(); k++ ){
				cluster_means[c][0] = cluster_means[c][0] + new_clusters[c][k][0];
				cluster_means[c][1] = cluster_means[c][1] + new_clusters[c][k][1];
				cluster_means[c][2] = cluster_means[c][2] + new_clusters[c][k][2];
				cluster_means[c][3] = cluster_means[c][3] + new_clusters[c][k][3];
				cluster_means[c][4] = cluster_means[c][4] + new_clusters[c][k][4];
				cluster_means[c][5] = cluster_means[c][5] + new_clusters[c][k][5];

				cluster.push_back( new_clusters[c][k]) ;

			}


			float div = (float)MAX(1,new_clusters[c].size());
			for(int n=0;n<6;++n)
				cluster_means[c][n] /= div;

			means.push_back( cluster_means[c] );
			clusters.push_back( cluster );

		}
		vector<double> re;
		for(int i=0;i<6;i++)
			re.push_back(cluster_means[0][i]);
		cout<<"prediction: "<<means[0][0]<<" "<<means[0][1]<<" "<<means[0][2]<<" "<<means[0][3]<<" "<<means[0][4]<<" "<<means[0][5]<<" "<<endl;
		return re;
	}




	virtual vector<Result> test(DataSet &dataset) {
		vector<Result> results;
		for(int i = 0; i < dataset.m_numSamples; i++) {
			results.push_back(eval(dataset.m_samples[i]));
		}

		return results;
	}

	virtual void train(DataSet &dataset_tr, DataSet &dataset_ts) {
		vector<int> randIndex;
		int sampRatio = dataset_tr.m_numSamples / 10;
		for(int n = 0; n <  m_hp->numEpochs; n++) {
			randPerm(dataset_tr.m_numSamples, randIndex);
			for(int i = 0; i < dataset_tr.m_numSamples; i++) {
				update(dataset_tr.m_samples[randIndex[i]],dataset_tr);
				cout<<i<<endl;
			}
		}
	}

	virtual vector<double> trainAndTest(DataSet &dataset_tr, DataSet &dataset_ts) {
		vector<Result> results;
		vector<int> randIndex;
		for(int n = 0; n <m_hp->numEpochs; n++) {
			randPerm(dataset_tr.m_numSamples, randIndex);
			for(int i = 0; i < dataset_tr.m_numSamples; i++) {
				update(dataset_tr.m_samples[randIndex[i]],dataset_tr);
				cout<<i<<endl;
			}
			vector<double> re;
			re=test(dataset_ts);
			return re;
		}
	}


	const int *m_numClasses;
	double m_counter;
	const Hyperparameters *m_hp;

	vector<OnlineTree*> m_trees;
};

#endif /* RANDOMERF_H_ */
