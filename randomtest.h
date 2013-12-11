#ifndef RANDOMTEST_H_
#define RANDOMTEST_H_

#include "data.h"
#include "utilities.h"
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;

using namespace std;


class RandomTest {
public:
    RandomTest() {

    }

    RandomTest(const int &numClasses) :
        m_numClasses(&numClasses), m_trueCount(0.0), m_falseCount(0.0),xyzR(3,vector<float>(0)),xyzL(3,vector<float>(0)),thitaR(3,vector<float>(0)),thitaL(3,vector<float>(0)) {
        for (int i = 0; i <numClasses; i++) {
            m_trueStats.push_back(0.0);
            m_falseStats.push_back(0.0);
		}
    }

   
    void updateStats(const Sample &sample, const bool decision) {
        if (decision) {
            m_trueCount += sample.w;
            m_trueStats[sample.ma] += sample.w;
			for(int i=0;i<3;i++){
				xyzR[i].push_back(sample.y[i]);
				thitaR[i].push_back(sample.y[i+3]);
			}
        } else {
            m_falseCount += sample.w;
            m_falseStats[sample.ma] += sample.w;
            for(int i=0;i<3;i++){
				xyzL[i].push_back(sample.y[i]);
				thitaL[i].push_back(sample.y[i+3]);
			}
		}
    }

    double score(DataSet &dataset_tr) {
        double totalCount = m_trueCount + m_falseCount;
		double p=1/totalCount;
        double p1,p2,splitEntropy = 0.0;
        if (m_trueCount) {
            p1 = m_trueCount / totalCount;
            splitEntropy -= p1 * log(p)/log(2.0);
        }
		if (m_falseCount) {
            p2 = m_falseCount / totalCount;
            splitEntropy -= p2 * log(p)/log(2.0);
        }
		double p3=1/m_trueCount, p4=1/m_falseCount;
		vector<double>m_totalStats(*m_numClasses,0.0);
        for (int i = 0; i < *m_numClasses; i++) 
          m_totalStats[i] = (m_trueStats[i] + m_falseStats[i]);
		double total=0;
		for(int i = 0; i < *m_numClasses; i++)
			total+= m_totalStats[i];
		double Entropytotal=0;
		for(int i = 0; i < *m_numClasses; i++)
			if(m_totalStats[i])
			{
				Entropytotal-= (m_totalStats[i]/total)*log((m_totalStats[i]/total))/log(2.0);

			}
		int counterR=0,counterL=0;
		int rowR=0,rowL=0;
		float avgxyzR[3]={0,0,0},avgxyzL[3]={0,0,0},avgthitaR[3]={0,0,0},avgthitaL[3]={0,0,0};
		for(int i=0;i<m_trueCount;i++)
		{
			avgxyzR[0]+=xyzR[0][i];
			avgxyzR[1]+=xyzR[1][i];
			avgxyzR[2]+=xyzR[2][i];
			avgthitaR[0]+=thitaR[0][i];
			avgthitaR[1]+=thitaR[0][i];
			avgthitaR[2]+=thitaR[0][i];
        }
		
		for(int i=0;i<m_falseCount;i++)
		{
			avgxyzL[0]+=xyzL[0][i];
			avgxyzL[1]+=xyzL[1][i];
			avgxyzL[2]+=xyzL[2][i];
			avgthitaL[0]+=thitaL[0][i];
			avgthitaL[1]+=thitaL[0][i];
			avgthitaL[2]+=thitaL[0][i];
        }
		for(int i=0;i<3;i++)
	   {
		   if(m_trueCount)
		   {
		   avgxyzR[i]=avgxyzR[i]/m_trueCount;
		   avgthitaR[i]=avgthitaR[i]/m_trueCount;
		   }
		   if(m_falseCount)
		   {
		   avgxyzL[i]=avgxyzL[i]/m_falseCount;
		   avgthitaL[i]=avgthitaL[i]/m_falseCount;
		   }
		   }
		
		float covxyzR[3][3]={0},covthitaR[3][3]={0},covxyzL[3][3]={0},covthitaL[3][3]={0};
		for(int i=0,j=0;i<3,j<3;i++,j++)
			for(int q=0;q<m_trueCount;q++)
			{
				 covxyzR[i][j]+=(xyzR[i][q]-avgxyzR[i])*(xyzR[j][q]-avgxyzR[j]);
				 covthitaR[i][j]+=(thitaR[i][q]-avgthitaR[i])*(thitaR[j][q]-avgthitaR[j]);
				 
			}
			if(m_trueCount)
				for(int i=0;i<3;i++)
			     for(int j=0;j<3;j++)
				 {
					
					covxyzR[i][j]=covxyzR[i][j]/m_trueCount;
		            covthitaR[i][j]=covthitaR[i][j]/m_trueCount;
			     }	
			
		    for(int i=0,j=0;i<3,j<3;i++,j++)
			for(int q=0;q<m_falseCount;q++)
			{
				 covxyzL[i][j]+=(xyzL[i][q]-avgxyzL[i])*(xyzL[j][q]-avgxyzL[j]);
				 covthitaL[i][j]+=(thitaL[i][q]-avgthitaL[i])*(thitaL[j][q]-avgthitaL[j]);
				 
			}
			
			if(m_falseCount)
              for(int i=0;i<3;i++)
			    for(int j=0;j<3;j++)
				{
					covxyzL[i][j]=covxyzL[i][j]/m_falseCount;
		            covthitaL[i][j]=covthitaL[i][j]/m_falseCount;
			    }
			
			float hlsxyzR=sqrt(covxyzR[0][0]*covxyzR[1][1]*covxyzR[2][2])+sqrt(covxyzR[0][1]*covxyzR[1][2]*covxyzR[2][0])+sqrt(covxyzR[0][2]*covxyzR[1][0]*covxyzR[2][1])-sqrt(covxyzR[0][2]*covxyzR[1][1]*covxyzR[2][0])-sqrt(covxyzR[0][0]*covxyzR[1][2]*covxyzR[2][1])-sqrt(covxyzR[0][1]*covxyzR[1][0]*covxyzR[2][2]);
			float hlsxyzL=sqrt(covxyzL[0][0]*covxyzL[1][1]*covxyzL[2][2])+sqrt(covxyzL[0][1]*covxyzL[1][2]*covxyzL[2][0])+sqrt(covxyzL[0][2]*covxyzL[1][0]*covxyzL[2][1])-sqrt(covxyzL[0][2]*covxyzL[1][1]*covxyzL[2][0])-sqrt(covxyzL[0][0]*covxyzL[1][2]*covxyzL[2][1])-sqrt(covxyzL[0][1]*covxyzL[1][0]*covxyzL[2][2]);
			float hlsthitaR=sqrt(covthitaR[0][0]*covthitaR[1][1]*covthitaR[2][2])+sqrt(covthitaR[0][1]*covthitaR[1][2]*covthitaR[2][0])+sqrt(covthitaR[0][2]*covthitaR[1][0]*covthitaR[2][1])-sqrt(covthitaR[0][2]*covthitaR[1][1]*covthitaR[2][0])-sqrt(covthitaR[0][0]*covthitaR[1][2]*covthitaR[2][1])-sqrt(covthitaR[0][1]*covthitaR[1][0]*covthitaR[2][2]);
            float hlsthitaL=sqrt(covthitaL[0][0]*covthitaL[1][1]*covthitaL[2][2])+sqrt(covthitaL[0][1]*covthitaL[1][2]*covthitaL[2][0])+sqrt(covthitaL[0][2]*covthitaL[1][0]*covthitaL[2][1])-sqrt(covthitaL[0][2]*covthitaL[1][1]*covthitaL[2][0])-sqrt(covthitaL[0][0]*covthitaL[1][2]*covthitaL[2][1])-sqrt(covthitaL[0][1]*covthitaL[1][0]*covthitaL[2][2]);
            
			double result=Entropytotal-p3*(hlsxyzR+hlsthitaR)-p4*(hlsxyzL+hlsthitaL);
			return result;
	}

    pair<vector<double> , vector<double> > getStats() {
        return pair<vector<double> , vector<double> > (m_trueStats, m_falseStats);
    }

protected:
    const int *m_numClasses;
    double m_threshold;
    double m_trueCount;
    double m_falseCount;
    vector<double> m_trueStats;
    vector<double> m_falseStats;
	vector<vector<float>> xyzR;
	vector<vector<float>> xyzL;
	vector<vector<float>> thitaR;
	vector<vector<float>> thitaL;
		
};

class HyperplaneFeature: public RandomTest {
public:
    HyperplaneFeature() {

    }
	
    HyperplaneFeature(const int &numClasses) :
        RandomTest(numClasses),test_point(20) {
			 }
     
     int test_point;
	 vector<double> dif_squ_rec; 
	
	
    void update(Sample &sample) {
        updateStats(sample, eval(sample));
    }
  bool eval(Sample &sample) {

	dif_squ_rec.resize(test_point/2);
	
	int p_width=80,p_height=80;
	
	int side=5;
	int q=0;

	vector<vector<double>> patches(p_width,vector<double>(p_height));
		for (int j=0;j<p_width;j++)
			for (int k=0;k<p_height;k++){
				patches[j][k]=sample.x[q];
				q++;
			}
	
    vector<Rect> points(test_point);
		for (int j=0;j<test_point;j++)
		{
			points[j].x=(int)randomFromRange(0,p_width-side);
			points[j].y=(int)randomFromRange(0,p_height-side);
			points[j].width=side;
			points[j].height=side;
		}

    vector<vector<vector<int>>> test_rec(test_point,vector<vector<int>>(side,vector<int>(side)));
    
		for (int j=0;j<test_point;j++)
               			for(int l=0;l<side;l++)
				for(int m=0;m<side;m++)
					test_rec[j][l][m]=patches[points[j].x+l][points[j].y+m];//i表示第i个patch，j若为双数，则表示第（j+2）/2对矩形中的第一个矩形
   
	 vector<vector<vector<int>>> dif_rec(test_point/2,vector<vector<int>>(side,vector<int>(side)));
		for (int j=0;j<test_point;j=j+2)
			for(int l=0;l<side;l++)
				for(int m=0;m<side;m++)
					dif_rec[j/2][l][m]=test_rec[j][l][m]-test_rec[j+1][l][m];
	 
		for (int j=0;j<test_point/2;j++)
			dif_squ_rec[j]=0;

			for (int j=0;j<test_point/2;j++)
				for(int l=0;l<side;l++)
				for(int m=0;m<side;m++)
						dif_squ_rec[j]+=sqrt((double)dif_rec[j][l][m]*dif_rec[j][l][m]);
		 
     int indexj=0;     
	 int maxvalue=0; 
	 int minvalue=0;
	 
	 for (int j=0;j<test_point/2;j++)
		{
			if(dif_squ_rec[j]>maxvalue)
				{
					maxvalue=dif_squ_rec[j];
					indexj=j;
			    }
			if(dif_squ_rec[j]<minvalue)
			    {
			    	minvalue=dif_squ_rec[j];
			    }

	    }
	 m_threshold = randomFromRange(minvalue, maxvalue);
	 int testsel=(int)randomFromRange(0,test_point/2);
	 double proj=dif_squ_rec[testsel];
	 return (proj >=m_threshold) ? true : false;

}




};

#endif /* RANDOMTEST_H_ */
