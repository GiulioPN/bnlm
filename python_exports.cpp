#include <pybind11/pybind11.h>

#include <iostream>
#include <unordered_map>
#include <cstdlib>
#include <fstream>
#include <string>

#include "src/hpyplm.h" 
#include "src/corpus.h"
#include "src/m.h"
#include "src/random.h"
#include "src/crp.h"
#include "src/tied_parameter_resampler.h"

#define kORDER 3 //servirà una factory per gestirlo dinamicamente

namespace py = pybind11;

using namespace std;
using namespace cpyp;


void hpypTrainTest(string train_file, string test_file, int samples, double da, double db, double ss, double sr, string estimations_txt, string performance_txt){
  MT19937 eng;
  	
  Dict dict;
  vector<double>  llikelihoods;
  vector<vector<unsigned> > corpuse;
	set<unsigned> vocabe, tv;
  	const unsigned kSOS = dict.Convert("<s>");
  	const unsigned kEOS = dict.Convert("</s>");
	//cerr << "Reading corpus...\n";
  	
  ReadFromFile(train_file, &dict, &corpuse, &vocabe);

  vector<vector<unsigned> > test;
  ReadFromFile(test_file, &dict, &test, &tv);
  PYPLM<kORDER> lm(vocabe.size(), da, db, ss, sr);
  	
  //Starting Train Model 
  vector<unsigned> ctx(kORDER - 1, kSOS);
  for (int sample=0; sample < samples; ++sample) {
  	for (const auto& s : corpuse) {
    		ctx.resize(kORDER - 1);
    		for (unsigned i = 0; i <= s.size(); ++i) {
      		unsigned w = (i < s.size() ? s[i] : kEOS);
      		if (sample > 0) lm.decrement(w, ctx, eng);
      		lm.increment(w, ctx, eng);
      		ctx.push_back(w);
    		}	
  	}
  	if (sample % 10 == 9) {
    		//cerr << " [LLH=" << lm.log_likelihood() << "]" << endl;
    		if (sample % 30u == 29) lm.resample_hyperparameters(eng);
  	} //else { cerr << '.' << flush; }
    if(sample>80) llikelihoods.push_back(lm.log_likelihood());

  }
  	
  std::ofstream fileLoglikelihood("log_likelihood.txt");
  if(fileLoglikelihood.is_open()){
    for(int i=0; i< llikelihoods.size(); i++)
      fileLoglikelihood << llikelihoods[i] << "\n";
  }
  fileLoglikelihood.close();

	//Starting Test Model 
  double llh = 0;
  unsigned cnt = 0;
  unsigned oovs = 0;

  std::ofstream fileEsitimations(estimations_txt);
  if(fileEsitimations.is_open()){
    for (auto& s : test) {
    	ctx.resize(kORDER - 1);
    	for (unsigned i = 0; i <= s.size(); ++i) {
    		unsigned w = (i < s.size() ? s[i] : kEOS);
    		double lp = log(lm.prob(w, ctx)) / log(2);
      		if (i < s.size() && vocabe.count(w) == 0) {
        		fileEsitimations << "**OOV ";
        		++oovs;
        		lp = 0;
      		}
      		fileEsitimations << "p(" << dict.Convert(w) << " |";
      		for (unsigned j = ctx.size() + 1 - kORDER; j < ctx.size(); ++j)
        		fileEsitimations << ' ' << dict.Convert(ctx[j]);
      			fileEsitimations << ") = " << lp << "\n";
      		ctx.push_back(w);
      		llh -= lp;
      		cnt++;
    	}
    }
  }
  fileEsitimations.close();

  std::ofstream filePerformance(performance_txt);
  if(filePerformance.is_open()){
    cnt -= oovs;
    filePerformance << "E-corpus size: " << corpuse.size() << " sentences\t (" << vocabe.size() << " word types)\n";
    filePerformance << "  Log_10 prob: " << (-llh * log(2) / log(10)) << "\n";
    filePerformance << "        Count: " << cnt << "\n";
    filePerformance << "         OOVs: " << oovs << "\n";
    filePerformance << "Cross-entropy: " << (llh / cnt) << "\n";
    filePerformance << "   Perplexity: " << pow(2, llh / cnt) << "\n";
  }
  filePerformance.close();
}




PYBIND11_MODULE(bnplm, m) {
    m.doc() = "Easy Life"; // optional module docstring

    m.def("hpypTT", &hpypTrainTest,
          "train and test the hpyp model");

}
