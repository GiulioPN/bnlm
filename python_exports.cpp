#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <unordered_map>
#include <cstdlib>
#include <fstream>
#include <string>

#include "src/hpyplm.h" 
#include "src/shpyplm.h" 
#include "src/corpus.h"
#include "src/m.h"
#include "src/random.h"
#include "src/crp.h"
#include "src/tied_parameter_resampler.h"

#define kORDER 2 //implement a factory

namespace py = pybind11;

using namespace std;
using namespace cpyp;

typedef unsigned int uint;


void hpypTrainTest(string train_file, string test_file, uint samples, uint burn_in, double da, double db, double ss, double sr, uint nSimulation){
  MT19937 eng;
  	
  Dict dict;
    const uint kSOS = dict.Convert("<s>");
    const uint kEOS = dict.Convert("</s>");
  
  vector<double>  llikelihoods;
	
  set<uint> train_voc, test_voc;
  
  vector<vector<uint> > train_corpus;
  ReadFromFile(train_file, &dict, &train_corpus, &train_voc);

  vector<vector<uint> > test_corpus;
  ReadFromFile(test_file, &dict, &test_corpus, &test_voc);
  
  PYPLM<kORDER> lm(train_voc.size(), da, db, ss, sr);
  	
  //Starting Train Model 
  vector<uint> ctx(kORDER - 1, kSOS);
  for (uint sample=0; sample < samples; ++sample) {
  	for (const auto& s : train_corpus) {
    		ctx.resize(kORDER - 1);
    		for (uint i = 0; i <= s.size(); ++i) {
      		uint w = (i < s.size() ? s[i] : kEOS);
      		if (sample > 0) lm.decrement(w, ctx, eng);
      		lm.increment(w, ctx, eng);
      		ctx.push_back(w);
    		}	
  	}
  	if (sample % 10 == 9) {
    		cout<<"sample number: "<<sample<<endl;
    		if (sample % 30u == 29) lm.resample_hyperparameters(eng);
  	} //else { cerr << '.' << flush; }
    if(sample>burn_in) llikelihoods.push_back(lm.log_likelihood());

  }

  std::string Loglikelihood_txt = "hpylm_nsim"+std::to_string(nSimulation)+"/LoglikelihoodHPYLM_nSim="+std::to_string(nSimulation)+"_da="+std::to_string(da)+"_db="+std::to_string(db)+"_ss="+std::to_string(ss)+"_sr="+std::to_string(sr)+".txt";  
  std::ofstream fileLoglikelihood(Loglikelihood_txt);
  if(fileLoglikelihood.is_open()){
    for(uint i=0; i< llikelihoods.size(); i++)
      fileLoglikelihood << llikelihoods[i] << "\n";
  }
  fileLoglikelihood.close();

	//Starting Test Model 
  double llh = 0; //sum of log predictive probabilities
  uint cnt = 0; // number of words in test set
  uint oovs = 0; // number of not found word in train set

  std::string estimations_txt = "hpylm_nsim"+std::to_string(nSimulation)+"/estimationsHPYLM_nSim="+std::to_string(nSimulation)+"_da="+std::to_string(da)+"_db="+std::to_string(db)+"_ss="+std::to_string(ss)+"_sr="+std::to_string(sr)+".txt";
  std::ofstream fileEsitimations(estimations_txt);
  if(fileEsitimations.is_open()){
    for (auto& s : test_corpus) {
    	ctx.resize(kORDER - 1);
    	for (uint i = 0; i <= s.size(); ++i) {
    		uint w = (i < s.size() ? s[i] : kEOS);
    		double lp = log(lm.prob(w, ctx)) / log(2);
      		if (i < s.size() && train_voc.count(w) == 0) {
        		fileEsitimations << "**OOV ";
        		++oovs;
        		lp = 0;
      		}
      		fileEsitimations << "p(" << dict.Convert(w) << " |";
      		for (uint j = ctx.size() + 1 - kORDER; j < ctx.size(); ++j)
        		fileEsitimations << ' ' << dict.Convert(ctx[j]);
      			fileEsitimations << ") = " << lp << "\n";
      		ctx.push_back(w);
      		llh -= lp;
      		cnt++;
    	}
    }
  }
  fileEsitimations.close();

  cnt -= oovs;

  std::string performance_txt = "hpylm_nsim"+std::to_string(nSimulation)+"/performanceHPYLM_nSim="+std::to_string(nSimulation)+"_da="+std::to_string(da)+"_db="+std::to_string(db)+"_ss="+std::to_string(ss)+"_sr="+std::to_string(sr)+".txt";
  std::ofstream filePerformance(performance_txt);
  if(filePerformance.is_open()){
    filePerformance << "Train-corpus size: " << train_corpus.size() << " sentences \t unique words: " << train_voc.size() << "\n";
    filePerformance << "Test-corpus size: " << test_corpus.size() << " sentences \t unique words: " << test_voc.size() << "\n";
    filePerformance << "Final Results: \n";
    filePerformance << "  Log_10 prob: " << (-llh * log(2) / log(10)) << "\n";
    filePerformance << "        Count: " << cnt << "\n";
    filePerformance << "         OOVs: " << oovs << "\n";
    filePerformance << "Cross-entropy: " << (llh / cnt) << "\n";
    filePerformance << "   Perplexity: " << pow(2, llh / cnt) << "\n";
  }
  filePerformance.close();

  cout << "Train-corpus size: " << train_corpus.size() << " sentences \t unique words: " << train_voc.size() << "\n";
  cout << "Test-corpus size: " << test_corpus.size() << " sentences \t unique words: " << test_voc.size() << "\n";
  cout << "Number of unique words: " << train_voc.size() << "\n";
  cout << "Final Results: \n";
  cout << "  Log_10 prob: " << (-llh * log(2) / log(10)) << "\n";
  cout << "        Count: " << cnt << "\n";
  cout << "         OOVs: " << oovs << "\n";
  cout << "Cross-entropy: " << (llh / cnt) << "\n";
  cout << "   Perplexity: " << pow(2, llh / cnt) << endl;
}

void shpypTrainTestCpp(vector<string> train_files, vector<string> test_files, uint samples, uint burn_in, double da, double db, double ss, double sr, double lambda, uint nSimulation){
  MT19937 eng;
  Dict dict;
  const uint kSOS = dict.Convert("<s>");
  const uint kEOS = dict.Convert("</s>");

  // set total number of texts
  uint num_texts = train_files.size();

  //#### set up train corpus
  // initialize vocabulary
  set<uint> vocab_train;
  // number of words per text
  vector<uint> num_words_text(num_texts,0);
  // set up vocab_train, and multiple corpora c
  vector<vector<vector<uint>>> corpora_train(num_texts);
  uint d = 0;
  for (const auto& train_file : train_files)
    ReadFromFile( train_file, &dict, &corpora_train[d++], &vocab_train );
  
  //#### set up train corpus
  // initialize vocabulary
  set<uint> vocab_test;
  // set up vocab_train, and multiple corpora c
  vector<vector<vector<uint>>> corpora_test(num_texts);
  d = 0;
  for (const auto& test_file : test_files)
    ReadFromFile( test_file, &dict, &corpora_test[d++], &vocab_test );
  

  //####################    START    ##############################
  // set r variable: the lable for shared or not for each words
  vector<vector<vector<uint>>> r = corpora_train;
  for(uint j =0; j<num_texts; j++){
    for(uint s=0; s<corpora_train[j].size(); s++){
      for(uint i=0; i<corpora_train[j][s].size(); i++){
        r[j][s][i]=1;
        num_words_text[j] ++;
      }
    }
  }
  
  vector<vector<double>>llikelihoods;
  for(uint j=0; j<num_texts; j++ ){
    llikelihoods.push_back(vector<double>());
  }
  // initialize shared language model
  SPYPLM<kORDER> shpyp_lm(vocab_train.size(), num_texts, num_words_text, da, db, ss, sr, lambda);
                              
  cout<<"Starting Train SHPYP Model "<<endl;
  cerr<<"sample number: ";
  //--------- Starting Train Model
  vector<uint> ctx(kORDER - 1, kSOS);
  for (uint sample=0; sample < samples; ++sample) {
    for(uint j=0; j< num_texts; j++){
      for (uint s=0; s<corpora_train[j].size(); s++) {
          ctx.resize(kORDER - 1);
          for (uint i = 0; i <= corpora_train[j][s].size(); ++i) {
            uint w = (i < corpora_train[j][s].size() ? corpora_train[j][s][i] : kEOS);
            if (sample > 0) shpyp_lm.decrement(w, ctx, j, r[j][s][i], eng);
            shpyp_lm.increment(w, ctx, j, r[j][s][i], eng);
            ctx.push_back(w);   
          } 
      }
      if (sample % 10 == 9) {
          if (sample % 30u == 29) shpyp_lm.resample_hyperparameters(eng);
          shpyp_lm.resample_lambda_parameters(eng);
      }
      if (sample>burn_in)
        llikelihoods[j].push_back(shpyp_lm.log_likelihood(j));
    }
    if(sample % 10 == 9) cerr<<sample<<"-";
  }
  cout<<"Fin!"<<endl;
  
  for(uint j=0; j< llikelihoods.size(); j++){
    std::string Loglikelihood_txt = "output/Loglikelihood"+std::to_string(j)+"SHPYLM_nSim="+std::to_string(nSimulation)+"_lambda="+std::to_string(lambda)+"_da="+std::to_string(da)+"_db="+std::to_string(db)+"_ss="+std::to_string(ss)+"_sr="+std::to_string(sr)+".txt";  
    std::ofstream fileLoglikelihood(Loglikelihood_txt);
    if(fileLoglikelihood.is_open()){
        for(uint i=0; i< llikelihoods[j].size(); i++)
          fileLoglikelihood << llikelihoods[j][i] << "\n";
    }
    fileLoglikelihood.close();
  }
  

  cout<<"Starting Test SHPYP Model "<<endl;  
  //--------- Starting Test Model 
  double llh = 0; //sum of log predictive probabilities
  uint cnt = 0; // number of words in test set
  uint oovs = 0; // number of not found word in train set
  
  std::string estimations_txt = "output/estimationsSHPYLM_nSim="+std::to_string(nSimulation)+"_lambda="+std::to_string(lambda)+"_da="+std::to_string(da)+"_db="+std::to_string(db)+"_ss="+std::to_string(ss)+"_sr="+std::to_string(sr)+".txt";
  std::ofstream fileEsitimations(estimations_txt);
  
  if(fileEsitimations.is_open()){
    for(uint j=0; j< num_texts; j++){
      for (uint s=0; s<corpora_test[j].size(); s++) {
        ctx.resize(kORDER - 1);   
        for (uint i = 0; i <= corpora_test[j][s].size(); ++i) {
          uint w = (i < corpora_test[j][s].size() ? corpora_test[j][s][i] : kEOS);
          double lp = log(shpyp_lm.prob(w, ctx, j)) / log(2);
          if (i < corpora_test[j][s].size() && vocab_train.count(w) == 0) {
            fileEsitimations << "**OOV ";
            ++oovs;
            lp = 0;
          }
          fileEsitimations << "p(" << dict.Convert(w) << " |";
          for (unsigned l = ctx.size() + 1 - kORDER; l < ctx.size(); ++l)
            fileEsitimations << ' ' << dict.Convert(ctx[l]);
          fileEsitimations << ") = " << lp <<"\n";
          ctx.push_back(w);
          llh -= lp;
          cnt++;
        }
      }
    } 
  }
  fileEsitimations.close();

  cnt -= oovs;

  std::string performance_txt = "output/performanceSHPYLM_nSim="+std::to_string(nSimulation)+"_lambda="+std::to_string(lambda)+"_da="+std::to_string(da)+"_db="+std::to_string(db)+"_ss="+std::to_string(ss)+"_sr="+std::to_string(sr)+".txt";
  std::ofstream filePerformance(performance_txt);
  if(filePerformance.is_open()){
    for(uint j=0; j< num_texts; j++){
      filePerformance << "Train Corpora Size: " << corpora_train[j].size() << " sentences\t (number of words: " << num_words_text[j] << ")\n";
      filePerformance << "Test Corpora Size: " << corpora_test[j].size() << " sentences\n";
    }
    filePerformance << "Train Vocabulary Size: " << vocab_train.size() << " word types \n";
    filePerformance << "Test Vocabulary Size: " << vocab_test.size() << " word types \n";
    filePerformance << "Final Results: \n";
    filePerformance << "  Log_10 prob: " << (-llh * log(2) / log(10)) << "\n";
    filePerformance << "        Count: " << cnt << "\n";
    filePerformance << "         OOVs: " << oovs << "\n";
    filePerformance << "Cross-entropy: " << (llh / cnt) << "\n";
    filePerformance << "   Perplexity: " << pow(2, llh / cnt) << "\n";
  }
    filePerformance.close();

  cout<<"Final Results: "<<endl;
  //cout << "      Log llh: " <<  shpyp_lm.log_likelihood()<< endl;
  cout << "  Log_10 prob: " << (-llh * log(2) / log(10)) << endl;
  cout << "        Count: " << cnt << endl;
  cout << "         OOVs: " << oovs << endl;
  cout << "Cross-entropy: " << (llh / cnt) << endl;
  cout << "   Perplexity: " << pow(2, llh / cnt) << endl;
}



PYBIND11_MODULE(bnplm, m) {
    m.doc() = "Easy Life"; // optional module docstring

    m.def("hpypTrainTest", &hpypTrainTest,
          "train and test the hpyp model");
    m.def("shpypTrainTest", &shpypTrainTestCpp,
          "n-gram train test for SHARED LM ");
    
}
