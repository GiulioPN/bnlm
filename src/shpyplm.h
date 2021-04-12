#ifndef SHPYPLM_H_
#define SHPYPLM_H_

#include <vector>
#include <unordered_map>

#include "hpyplm.h"
#include "random.h"

namespace cpyp {

template <unsigned N> class SPYPLM {
 
private:
 PYPLM<N> shared_lm; //!< shared Pitman-Yor for LM 
 std::vector<PYPLM<N>> local_lm; //!< local Pitman-Yor process for LM
 std::vector<double> local_lambda; //!< lambda parameters 
 std::vector<unsigned int> N_words; //!< number of words per text
 std::vector<unsigned int> N_shared; //!< number of words in shared model.
 double a; //!< used for lambda resample
 double b; //!< used for lambda resample

public:
 /* Add costumer w to shared CRP or local j CRP
 *	\param vs vocabulary size
 *	\param n_txt number of corpus
 */
 explicit SPYPLM(unsigned vs, unsigned n_txt, std::vector<unsigned> num_of_words_per_text, double da = 1.0, double db = 1.0, double ss = 1.0, double sr = 1.0, double lambda=0.5, double a_in = 1.0, double b_in = 1.0) :
    shared_lm(vs, da, db, ss, sr), local_lm(n_txt,PYPLM<N> (vs, da, db, ss, sr)), local_lambda(n_txt,lambda), N_words(num_of_words_per_text), N_shared(n_txt,0), a(a_in), b(b_in)
	{}

 ~SPYPLM(){
 	local_lm.clear();
 	local_lambda.clear();
 }

 /* Add costumer w to shared CRP or local j CRP
 *	\param w add costumer
 *	\param j text index
 *	\param eng random engine
 * 	\return r a bool, called label obs: if 0 the observation w belongs to shared process, else (1) w belongs to the j process
 */
 template<typename Engine>
 void increment(unsigned w, const std::vector<unsigned>& context, unsigned int j, unsigned int& r,Engine& eng){
 	//sample
 	const unsigned int r_prec = r;
 	const double p_shared = (1 - local_lambda[j])*shared_lm.prob(w, context);
 	const double p_local = local_lambda[j]*local_lm[j].prob(w, context);
 	const double tot = p_shared + p_local;

 	r = sample_bernoulli(p_shared/tot, p_local/tot, eng);
 	if(r){
 		local_lm[j].increment(w, context, eng);
 		if(r != r_prec ) N_shared[j]--;
 	}
 	else {
 		shared_lm.increment(w, context, eng);
 		if(r != r_prec )N_shared[j] ++;
 	}
 }

 /* Remove costumer w to shared CRP or local j CRP
 *	\param w add costumer
 *	\param j text index
 *	\param eng random engine
 * 	\param r a bool, called label obs: if 0 the observation w belongs to shared process, else (1) w belongs to the j process
 */
 template<typename Engine>
 void decrement(unsigned w, const std::vector<unsigned>& context, unsigned j, bool r, Engine& eng){
 	if(r)
 		local_lm[j].decrement(w, context, eng);
 	else 
 		shared_lm.decrement(w, context, eng);
 }


 /* Parameters resemple of language models
  \param eng random engine
 */
 template<typename Engine>
 void resample_hyperparameters(Engine& eng){
	//shared_lm parameters
	shared_lm.resample_hyperparameters(eng);
	//resample local_lm parameters (for each local_lm)
	for(int j =0; j< local_lm.size(); j++)
		local_lm[j].resample_hyperparameters(eng);
 } 
 


 /* Parameters resemple of language models
  \param r a bool if 0 the observation w belongs to shared process, else (1) w belongs to the j process
  \param eng random engine
 */
template<typename Engine>
void resample_lambda_parameters(Engine& eng){
	//sample lambda
	double a_lambda=0, b_lambda=0;
	for(int j=0; j< local_lambda.size(); j++){
		a_lambda = a + N_shared[j];
		b_lambda = b + (N_words[j]-N_shared[j]);
		local_lambda[j] = sample_beta(a_lambda, b_lambda, eng);
	}
}


 /* helper function for logsumexp calculation
 *	\param nums a vector of numbers, where each number is x_i
 *	\return log(sum_i exp(x_i))
 */
 double logsumexp(std::vector<double>& nums) {
  double max_exp = nums[0], sum = 0.0;
  size_t i;

  for (i = 1 ; i < nums.size() ; i++)
    if (nums[i] > max_exp)
      max_exp = nums[i];

  for (i = 0; i < nums.size() ; i++)
    sum += exp(nums[i] - max_exp);

  return log(sum) + max_exp;
}


 /* Predict a new word
 *	\param w word index to be predicted
 *	\param j text index
 * 	\param r a bool if 0 the observation w belongs to shared process, else (1) w belongs to the j process
 *	\return pred_prob a double which store the prediction log probability
 */

 double prob(unsigned w, const std::vector<unsigned>& context, const unsigned j) const {
 	//Note: shared_lm.prob return a prob, not a log prob
 	double predictive_prob = (1-local_lambda[j])*shared_lm.prob(w, context) + local_lambda[j]*local_lm[j].prob(w, context) ;
 	return predictive_prob;
 }

 /* log_likelihood of the model j
 *	\param w word index to be predicted
 *	\param j text index
 * 	\param r a bool if 0 the observation w belongs to shared process, else (1) w belongs to the j process
 *	\return log likelihood of the SHPYP model
 */
 double log_likelihood(const unsigned j){
 	std::vector<double> log_llh;
 	log_llh.push_back( log( 1-local_lambda[j] ) + shared_lm.log_likelihood() ); // log(1-lambda) + shared_lm.log_llh --> x_1
 	log_llh.push_back( log( local_lambda[j] ) + local_lm[j].log_likelihood() ); // log(local_lambda[j]) + local_lm[j].log_llh --> x_2
 	return  logsumexp(log_llh); // log(exp(x_1) + exp(x_2)) = log ( (1-lambda)*shared.log_llh + local_lambda[j]*local_lm[j].log_llh )
 }


 void print(){
 	std::cout<<"-----------------------"<<std::endl;
 	std::cout<<"SHARED CRP:"<<std::endl;
 	shared_lm.print();
 	std::cout<<"LOCAL CRPs:"<<std::endl;
 	for (int i=0; i<local_lm.size(); i++){
 		std::cout<<"loc "<<i<<": "<<std::endl;
 		local_lm[i].print();
 	}
 }

};


}

#endif
