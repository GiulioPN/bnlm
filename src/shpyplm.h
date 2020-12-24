#ifndef SHPYPLM_H_
#define SHPYPLM_H_

#include <vector>
#include <unordered_map>

#include "hpyplm.h"

namespace cpyp {

template <unsigned N> struct SPYPLM {
 
 PYPLM<N> shared_lm;
 std::vector<PYPLM<N>> local_lm;
 std::vector<double> local_lambda;

public:

 explicit SPYPLM(unsigned vs, unsigned n_txt, double da = 1.0, double db = 1.0, double ss = 1.0, double sr = 1.0, double lambda=0.5) :
    shared_lm(vs, da, db, ss, sr), local_lm(n_txt,PYPLM<N> (vs, da, db, ss, sr)), local_lambda(n_txt,lambda)
	{}

 ~SPYPLM(){
 	local_lm.clear();
 	local_lambda.clear();
 }

 /* Add costumer w to shared CRP or local j CRP
 *	\param w add costumer
 *	\param j text index
 *	\param eng random engine
 * 	\return r a bool if 0 the observation w belongs to shared process, else (1) w belongs to the j process
 */
 template<typename Engine>
 bool increment(unsigned w, const std::vector<unsigned>& context, unsigned j, Engine& eng){
 	//sample
 	const double p_shared = 1 - local_lambda[j];
 	const double p_local = local_lambda[j];
 	bool r = sample_bernoulli(p_shared, p_local, eng);
 	if(r)
 		local_lm[j].increment(w, context, eng);
 	else 
 		shared_lm.increment(w, context, eng);

 	return r;
 }

 /* Remove costumer w to shared CRP or local j CRP
 *	\param w add costumer
 *	\param j text index
 *	\param eng random engine
 * 	\param r a bool if 0 the observation w belongs to shared process, else (1) w belongs to the j process
 */
 template<typename Engine>
 void decrement(unsigned w, const std::vector<unsigned>& context, unsigned j, bool r, Engine& eng){
 	if(r)
 		local_lm[j].decrement(w, context, eng);
 	else 
 		shared_lm.decrement(w, context, eng);
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
