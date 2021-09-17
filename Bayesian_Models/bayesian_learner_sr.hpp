
// Code generated by stanc v2.27.0
#include <stan/model/model_header.hpp>
namespace bayesian_learner_sr_model_namespace {

using stan::io::dump;
using stan::model::assign;
using stan::model::index_uni;
using stan::model::index_max;
using stan::model::index_min;
using stan::model::index_min_max;
using stan::model::index_multi;
using stan::model::index_omni;
using stan::model::model_base_crtp;
using stan::model::rvalue;
using namespace stan::math;


stan::math::profile_map profiles__;
static constexpr std::array<const char*, 33> locations_array__ = 
{" (found before start of program)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 8, column 2 to column 9)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 9, column 2 to column 38)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 10, column 2 to column 38)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 11, column 2 to column 12)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 15, column 2 to column 20)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 23, column 6 to column 35)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 29, column 8 to column 53)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 30, column 8 to column 53)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 31, column 8 to column 37)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 28, column 33 to line 32, column 7)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 28, column 12 to line 32, column 7)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 25, column 8 to column 53)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 26, column 8 to column 53)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 27, column 8 to column 37)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 24, column 27 to line 28, column 7)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 24, column 6 to line 32, column 7)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 22, column 8 to line 33, column 5)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 18, column 6 to column 31)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 19, column 6 to column 32)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 20, column 6 to column 32)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 17, column 14 to line 21, column 5)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 17, column 4 to line 33, column 5)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 16, column 15 to line 34, column 3)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 16, column 2 to line 34, column 3)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 2, column 2 to column 17)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 3, column 12 to column 13)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 3, column 2 to column 15)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 4, column 16 to column 17)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 4, column 2 to column 19)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 9, column 35 to column 36)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 10, column 35 to column 36)",
 " (in '/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan', line 11, column 9 to column 10)"};



class bayesian_learner_sr_model final : public model_base_crtp<bayesian_learner_sr_model> {

 private:
  int N;
  std::vector<int> react;
  std::vector<int> space_loc; 
  
 
 public:
  ~bayesian_learner_sr_model() { }
  
  inline std::string model_name() const final { return "bayesian_learner_sr_model"; }

  inline std::vector<std::string> model_compile_info() const noexcept {
    return std::vector<std::string>{"stanc_version = stanc3 v2.27.0", "stancflags = "};
  }
  
  
  bayesian_learner_sr_model(stan::io::var_context& context__,
                            unsigned int random_seed__ = 0,
                            std::ostream* pstream__ = nullptr) : model_base_crtp(0) {
    int current_statement__ = 0;
    using local_scalar_t__ = double ;
    boost::ecuyer1988 base_rng__ = 
        stan::services::util::create_rng(random_seed__, 0);
    (void) base_rng__;  // suppress unused var warning
    static constexpr const char* function__ = "bayesian_learner_sr_model_namespace::bayesian_learner_sr_model";
    (void) function__;  // suppress unused var warning
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    (void) DUMMY_VAR__;  // suppress unused var warning
    try {
      int pos__;
      pos__ = std::numeric_limits<int>::min();
      
      pos__ = 1;
      current_statement__ = 25;
      context__.validate_dims("data initialization","N","int",
           std::vector<size_t>{});
      N = std::numeric_limits<int>::min();
      
      current_statement__ = 25;
      N = context__.vals_i("N")[(1 - 1)];
      current_statement__ = 25;
      check_greater_or_equal(function__, "N", N, 1);
      current_statement__ = 26;
      validate_non_negative_index("react", "N", N);
      current_statement__ = 27;
      context__.validate_dims("data initialization","react","int",
           std::vector<size_t>{static_cast<size_t>(N)});
      react = std::vector<int>(N, std::numeric_limits<int>::min());
      
      current_statement__ = 27;
      react = context__.vals_i("react");
      current_statement__ = 28;
      validate_non_negative_index("space_loc", "N", N);
      current_statement__ = 29;
      context__.validate_dims("data initialization","space_loc","int",
           std::vector<size_t>{static_cast<size_t>(N)});
      space_loc = std::vector<int>(N, std::numeric_limits<int>::min());
      
      current_statement__ = 29;
      space_loc = context__.vals_i("space_loc");
      current_statement__ = 30;
      validate_non_negative_index("r_l", "N", N);
      current_statement__ = 31;
      validate_non_negative_index("r_r", "N", N);
      current_statement__ = 32;
      validate_non_negative_index("v", "N", N);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
    }
    num_params_r__ = 1 + N + N + N;
    
  }
  
  template <bool propto__, bool jacobian__ , typename VecR, typename VecI, 
  stan::require_vector_like_t<VecR>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr> 
  inline stan::scalar_type_t<VecR> log_prob_impl(VecR& params_r__,
                                                 VecI& params_i__,
                                                 std::ostream* pstream__ = nullptr) const {
    using T__ = stan::scalar_type_t<VecR>;
    using local_scalar_t__ = T__;
    T__ lp__(0.0);
    stan::math::accumulator<T__> lp_accum__;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    (void) DUMMY_VAR__;  // suppress unused var warning
    static constexpr const char* function__ = "bayesian_learner_sr_model_namespace::log_prob";
    (void) function__;  // suppress unused var warning
    
    try {
      local_scalar_t__ k;
      k = DUMMY_VAR__;
      
      current_statement__ = 1;
      k = in__.template read<local_scalar_t__>();
      std::vector<local_scalar_t__> r_l;
      r_l = std::vector<local_scalar_t__>(N, DUMMY_VAR__);
      
      current_statement__ = 2;
      r_l = in__.template read_constrain_lub<std::vector<local_scalar_t__>, jacobian__>(
              0.01, 0.99, lp__, N);
      std::vector<local_scalar_t__> r_r;
      r_r = std::vector<local_scalar_t__>(N, DUMMY_VAR__);
      
      current_statement__ = 3;
      r_r = in__.template read_constrain_lub<std::vector<local_scalar_t__>, jacobian__>(
              0.01, 0.99, lp__, N);
      std::vector<local_scalar_t__> v;
      v = std::vector<local_scalar_t__>(N, DUMMY_VAR__);
      
      current_statement__ = 4;
      v = in__.template read<std::vector<local_scalar_t__>>(N);
      {
        current_statement__ = 5;
        lp_accum__.add(uniform_lpdf<propto__>(k, -10, 10));
        current_statement__ = 24;
        for (int t = 1; t <= N; ++t) {
          current_statement__ = 22;
          if (logical_eq(t, 1)) {
            current_statement__ = 18;
            lp_accum__.add(
              uniform_lpdf<propto__>(rvalue(v, "v", index_uni(t)), -100, 100));
            current_statement__ = 19;
            lp_accum__.add(
              normal_lpdf<propto__>(rvalue(r_l, "r_l", index_uni(t)), 0.5,
                0.45));
            current_statement__ = 20;
            lp_accum__.add(
              normal_lpdf<propto__>(rvalue(r_r, "r_r", index_uni(t)), 0.5,
                0.45));
          } else {
            current_statement__ = 6;
            lp_accum__.add(
              normal_lpdf<propto__>(rvalue(v, "v", index_uni(t)),
                rvalue(v, "v", index_uni((t - 1))), stan::math::exp(k)));
            current_statement__ = 16;
            if (logical_eq(rvalue(space_loc, "space_loc", index_uni(t)), 0)) {
              current_statement__ = 12;
              lp_accum__.add(
                beta_proportion_lpdf<propto__>(
                  rvalue(r_l, "r_l", index_uni(t)),
                  rvalue(r_l, "r_l", index_uni((t - 1))),
                  stan::math::exp(rvalue(v, "v", index_uni(t)))));
              current_statement__ = 13;
              lp_accum__.add(
                beta_proportion_lpdf<propto__>(
                  rvalue(r_r, "r_r", index_uni(t)),
                  rvalue(r_r, "r_r", index_uni((t - 1))),
                  stan::math::exp(rvalue(v, "v", index_uni(t)))));
              current_statement__ = 14;
              lp_accum__.add(
                bernoulli_lpmf<propto__>(
                  rvalue(react, "react", index_uni(t)),
                  rvalue(r_l, "r_l", index_uni(t))));
            } else {
              current_statement__ = 11;
              if (logical_eq(rvalue(space_loc, "space_loc", index_uni(t)), 1)) {
                current_statement__ = 7;
                lp_accum__.add(
                  beta_proportion_lpdf<propto__>(
                    rvalue(r_r, "r_r", index_uni(t)),
                    rvalue(r_r, "r_r", index_uni((t - 1))),
                    stan::math::exp(rvalue(v, "v", index_uni(t)))));
                current_statement__ = 8;
                lp_accum__.add(
                  beta_proportion_lpdf<propto__>(
                    rvalue(r_l, "r_l", index_uni(t)),
                    rvalue(r_l, "r_l", index_uni((t - 1))),
                    stan::math::exp(rvalue(v, "v", index_uni(t)))));
                current_statement__ = 9;
                lp_accum__.add(
                  bernoulli_lpmf<propto__>(
                    rvalue(react, "react", index_uni(t)),
                    rvalue(r_r, "r_r", index_uni(t))));
              } 
            }
          }
        }
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
    }
    lp_accum__.add(lp__);
    return lp_accum__.sum();
    } // log_prob_impl() 
    
  template <typename RNG, typename VecR, typename VecI, typename VecVar, 
  stan::require_vector_like_vt<std::is_floating_point, VecR>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr, 
  stan::require_std_vector_vt<std::is_floating_point, VecVar>* = nullptr> 
  inline void write_array_impl(RNG& base_rng__, VecR& params_r__,
                               VecI& params_i__, VecVar& vars__,
                               const bool emit_transformed_parameters__ = true,
                               const bool emit_generated_quantities__ = true,
                               std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    vars__.resize(0);
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    static constexpr bool propto__ = true;
    (void) propto__;
    double lp__ = 0.0;
    (void) lp__;  // dummy to suppress unused var warning
    int current_statement__ = 0; 
    stan::math::accumulator<double> lp_accum__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    constexpr bool jacobian__ = false;
    (void) DUMMY_VAR__;  // suppress unused var warning
    static constexpr const char* function__ = "bayesian_learner_sr_model_namespace::write_array";
    (void) function__;  // suppress unused var warning
    
    try {
      double k;
      k = std::numeric_limits<double>::quiet_NaN();
      
      current_statement__ = 1;
      k = in__.template read<local_scalar_t__>();
      std::vector<double> r_l;
      r_l = std::vector<double>(N, std::numeric_limits<double>::quiet_NaN());
      
      current_statement__ = 2;
      r_l = in__.template read_constrain_lub<std::vector<local_scalar_t__>, jacobian__>(
              0.01, 0.99, lp__, N);
      std::vector<double> r_r;
      r_r = std::vector<double>(N, std::numeric_limits<double>::quiet_NaN());
      
      current_statement__ = 3;
      r_r = in__.template read_constrain_lub<std::vector<local_scalar_t__>, jacobian__>(
              0.01, 0.99, lp__, N);
      std::vector<double> v;
      v = std::vector<double>(N, std::numeric_limits<double>::quiet_NaN());
      
      current_statement__ = 4;
      v = in__.template read<std::vector<local_scalar_t__>>(N);
      vars__.emplace_back(k);
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        vars__.emplace_back(r_l[(sym1__ - 1)]);
      }
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        vars__.emplace_back(r_r[(sym1__ - 1)]);
      }
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        vars__.emplace_back(v[(sym1__ - 1)]);
      }
      if (logical_negation((primitive_value(emit_transformed_parameters__) ||
            primitive_value(emit_generated_quantities__)))) {
        return ;
      } 
      if (logical_negation(emit_generated_quantities__)) {
        return ;
      } 
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
    }
    } // write_array_impl() 
    
  template <typename VecVar, typename VecI, 
  stan::require_std_vector_t<VecVar>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr> 
  inline void transform_inits_impl(const stan::io::var_context& context__,
                                   VecI& params_i__, VecVar& vars__,
                                   std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    vars__.clear();
    vars__.reserve(num_params_r__);
    int current_statement__ = 0; 
    
    try {
      int pos__;
      pos__ = std::numeric_limits<int>::min();
      
      pos__ = 1;
      double k;
      k = std::numeric_limits<double>::quiet_NaN();
      
      current_statement__ = 1;
      k = context__.vals_r("k")[(1 - 1)];
      std::vector<double> r_l;
      r_l = std::vector<double>(N, std::numeric_limits<double>::quiet_NaN());
      
      current_statement__ = 2;
      r_l = context__.vals_r("r_l");
      std::vector<double> r_l_free__;
      r_l_free__ = std::vector<double>(N, std::numeric_limits<double>::quiet_NaN());
      
      
      current_statement__ = 2;
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        current_statement__ = 2;
        assign(r_l_free__,
          stan::math::lub_free(r_l[(sym1__ - 1)], 0.01, 0.99),
          "assigning variable r_l_free__", index_uni(sym1__));
      }
      std::vector<double> r_r;
      r_r = std::vector<double>(N, std::numeric_limits<double>::quiet_NaN());
      
      current_statement__ = 3;
      r_r = context__.vals_r("r_r");
      std::vector<double> r_r_free__;
      r_r_free__ = std::vector<double>(N, std::numeric_limits<double>::quiet_NaN());
      
      
      current_statement__ = 3;
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        current_statement__ = 3;
        assign(r_r_free__,
          stan::math::lub_free(r_r[(sym1__ - 1)], 0.01, 0.99),
          "assigning variable r_r_free__", index_uni(sym1__));
      }
      std::vector<double> v;
      v = std::vector<double>(N, std::numeric_limits<double>::quiet_NaN());
      
      current_statement__ = 4;
      v = context__.vals_r("v");
      vars__.emplace_back(k);
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        vars__.emplace_back(r_l_free__[(sym1__ - 1)]);
      }
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        vars__.emplace_back(r_r_free__[(sym1__ - 1)]);
      }
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        vars__.emplace_back(v[(sym1__ - 1)]);
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
    }
    } // transform_inits_impl() 
    
  inline void get_param_names(std::vector<std::string>& names__) const {
    
    names__ = std::vector<std::string>{"k", "r_l", "r_r", "v"};
    
    } // get_param_names() 
    
  inline void get_dims(std::vector<std::vector<size_t>>& dimss__) const {
    
    dimss__ = std::vector<std::vector<size_t>>{std::vector<size_t>{},
      std::vector<size_t>{static_cast<size_t>(N)},
      std::vector<size_t>{static_cast<size_t>(N)},
      std::vector<size_t>{static_cast<size_t>(N)}};
    
    } // get_dims() 
    
  inline void constrained_param_names(
                                      std::vector<std::string>& param_names__,
                                      bool emit_transformed_parameters__ = true,
                                      bool emit_generated_quantities__ = true) const
    final {
    
    param_names__.emplace_back(std::string() + "k");
    for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "r_l" + '.' + std::to_string(sym1__));
      } 
    }
    for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "r_r" + '.' + std::to_string(sym1__));
      } 
    }
    for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "v" + '.' + std::to_string(sym1__));
      } 
    }
    if (emit_transformed_parameters__) {
      
    }
    
    if (emit_generated_quantities__) {
      
    }
    
    } // constrained_param_names() 
    
  inline void unconstrained_param_names(
                                        std::vector<std::string>& param_names__,
                                        bool emit_transformed_parameters__ = true,
                                        bool emit_generated_quantities__ = true) const
    final {
    
    param_names__.emplace_back(std::string() + "k");
    for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "r_l" + '.' + std::to_string(sym1__));
      } 
    }
    for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "r_r" + '.' + std::to_string(sym1__));
      } 
    }
    for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "v" + '.' + std::to_string(sym1__));
      } 
    }
    if (emit_transformed_parameters__) {
      
    }
    
    if (emit_generated_quantities__) {
      
    }
    
    } // unconstrained_param_names() 
    
  inline std::string get_constrained_sizedtypes() const {
    
    return std::string("[{\"name\":\"k\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"r_l\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(N) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"parameters\"},{\"name\":\"r_r\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(N) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"parameters\"},{\"name\":\"v\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(N) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"parameters\"}]");
    
    } // get_constrained_sizedtypes() 
    
  inline std::string get_unconstrained_sizedtypes() const {
    
    return std::string("[{\"name\":\"k\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"r_l\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(N) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"parameters\"},{\"name\":\"r_r\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(N) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"parameters\"},{\"name\":\"v\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(N) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"parameters\"}]");
    
    } // get_unconstrained_sizedtypes() 
    
  
    // Begin method overload boilerplate
    template <typename RNG>
    inline void write_array(RNG& base_rng,
                            Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,
                            Eigen::Matrix<double,Eigen::Dynamic,1>& vars,
                            const bool emit_transformed_parameters = true,
                            const bool emit_generated_quantities = true,
                            std::ostream* pstream = nullptr) const {
      std::vector<double> vars_vec;
      vars_vec.reserve(vars.size());
      std::vector<int> params_i;
      write_array_impl(base_rng, params_r, params_i, vars_vec,
          emit_transformed_parameters, emit_generated_quantities, pstream);
      vars = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,1>>(
        vars_vec.data(), vars_vec.size());
    }

    template <typename RNG>
    inline void write_array(RNG& base_rng, std::vector<double>& params_r,
                            std::vector<int>& params_i,
                            std::vector<double>& vars,
                            bool emit_transformed_parameters = true,
                            bool emit_generated_quantities = true,
                            std::ostream* pstream = nullptr) const {
      write_array_impl(base_rng, params_r, params_i, vars,
       emit_transformed_parameters, emit_generated_quantities, pstream);
    }

    template <bool propto__, bool jacobian__, typename T_>
    inline T_ log_prob(Eigen::Matrix<T_,Eigen::Dynamic,1>& params_r,
                       std::ostream* pstream = nullptr) const {
      Eigen::Matrix<int, -1, 1> params_i;
      return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
    }

    template <bool propto__, bool jacobian__, typename T__>
    inline T__ log_prob(std::vector<T__>& params_r,
                        std::vector<int>& params_i,
                        std::ostream* pstream = nullptr) const {
      return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
    }


    inline void transform_inits(const stan::io::var_context& context,
                         Eigen::Matrix<double, Eigen::Dynamic, 1>& params_r,
                         std::ostream* pstream = nullptr) const final {
      std::vector<double> params_r_vec;
      params_r_vec.reserve(params_r.size());
      std::vector<int> params_i;
      transform_inits_impl(context, params_i, params_r_vec, pstream);
      params_r = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,1>>(
        params_r_vec.data(), params_r_vec.size());
    }
    inline void transform_inits(const stan::io::var_context& context,
                                std::vector<int>& params_i,
                                std::vector<double>& vars,
                                std::ostream* pstream = nullptr) const final {
      transform_inits_impl(context, params_i, vars, pstream);
    }

};
}

using stan_model = bayesian_learner_sr_model_namespace::bayesian_learner_sr_model;

#ifndef USING_R

// Boilerplate
stan::model::model_base& new_model(
        stan::io::var_context& data_context,
        unsigned int seed,
        std::ostream* msg_stream) {
  stan_model* m = new stan_model(data_context, seed, msg_stream);
  return *m;
}

stan::math::profile_map& get_stan_profile_data() {
  return bayesian_learner_sr_model_namespace::profiles__;
}

#endif


