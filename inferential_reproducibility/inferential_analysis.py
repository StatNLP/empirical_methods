
#The authors of pymer4 recommend to add the following lines when pymer is run inside a jupyter notebook.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy  as np
import pandas as pd
from scipy.stats import chi2
from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_string_dtype
from pymer4.models import Lmer
from plotnine import *

class InferentialAnalysis:
    '''
    '''
    
    
    class SystemComparison:
        
        def __init__(self):
            self.glrt = []
            self.means = pd.DataFrame()
            self.contrasts = pd.DataFrame()
            
    
    class ConditionalSystemComparison:
        
        def __init__(self):
            self.glrt = []
            self.means = pd.DataFrame()
            self.slopes = pd.DataFrame()
            self.contrasts = pd.DataFrame()
            self.interaction_plot = ggplot()
            self.data_property = ""
    
    
    class Reliability:
        
        def __init__(self):
            self.algorithm = ""
            self.icc = pd.DataFrame()
            
    
    class HyperParameterAssessment:
        
        def __init__(self):
            self.algorithm = ""
            self.glrt = []
            self.means = pd.DataFrame()
            self.contrasts = pd.DataFrame()
            
            
    class ConditionalHyperParameterAssessment:
        
        def __init__(self):
            self.algorithm = ""
            self.glrt = []
            self.means = pd.DataFrame()
            self.slopes = pd.DataFrame()
            self.contrasts = pd.DataFrame()
            self.interaction_plot = ggplot()
            self.data_property = ""
    

    
    def __init__(self, evaluation_data, eval_metric_col, system_col, input_identifier_col, distribution = "gaussian"):
        self.data   = evaluation_data
        self.metric = eval_metric_col
        self.system = system_col 
        self.input_id = input_identifier_col
        self.distribution = distribution
        self.SystemComparison = self.SystemComparison()
        self.ConditionalSystemComparison = self.ConditionalSystemComparison()
        self.Reliability = self.Reliability()
        self.HyperParameterAssessment = self.HyperParameterAssessment()
        self.ConditionalHyperParameterAssessment = self.ConditionalHyperParameterAssessment()
    
        #check input consistency
        if distribution == "gaussian":
            if not is_numeric_dtype(self.data[self.metric]):
                raise ValueError("Data type of provided evaluation metric" + self.metric + " is not numerical!")
        elif distribution == "binomial":
            if not len(self.data[self.metric].unique()) == 2:
                raise ValueError("Provided evaluation metric data column " + self.metric + " has more or less than 2 values. You can't run a binomial model.")
        else:
            raise ValueError("You have choosen an currently unsupported distribution.")
            
        #make sure that self.system and input_identifer_var are proper categoricals
        if not is_categorical_dtype(self.data[self.input_id]):
            self.data = self.data.astype({self.input_id : 'string'})
            self.data = self.data.astype({self.input_id : 'categorical'})
 
        if not is_categorical_dtype(self.data[self.system]):
            print("WARNING: " + self.system + " is not categorical! Datatype will be converted.")
            self.data = self.data.astype({self.system : 'string'})
            self.data = self.data.astype({self.system : 'categorical'})
    
        #make sure that self.system categories are all strings. This is important for Lmer.
        self.data[self.system] = self.data[self.system].cat.rename_categories(lambda x : str(x))
        
       
    
    def GLRT(self, mod1, mod2):
    
        chi_square = 2 * abs(mod1.logLike - mod2.logLike)
        delta_params = abs(len(mod1.coefs) - len(mod2.coefs)) 
    
        return {"chi_square" : chi_square, "df": delta_params, "p" : 1 - chi2.cdf(chi_square, df=delta_params)}
          
        
        
    def system_comparison(self, alpha = .05, verbose = True, row_filter = ""):

        #check input consistency                   
        if alpha <= 0 or alpha >=1:
            raise ValueError("alpha must be set to a value in (0,1)!")
        
        #minimize data to speed processing and remove rows with missing values     
        if row_filter:
            model_data = self.data.query(row_filter).copy()
            model_data = model_data[[self.system, self.metric, self.input_id]].dropna()
        else:
            model_data = self.data[[self.system, self.metric, self.input_id]].dropna()
            
        model_data[self.system] = model_data[self.system].cat.remove_unused_categories()
                               
        #instantiate and fit models
        formula_H1 = self.metric + " ~ " + self.system + " + ( 1 | " + self.input_id + " )"
        formula_H0 = self.metric + " ~ " +                 "( 1 | " + self.input_id + " )"
        
        model_H1 = Lmer(formula = formula_H1, data = model_data, family = self.distribution)
        model_H0 = Lmer(formula = formula_H0, data = model_data, family = self.distribution)
    
        model_factors = {} 
        model_factors[self.system] = [s for s in model_data[self.system].cat.categories]
        
        print("Fitting H0-model.")
        model_H0.fit(REML = False, summarize = False)
        print("Fitting H1-model.")
        model_H1.fit(factors = model_factors, REML = False, summarize = False)
 
        
        #compare models via GLRT
        self.SystemComparison.glrt = self.GLRT(model_H0, model_H1)
    
        #create means and contasts 
        postHoc_result = [r for r in model_H1.post_hoc(marginal_vars = self.system)]
        
        self.SystemComparison.means = postHoc_result[0].drop(columns = "DF").rename(columns = {"2.5_ci" : "95CI_lo", "97.5_ci" : "95CI_up"})
        self.SystemComparison.contrasts = postHoc_result[1].drop(columns = ["DF", "T-stat", "Z-stat", "Sig"], errors = "ignore").rename(columns = {"2.5_ci" : "95CI_lo", "97.5_ci" : "95CI_up"})

        #add effect size (a Hodge's g derivate) to contrasts
        sigma_residuals = model_H1.ranef_var.loc["Residual", "Std"]
        self.SystemComparison.contrasts = self.SystemComparison.contrasts.assign(Effect_size_g = lambda df : df.Estimate / sigma_residuals)
        
        
        if self.SystemComparison.glrt["p"] <= alpha and verbose:
            print("GLRT p-value <= alpha: Null hypothesis can be rejected! At least two systems are different. See contrasts for pairwise comparisons.")
        elif verbose:
            print("GLRT p-value > alpha: Null hypothesis can not be rejected! No statistical signifcant difference(s) between systems.")
        
        return        
    
    
    
    def conditional_system_comparison(self, data_prop_col, alpha = .05, scale_data_prop = False, verbose = True, row_filter = ""):
        '''
        '''

        #check input consistency      
        if alpha <= 0 or alpha >=1:
            raise ValueError("alpha must be set to a value in (0,1)!")
    
        #check data type of data_prop_col
        if is_categorical_dtype(self.data[data_prop_col]):
            self.data[data_prop_col] = self.data[data_prop_col].cat.rename_categories(lambda x : str(x))
            print("Data property is a categorical variable. Applying cell means model and reporting means.")
            reported_estimates = "means"
        elif is_string_dtype(self.data[data_prop_col]):   
            self.data = self.data.astype({data_prop_col : 'categorical'})
            print("Data property is a categorical variable. Applying cell means model and reporting means.")
            reported_estimates = "means"
        elif is_numeric_dtype(self.data[data_prop_col]):
            print("Data property is a numeric variable. Applying indivdual trends model and reporting slopes.")
            reported_estimates = "slopes"
        else:
            raise ValueError("Data property column " + data_prop_col + " data type is neither numeric nor categorical/string.")
        
    
        #minimize data to speed processing and removing rows with missing values       
        if row_filter:
            model_data = self.data.query(row_filter).copy()
            model_data = model_data[[self.system, self.metric, data_prop_col, self.input_id]].dropna()
        else:
            model_data = self.data[[self.system, self.metric, data_prop_col, self.input_id]].dropna()
            
        model_data[self.system] = model_data[self.system].cat.remove_unused_categories()
        if is_categorical_dtype(model_data[data_prop_col]):
            model_data[data_prop_col] = model_data[data_prop_col].cat.remove_unused_categories()
        
        #instantiate and fit models
        if scale_data_prop:
            data_prop_col_m = "scale(" + data_prop_col + ")"
        else:
            data_prop_col_m = data_prop_col
        
        formula_H1 = self.metric + " ~ " + self.system + " + " + data_prop_col_m + " + " + self.system + ":" + data_prop_col_m + " + ( 1 | " + self.input_id + " )"
        formula_H0 = self.metric + " ~ " + self.system + " + " + data_prop_col_m +                                               " + ( 1 | " + self.input_id + " )"
    
        model_H1 = Lmer(formula = formula_H1, data = model_data, family = self.distribution)
        model_H0 = Lmer(formula = formula_H0, data = model_data, family = self.distribution)
    
        model_factors = {} 
        model_factors[self.system] = [s for s in self.data[self.system].cat.categories]
    
        if is_categorical_dtype(self.data[data_prop_col]):
            model_data[data_prop_col] = model_data[data_prop_col].cat.remove_unused_categories()
            model_factors[data_prop_col] = [p for p in model_data[data_prop_col].cat.categories]
         
        print("Fitting H0-model.")
        model_H0.fit(factors = model_factors, REML = False, summarize = False)
        print("Fitting H1-model.")
        model_H1.fit(factors = model_factors, REML = False, summarize = False)
        
        
        #compare models and calculate postHoc
        self.ConditionalSystemComparison.glrt = self.GLRT(model_H0, model_H1)
        
        #FOR CATEGORICAL data property!!!!
        if is_categorical_dtype(model_data[data_prop_col]):
            postHoc_result = [r for r in model_H1.post_hoc(marginal_vars = self.system, grouping_vars = data_prop_col)]
        
        if is_numeric_dtype(model_data[data_prop_col]):
            postHoc_result = [r for r in model_H1.post_hoc(marginal_vars = data_prop_col, grouping_vars = self.system)]
        
        
        #simplify postHoc result
        if reported_estimates == "means":
            self.ConditionalSystemComparison.means = postHoc_result[0].drop(columns = "DF").rename(columns = {"2.5_ci" : "95CI_lo", "97.5_ci" : "95CI_up"})
        else:
            self.ConditionalSystemComparison.slopes = postHoc_result[0].drop(columns = "DF").rename(columns = {"2.5_ci" : "95CI_lo", "97.5_ci" : "95CI_up"})
            
        self.ConditionalSystemComparison.contrasts = postHoc_result[1].drop(columns = ["DF", "T-stat", "Z-stat", "Sig"], errors = "ignore").rename(columns = {"2.5_ci" : "95CI_lo", "97.5_ci" : "95CI_up"})
           
        #add effect size (a Hedge's g derivate) to mean model contrasts
        if reported_estimates == "means":
            sigma_residuals = model_H1.ranef_var.loc["Residual", "Std"]
            self.ConditionalSystemComparison.contrasts = self.ConditionalSystemComparison.contrasts.assign(Effect_size_g = lambda df : df.Estimate / sigma_residuals)
        
        
        if is_categorical_dtype(model_data[data_prop_col]):
            self.ConditionalSystemComparison.interaction_plot = (
                ggplot(postHoc_result[0]) 
                + theme_bw() 
                + theme(panel_grid = element_blank(),
                        legend_position = "top",
                        legend_title = element_blank())
                + xlab(data_prop_col) 
                + ylab("Estimated Expectation of Evaluation Metric")  
                + geom_pointrange(aes(x      = data_prop_col, 
                                      y      = "Estimate",
                                      ymin   = "Estimate - SE",
                                      ymax   = "Estimate + SE",
                                      colour = self.system),
                                  alpha = .7) 
                + geom_line(aes(x      = data_prop_col,
                                y      = "Estimate",
                                group  = self.system,
                                colour = self.system), 
                            alpha = .3)
            )
        
        if is_numeric_dtype(model_data[data_prop_col]):
            self.ConditionalSystemComparison.interaction_plot = (
                ggplot(data = model_data)
                + theme_bw()
                + theme(panel_grid = element_blank(),
                        legend_position = "top",
                        legend_title = element_blank())
                + xlab(data_prop_col)
                + ylab("Estimated Expected Evaluation Metric")
                + geom_smooth(aes(x = data_prop_col, y = self.metric, group = self.system, linetype = self.system),
                              method = "lm", 
                              colour = "black", 
                              se     = False)
            )   
    
        
        if self.ConditionalSystemComparison.glrt["p"] <= alpha and verbose :
            print("GLRT p-value <= alpha: Null hypothesis can be rejected! At least two systems depend differently to the data property. See contrasts for pairwise comparisons.")
        elif verbose :
            print("GLRT p-value > alpha: Null hypothesis can not be rejected! No statistical signifcant difference(s) between systems.")

        self.ConditionalSystemComparison.data_property = data_prop_col
    
        return    
    
    
    def icc(self, algorithm_id, facet_cols, row_filter = ""):
        '''
        '''
        
        #check if variables for random interceps have the correct data type
        if isinstance(facet_cols, str):
            facet_cols = [facet_cols]
        
        var_components = [self.input_id] + facet_cols
    
        for c in var_components:
            if not is_categorical_dtype(self.data[c]):
                print("WARNING: " + c + " is not categorical! Datatype will be converted.")
                self.data = self.data.astype({c : 'string'})
                self.data = self.data.astype({c : 'category'})
        
        
        self.Reliability.algorithm = algorithm_id
        
        #minimize data to speed processing and removing rows with missing values
        model_data = self.data.query(f"{self.system} == '{algorithm_id}'").copy()
        
        if row_filter:
            model_data = model_data.query(row_filter).copy()
       
        model_data = model_data[[self.metric] + var_components].dropna()
        
        #instantiate and fit models
        formula_var_decomposition_model = self.metric + " ~ " + " + ".join([f"( 1 | {c})" for c in var_components])
        
        var_decomposition_model = Lmer(formula_var_decomposition_model, data = model_data, family = self.distribution)
    
        print("Calculating variance decomposition.")
        var_decomposition_model.fit(summarize = False, control = "calc.derivs = FALSE")
        
        
        #calculate icc based on the variance decomposition
        self.Reliability.icc = var_decomposition_model.ranef_var.drop(columns = ["Name","Std"])
        self.Reliability.icc["ICC"] = self.Reliability.icc["Var"] * 100 / sum(self.Reliability.icc["Var"])
        
        return
    
    
    def hyperparameter_assessment(self, algorithm_id, hyperparameter_col, alpha = .05, verbose = True, row_filter = ""): 
        '''
        '''

        #check input consistency                  
        if alpha <= 0 or alpha >=1:
            raise ValueError("alpha must be set to a value in (0,1)!")

        #make sure that hyperparameter_col and input_identifer_var are proper categoricals
        if not is_categorical_dtype(self.data[hyperparameter_col]):
            print("WARNING: " + hyperparameter_col + " is not categorical! Datatype will be converted.")
            self.data = self.data.astype({hyperparameter_col : 'string'})
            self.data = self.data.astype({hyperparameter_col : 'category'})
        
        #make sure that hyperparameter_col categories are all strings. This is important for Lmer.
        self.data[hyperparameter_col] = self.data[hyperparameter_col].cat.rename_categories(lambda x : str(x))
     
        #minimize data to speed processing and remove rows with missing values  
        model_data = self.data.query(f"{self.system} == '{algorithm_id}'").copy()
        
        if row_filter:
            model_data = model_data.query(row_filter).copy()
        
        model_data = model_data[[hyperparameter_col, self.metric, self.input_id]].dropna()
        model_data[hyperparameter_col].cat = model_data[hyperparameter_col].cat.remove_unused_categories()

        
        #instantiate and fit models
        formula_H1 = self.metric + " ~ " + hyperparameter_col + " + ( 1 | " + self.input_id + " )"
        formula_H0 = self.metric + " ~ " +                         "( 1 | " + self.input_id + " )"
    
        model_H1 = Lmer(formula = formula_H1, data = model_data, family = self.distribution)
        model_H0 = Lmer(formula = formula_H0, data = model_data, family = self.distribution)
        
        model_factors = {} 
        model_factors[hyperparameter_col] = [s for s in model_data[hyperparameter_col].cat.categories]

        print("Fitting H0-model.")
        model_H0.fit(REML = False, summarize = False)
        print("Fitting H1-model.")
        model_H1.fit(factors = model_factors, REML = False, summarize = False)
 

        #compare models and calculate postHoc stats
        self.HyperParameterAssessment.glrt = self.GLRT(model_H0, model_H1)
        postHoc_result = [r for r in model_H1.post_hoc(marginal_vars = hyperparameter_col)]
    
        #simplify postHoc result
        self.HyperParameterAssessment.means = postHoc_result[0].drop(columns = "DF").rename(columns = {"2.5_ci" : "95CI_lo", "97.5_ci" : "95CI_up"})
        self.HyperParameterAssessment.contrasts = postHoc_result[1].drop(columns = ["DF", "T-stat", "Z-stat", "Sig"], errors = "ignore").rename(columns = {"2.5_ci" : "95CI_lo", "97.5_ci" : "95CI_up"})
        
        #add effect size (a Hedge's g derivate) to mean model contrasts
        sigma_residuals = model_H1.ranef_var.loc["Residual", "Std"]
        self.HyperParameterAssessment.contrasts = self.HyperParameterAssessment.contrasts.assign(Effect_size_g = lambda df : df.Estimate / sigma_residuals)
             
            
        if self.HyperParameterAssessment.glrt["p"] <= alpha and verbose:
            print("GLRT p-value <= alpha: Null hypothesis can be rejected! At least two systems are different. See contrasts for pairwise comparisons.")
        elif verbose:
            print("GLRT p-value > alpha: Null hypothesis can not be rejected! No statistical signifcant difference(s) between systems.")
            
        self.HyperParameterAssessment.algorithm = algorithm_id
        
        return
    
    
    
    def conditional_hyperparameter_assessment(self, algorithm_id, hyperparameter_col, data_prop_col, alpha = .05, scale_data_prop = False, verbose = True, row_filter = ""): 
        '''
        '''
        
        #check input consistency     
        if alpha <= 0 or alpha >=1:
            raise ValueError("alpha must be set to a value in (0,1)!")

        #make sure that hyperparameter_col is a proper categorical
        if not is_categorical_dtype(self.data[hyperparameter_col]):
            print("WARNING: " + hyperparameter_col + " is not categorical! Datatype will be converted.")
            self.data = self.data.astype({hyperparameter_col : 'string'})
            self.data = self.data.astype({hyperparameter_col : 'category'})
    
        #make sure that hyperparameter_col categories are all strings. This is important for Lmer.
        self.data[hyperparameter_col] = self.data[hyperparameter_col].cat.rename_categories(lambda x : str(x))
    
        #check data type of data_prop_col
        if is_categorical_dtype(self.data[data_prop_col]):
            self.data[data_prop_col] = self.data[data_prop_col].cat.rename_categories(lambda x : str(x))
            print("Data property is a categorical variable. Applying cell means model and reporting means.")
            reported_estimates = "means"
        elif is_string_dtype(self.data[data_prop_col]):   
            self.data = self.data.astype({data_prop_col : 'categorical'})
            print("Data property is a categorical variable. Applying cell means model and reporting means.")
            reported_estimates = "means"
        elif is_numeric_dtype(self.data[data_prop_col]):
            print("Data property is a numeric variable. Applying indivdual trends model and reporting slopes.")
            reported_estimates = "slopes"
        else:
            raise ValueError("Data property column " + data_prop_col + " data type is neither numeric nor categorical/string.") 
    
        #minimize data to speed processing and remove rows with missing values  
        model_data = self.data.query(f"{self.system} == '{algorithm_id}'").copy()
        
        if row_filter:
            model_data = model_data.query(row_filter).copy()
        
        model_data = model_data[[hyperparameter_col, data_prop_col, self.metric, self.input_id]].dropna()
        model_data[hyperparameter_col] = model_data[hyperparameter_col].cat.remove_unused_categories()
        if is_categorical_dtype(model_data[data_prop_col]):
            model_data[data_prop_col] = model_data[data_prop_col].cat.remove_unused_categories()
        
        #instantiate and fit models
        if scale_data_prop:
            data_prop_col_m = "scale(" + data_prop_col + ")"
        else:
            data_prop_col_m = data_prop_col
        
        formula_H1 = self.metric + " ~ " + hyperparameter_col + " + " + data_prop_col_m + " + " + hyperparameter_col + ":" + data_prop_col_m + " + ( 1 | " + self.input_id + " )"
        formula_H0 = self.metric + " ~ " + hyperparameter_col + " + " + data_prop_col_m +                                                      " + ( 1 | " + self.input_id + " )"
        
        model_H1 = Lmer(formula = formula_H1, data = model_data, family = self.distribution)
        model_H0 = Lmer(formula = formula_H0, data = model_data, family = self.distribution)
        
        model_factors = {} 
        model_factors[hyperparameter_col] = [s for s in model_data[hyperparameter_col].cat.categories]
        
        if is_categorical_dtype(model_data[data_prop_col]):
            model_data[data_prop_col] = model_data[data_prop_col].cat.remove_unused_categories()
            model_factors[data_prop_col] = [p for p in model_data[data_prop_col].cat.categories]
         
        print("Fitting H0-model.")
        model_H0.fit(factors = model_factors, REML = False, summarize = False)
        print("Fitting H1-model.")
        model_H1.fit(factors = model_factors, REML = False, summarize = False)
        
        
        #compare models and calculate postHoc
        self.ConditionalHyperParameterAssessment.glrt = self.GLRT(model_H0, model_H1)
    
        #FOR CATEGORICAL data property!!!!
        if is_categorical_dtype(model_data[data_prop_col]):
            postHoc_result = [r for r in model_H1.post_hoc(marginal_vars = hyperparameter_col, grouping_vars = data_prop_col)]
        
        if is_numeric_dtype(model_data[data_prop_col]):
            postHoc_result = [r for r in model_H1.post_hoc(marginal_vars = data_prop_col, grouping_vars = hyperparameter_col)]
        
        #simplify postHoc result
        if reported_estimates == "means":
            self.ConditionalHyperParameterAssessment.means = postHoc_result[0].drop(columns = "DF").rename(columns = {"2.5_ci" : "95CI_lo", "97.5_ci" : "95CI_up"})
        else:
            self.ConditionalHyperParameterAssessment.slopes = postHoc_result[0].drop(columns = "DF").rename(columns = {"2.5_ci" : "95CI_lo", "97.5_ci" : "95CI_up"})
            
        self.ConditionalHyperParameterAssessment.contrasts = postHoc_result[1].drop(columns = ["DF", "T-stat", "Z-stat", "Sig"], errors = "ignore").rename(columns = {"2.5_ci" : "95CI_lo", "97.5_ci" : "95CI_up"})
        
        
        
        #add effect size (a Hedge's g derivate) to mean model contrasts
        if reported_estimates == "means":
            sigma_residuals = model_H1.ranef_var.loc["Residual", "Std"]
            self.ConditionalHyperParameterAssessment.contrasts = self.ConditionalHyperParameterAssessment.contrasts.assign(Effect_size_g = lambda df : df.Estimate / sigma_residuals)
        
        
        if is_categorical_dtype(model_data[data_prop_col]):
            self.ConditionalHyperParameterAssessment.interaction_plot = (
                ggplot(postHoc_result[0]) 
                + theme_bw() 
                + theme(panel_grid = element_blank(),
                        legend_position = "top",
                        legend_title = element_blank())
                + xlab(data_prop_col) 
                + ylab("Estimated Expectation of Evaluation Metric")  
                + geom_pointrange(aes(x      = data_prop_col, 
                                      y      = "Estimate",
                                      ymin   = "Estimate - SE",
                                      ymax   = "Estimate + SE",
                                      colour = hyperparameter_col),
                                  alpha = .7) 
                + geom_line(aes(x      = data_prop_col,
                                y      = "Estimate",
                                group  = hyperparameter_col,
                                colour = hyperparameter_col), 
                            alpha = .3)
            )
        
        if is_numeric_dtype(model_data[data_prop_col]):
            self.ConditionalHyperParameterAssessment.interaction_plot = (
                ggplot(data = model_data)
                + theme_bw()
                + theme(panel_grid = element_blank(),
                        legend_position = "top",
                        legend_title = element_blank())
                + xlab(data_prop_col)
                + ylab("Estimated Expected Evaluation Metric")
                + geom_smooth(aes(x = data_prop_col, y = self.metric, group = hyperparameter_col, linetype = hyperparameter_col),
                              method = "lm", 
                              colour = "black", 
                              se     = False)
            )   
        
    
        if self.ConditionalHyperParameterAssessment.glrt["p"] <= alpha and verbose:
            print("GLRT p-value <= alpha: Null hypothesis can be rejected! At least two systems depend differently to the data property. See contrasts for pairwise comparisons.")
        elif verbose:
            print("GLRT p-value > alpha: Null hypothesis can not be rejected! No statistical signifcant difference(s) between systems.")
        
        self.ConditionalHyperParameterAssessment.algorithm = algorithm_id
        self.ConditionalHyperParameterAssessment.data_property = data_prop_col
        return 
    
    
        def conditional_system_comparison_plot(self, data_prop_col, row_filter = ""):
            '''
            '''
            #minimize data to speed processing and removing rows with missing values       
            if row_filter:
                model_data = self.data.query(row_filter).copy()
                model_data = model_data[[self.system, self.metric, data_prop_col, self.input_id]].dropna()
            else:
                model_data = self.data[[self.system, self.metric, data_prop_col, self.input_id]].dropna()
            
            model_data[self.system] = model_data[self.system].cat.remove_unused_categories()
            if is_categorical_dtype(model_data[data_prop_col]):
                model_data[data_prop_col] = model_data[data_prop_col].cat.remove_unused_categories()
    
            if is_numeric_dtype(model_data[data_prop_col]):
                descriptive_plot = (
                    ggplot(data    = model_data, 
                           mapping = aes(x = data_prop_col, y = self.metric, color = self.system))
                    + theme_bw()
                    + theme(panel_grid      = element_blank(),
                            legend_position = "none",
                            legend_title    = element_blank())
                    + xlab(data_prop_col)
                    + ylab("Evaluation Metric")
                    + facet_wrap("system")
                    + geom_point(alpha = .01)
                    #+ geom_density_2d(alpha = .3)                   
                    + geom_smooth(method = "loess", se = False) 
                )
            elif is_categorical_dtype(model_data[data_prop_col]) or is_string_dtype(model_data[data_prop_col]):
                descriptive_plot = (
                    ggplot(data    = model_data, 
                           mapping = aes(x = data_prop_col, y = self.metric, fill = self.system))
                    + theme_bw()
                    + theme(panel_grid      = element_blank(),
                            legend_position = "none",
                            legend_title    = element_blank())
                    + xlab(data_prop_col)
                    + ylab("Evaluation Metric")
                    + geom_boxplot(alpha = .3)
                )
            else:
                raise ValueError("No plot defined for the data type of data_prop_var " + data_prop_var + ".")
        
            return descriptive_plot