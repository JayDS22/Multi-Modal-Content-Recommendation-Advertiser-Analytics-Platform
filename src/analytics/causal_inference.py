"""
Causal Inference and A/B Testing Framework
Implements DiD, Synthetic Control, and statistical testing
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging


@dataclass
class ABTestResult:
    """Results from A/B test analysis"""
    metric_name: str
    control_mean: float
    treatment_mean: float
    absolute_lift: float
    relative_lift: float
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    sample_size_control: int
    sample_size_treatment: int
    

class CausalInferenceAnalyzer:
    """
    Implements causal inference methods for measuring campaign effectiveness
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Significance level for hypothesis testing
        """
        self.alpha = alpha
        self.logger = logging.getLogger(__name__)
    
    def difference_in_differences(
        self,
        pre_treatment_control: np.ndarray,
        post_treatment_control: np.ndarray,
        pre_treatment_treated: np.ndarray,
        post_treatment_treated: np.ndarray
    ) -> Dict[str, float]:
        """
        Difference-in-Differences (DiD) estimation
        
        Args:
            pre_treatment_control: Control group metrics before treatment
            post_treatment_control: Control group metrics after treatment
            pre_treatment_treated: Treatment group metrics before treatment
            post_treatment_treated: Treatment group metrics after treatment
        
        Returns:
            Dictionary with DiD estimate, standard error, and p-value
        """
        # Calculate means
        mean_pre_control = np.mean(pre_treatment_control)
        mean_post_control = np.mean(post_treatment_control)
        mean_pre_treated = np.mean(pre_treatment_treated)
        mean_post_treated = np.mean(post_treatment_treated)
        
        # DiD estimate
        diff_control = mean_post_control - mean_pre_control
        diff_treated = mean_post_treated - mean_pre_treated
        did_estimate = diff_treated - diff_control
        
        # Calculate standard error using pooled variance
        var_pre_control = np.var(pre_treatment_control, ddof=1)
        var_post_control = np.var(post_treatment_control, ddof=1)
        var_pre_treated = np.var(pre_treatment_treated, ddof=1)
        var_post_treated = np.var(post_treatment_treated, ddof=1)
        
        n_pre_control = len(pre_treatment_control)
        n_post_control = len(post_treatment_control)
        n_pre_treated = len(pre_treatment_treated)
        n_post_treated = len(post_treatment_treated)
        
        se_did = np.sqrt(
            var_pre_control / n_pre_control +
            var_post_control / n_post_control +
            var_pre_treated / n_pre_treated +
            var_post_treated / n_post_treated
        )
        
        # T-test
        t_stat = did_estimate / se_did
        df = n_pre_control + n_post_control + n_pre_treated + n_post_treated - 4
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        # 95% confidence interval
        ci_lower = did_estimate - 1.96 * se_did
        ci_upper = did_estimate + 1.96 * se_did
        
        return {
            'did_estimate': did_estimate,
            'standard_error': se_did,
            't_statistic': t_stat,
            'p_value': p_value,
            'confidence_interval': (ci_lower, ci_upper),
            'is_significant': p_value < self.alpha
        }
    
    def synthetic_control(
        self,
        treated_pre: np.ndarray,
        treated_post: np.ndarray,
        control_units_pre: np.ndarray,
        control_units_post: np.ndarray,
        optimization_method: str = 'minimize_mse'
    ) -> Dict[str, float]:
        """
        Synthetic Control Method
        
        Args:
            treated_pre: Pre-treatment outcomes for treated unit (T,)
            treated_post: Post-treatment outcomes for treated unit (T,)
            control_units_pre: Pre-treatment outcomes for control units (N, T)
            control_units_post: Post-treatment outcomes for control units (N, T)
            optimization_method: Method to find optimal weights
        
        Returns:
            Dictionary with treatment effect and weights
        """
        n_control_units = control_units_pre.shape[0]
        
        # Find optimal weights to match pre-treatment period
        if optimization_method == 'minimize_mse':
            from scipy.optimize import minimize
            
            def objective(weights):
                synthetic = np.dot(weights, control_units_pre)
                return np.sum((treated_pre - synthetic) ** 2)
            
            # Constraints: weights sum to 1 and are non-negative
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0, 1) for _ in range(n_control_units)]
            initial_weights = np.ones(n_control_units) / n_control_units
            
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            optimal_weights = result.x
        else:
            # Equal weights as baseline
            optimal_weights = np.ones(n_control_units) / n_control_units
        
        # Create synthetic control
        synthetic_pre = np.dot(optimal_weights, control_units_pre)
        synthetic_post = np.dot(optimal_weights, control_units_post)
        
        # Calculate treatment effect
        pre_treatment_fit = np.mean(np.abs(treated_pre - synthetic_pre))
        treatment_effect = np.mean(treated_post - synthetic_post)
        
        # Calculate post-treatment RMSE
        post_rmse = np.sqrt(np.mean((treated_post - synthetic_post) ** 2))
        
        return {
            'treatment_effect': treatment_effect,
            'weights': optimal_weights,
            'pre_treatment_fit': pre_treatment_fit,
            'post_treatment_rmse': post_rmse
        }
    
    def calculate_roas(
        self,
        revenue: np.ndarray,
        ad_spend: np.ndarray,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate Return on Ad Spend (ROAS) with confidence intervals
        
        Args:
            revenue: Array of revenue values
            ad_spend: Array of ad spend values
            confidence_level: Confidence level for intervals
        
        Returns:
            Dictionary with ROAS metrics
        """
        # Calculate ROAS for each observation
        roas_values = revenue / (ad_spend + 1e-10)  # Avoid division by zero
        
        # Statistics
        mean_roas = np.mean(roas_values)
        std_roas = np.std(roas_values, ddof=1)
        se_roas = std_roas / np.sqrt(len(roas_values))
        
        # Confidence interval
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        ci_lower = mean_roas - z_score * se_roas
        ci_upper = mean_roas + z_score * se_roas
        
        # Total ROAS
        total_roas = np.sum(revenue) / np.sum(ad_spend)
        
        return {
            'mean_roas': mean_roas,
            'total_roas': total_roas,
            'std_roas': std_roas,
            'confidence_interval': (ci_lower, ci_upper),
            'sample_size': len(roas_values)
        }
    
    def incremental_lift(
        self,
        treatment_conversions: int,
        treatment_impressions: int,
        control_conversions: int,
        control_impressions: int
    ) -> Dict[str, float]:
        """
        Calculate incremental lift from treatment
        
        Args:
            treatment_conversions: Number of conversions in treatment
            treatment_impressions: Number of impressions in treatment
            control_conversions: Number of conversions in control
            control_impressions: Number of impressions in control
        
        Returns:
            Dictionary with lift metrics
        """
        # Conversion rates
        treatment_rate = treatment_conversions / treatment_impressions
        control_rate = control_conversions / control_impressions
        
        # Absolute and relative lift
        absolute_lift = treatment_rate - control_rate
        relative_lift = (absolute_lift / control_rate) * 100 if control_rate > 0 else 0
        
        # Statistical test (two-proportion z-test)
        pooled_rate = (treatment_conversions + control_conversions) / \
                     (treatment_impressions + control_impressions)
        
        se = np.sqrt(pooled_rate * (1 - pooled_rate) * 
                    (1/treatment_impressions + 1/control_impressions))
        
        z_stat = absolute_lift / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Confidence interval
        se_diff = np.sqrt(
            treatment_rate * (1 - treatment_rate) / treatment_impressions +
            control_rate * (1 - control_rate) / control_impressions
        )
        ci_lower = absolute_lift - 1.96 * se_diff
        ci_upper = absolute_lift + 1.96 * se_diff
        
        return {
            'treatment_rate': treatment_rate,
            'control_rate': control_rate,
            'absolute_lift': absolute_lift,
            'relative_lift': relative_lift,
            'z_statistic': z_stat,
            'p_value': p_value,
            'confidence_interval': (ci_lower, ci_upper),
            'is_significant': p_value < self.alpha
        }


class ABTestFramework:
    """
    Comprehensive A/B testing framework with multiple metrics
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.causal_analyzer = CausalInferenceAnalyzer(alpha=alpha)
        self.logger = logging.getLogger(__name__)
    
    def analyze_metric(
        self,
        control_data: np.ndarray,
        treatment_data: np.ndarray,
        metric_name: str,
        test_type: str = 'ttest'
    ) -> ABTestResult:
        """
        Analyze a single metric between control and treatment
        
        Args:
            control_data: Metric values for control group
            treatment_data: Metric values for treatment group
            metric_name: Name of the metric
            test_type: Type of statistical test ('ttest', 'mannwhitney')
        
        Returns:
            ABTestResult object with analysis results
        """
        # Calculate means
        control_mean = np.mean(control_data)
        treatment_mean = np.mean(treatment_data)
        
        # Calculate lift
        absolute_lift = treatment_mean - control_mean
        relative_lift = (absolute_lift / control_mean * 100) if control_mean != 0 else 0
        
        # Statistical test
        if test_type == 'ttest':
            t_stat, p_value = stats.ttest_ind(treatment_data, control_data)
            
            # Confidence interval for difference
            se_diff = np.sqrt(
                np.var(treatment_data, ddof=1) / len(treatment_data) +
                np.var(control_data, ddof=1) / len(control_data)
            )
            ci_lower = absolute_lift - 1.96 * se_diff
            ci_upper = absolute_lift + 1.96 * se_diff
            
        elif test_type == 'mannwhitney':
            u_stat, p_value = stats.mannwhitneyu(
                treatment_data, control_data, alternative='two-sided'
            )
            
            # Bootstrap confidence interval
            n_bootstrap = 1000
            diffs = []
            for _ in range(n_bootstrap):
                ctrl_sample = np.random.choice(control_data, len(control_data), replace=True)
                treat_sample = np.random.choice(treatment_data, len(treatment_data), replace=True)
                diffs.append(np.mean(treat_sample) - np.mean(ctrl_sample))
            
            ci_lower = np.percentile(diffs, 2.5)
            ci_upper = np.percentile(diffs, 97.5)
        
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        return ABTestResult(
            metric_name=metric_name,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            absolute_lift=absolute_lift,
            relative_lift=relative_lift,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < self.alpha,
            sample_size_control=len(control_data),
            sample_size_treatment=len(treatment_data)
        )
    
    def analyze_campaign(
        self,
        campaign_data: pd.DataFrame,
        metrics: List[str]
    ) -> Dict[str, ABTestResult]:
        """
        Analyze multiple metrics for a campaign
        
        Args:
            campaign_data: DataFrame with columns ['group', metric1, metric2, ...]
            metrics: List of metric names to analyze
        
        Returns:
            Dictionary mapping metric names to ABTestResult objects
        """
        results = {}
        
        for metric in metrics:
            control_data = campaign_data[campaign_data['group'] == 'control'][metric].values
            treatment_data = campaign_data[campaign_data['group'] == 'treatment'][metric].values
            
            result = self.analyze_metric(
                control_data,
                treatment_data,
                metric_name=metric
            )
            
            results[metric] = result
            
            self.logger.info(
                f"{metric}: Control={result.control_mean:.4f}, "
                f"Treatment={result.treatment_mean:.4f}, "
                f"Lift={result.relative_lift:.2f}%, "
                f"p-value={result.p_value:.4f}"
            )
        
        return results
    
    def power_analysis(
        self,
        baseline_mean: float,
        baseline_std: float,
        mde: float,  # Minimum detectable effect
        alpha: float = 0.05,
        power: float = 0.8
    ) -> int:
        """
        Calculate required sample size for A/B test
        
        Args:
            baseline_mean: Baseline metric mean
            baseline_std: Baseline metric standard deviation
            mde: Minimum detectable effect (absolute or relative)
            alpha: Significance level
            power: Statistical power (1 - beta)
        
        Returns:
            Required sample size per group
        """
        from statsmodels.stats.power import tt_ind_solve_power
        
        # Effect size (Cohen's d)
        effect_size = mde / baseline_std
        
        # Calculate sample size
        n = tt_ind_solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            ratio=1.0,
            alternative='two-sided'
        )
        
        return int(np.ceil(n))


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate synthetic campaign data
    n_samples = 10000
    
    campaign_data = pd.DataFrame({
        'group': ['control'] * (n_samples // 2) + ['treatment'] * (n_samples // 2),
        'ctr': np.concatenate([
            np.random.normal(0.024, 0.01, n_samples // 2),  # Control
            np.random.normal(0.031, 0.01, n_samples // 2)   # Treatment (+31%)
        ]),
        'conversion_rate': np.concatenate([
            np.random.normal(0.018, 0.005, n_samples // 2),  # Control
            np.random.normal(0.022, 0.005, n_samples // 2)   # Treatment (+23%)
        ]),
        'dwell_time': np.concatenate([
            np.random.normal(45, 15, n_samples // 2),  # Control
            np.random.normal(64, 15, n_samples // 2)   # Treatment (+42%)
        ])
    })
    
    # Run analysis
    ab_framework = ABTestFramework()
    results = ab_framework.analyze_campaign(
        campaign_data,
        metrics=['ctr', 'conversion_rate', 'dwell_time']
    )
    
    print("\n=== A/B Test Results ===")
    for metric, result in results.items():
        print(f"\n{metric.upper()}:")
        print(f"  Control: {result.control_mean:.4f}")
        print(f"  Treatment: {result.treatment_mean:.4f}")
        print(f"  Lift: {result.relative_lift:.2f}%")
        print(f"  p-value: {result.p_value:.4f}")
        print(f"  Significant: {result.is_significant}")
