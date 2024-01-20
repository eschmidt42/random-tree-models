use std::collections::HashMap;

use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::utils::{SplitScoreMetrics, TreeGrowthParameters};

// compute gini impurity of an array of discrete values
#[pyfunction(name = "gini_impurity")]
pub fn gini_impurity_py(values: Vec<usize>) -> PyResult<f64> {
    // if values if empty raise pyvalueerror
    if values.is_empty() {
        return Err(PyValueError::new_err("values cannot be empty"));
    }

    Ok(gini_impurity(values))
}

fn gini_impurity(values: Vec<usize>) -> f64 {
    let mut counts: HashMap<usize, usize> = HashMap::new();
    let mut total = 0;
    for v in values {
        let count = counts.entry(v).or_insert(0);
        *count += 1;
        total += 1;
    }
    let mut impurity = 0.0;
    for (_, count) in counts {
        let p = count as f64 / total as f64;
        impurity += p * (1.0 - p);
    }
    -impurity
}

// pyo3 handle of entropy function
#[pyfunction(name = "entropy")]
pub fn entropy_py(values: Vec<usize>) -> PyResult<f64> {
    // if values if empty raise pyvalueerror
    if values.is_empty() {
        return Err(PyValueError::new_err("values cannot be empty"));
    }
    Ok(entropy(values)) // unwrap the result and return
}

// compute entropy of an array of discrete values
fn entropy(values: Vec<usize>) -> f64 {
    let mut counts: HashMap<usize, usize> = HashMap::new();
    let mut total = 0;
    for v in values {
        let count = counts.entry(v).or_insert(0);
        *count += 1;
        total += 1;
    }
    let mut entropy = 0.0;
    for (_, count) in counts {
        let p = count as f64 / total as f64;
        entropy += p * p.log2();
    }
    entropy
}

pub fn count_y_values(y: &Series) -> Series {
    let df = y.value_counts(false, false).unwrap();
    let counts: Series = df.select_at_idx(1).unwrap().clone();
    counts
}

pub fn calc_probabilities(y: &Series) -> Series {
    let msg = "Could not cast to f64";
    let counts = count_y_values(y);
    let counts = counts.cast(&DataType::Float64).expect(msg);
    let total: f64 = counts.sum().unwrap();
    let ps = Series::new("probs", counts / total);
    ps
}

pub fn calc_neg_entropy_series(ps: &Series) -> f64 {
    let neg_entropy = ps
        .f64()
        .expect("not f64 dtype")
        .into_iter()
        .map(|x| x.unwrap() * x.unwrap().log2())
        .sum();
    neg_entropy
}

pub fn neg_entropy_rs(y: &Series, target_groups: &Series) -> f64 {
    let msg = "Could not cast to f64";
    let w_left: f64 = (*target_groups)
        .cast(&polars::datatypes::DataType::Float64)
        .expect(msg)
        .sum::<f64>()
        .unwrap()
        / y.len() as f64;
    let w_right: f64 = 1.0 - w_left;

    // generate boolean chunked array of target_groups
    let trues = Series::new("", vec![true; target_groups.len()]);
    let target_groups = target_groups.equal(&trues).unwrap();

    let entropy_left: f64;
    let entropy_right: f64;
    if w_left > 0. {
        let y_left = y.filter(&target_groups).unwrap();
        let probs = calc_probabilities(&y_left);
        entropy_left = calc_neg_entropy_series(&probs);
    } else {
        entropy_left = 0.0;
    }
    if w_right > 0. {
        let y_right = y.filter(&!target_groups).unwrap();
        let probs = calc_probabilities(&y_right);
        entropy_right = calc_neg_entropy_series(&probs);
    } else {
        entropy_right = 0.0;
    }
    let score = (w_left * entropy_left) + (w_right * entropy_right);
    score
}

pub fn neg_variance_rs(y: &Series, target_groups: &Series) -> f64 {
    let msg = "Could not cast to f64";
    let w_left: f64 = (*target_groups)
        .cast(&polars::datatypes::DataType::Float64)
        .expect(msg)
        .sum::<f64>()
        .unwrap()
        / y.len() as f64;
    let w_right: f64 = 1.0 - w_left;

    // generate boolean chunked array of target_groups
    let trues = Series::new("", vec![true; target_groups.len()]);
    let target_groups = target_groups.equal(&trues).unwrap();

    let variance_left: f64;
    let variance_right: f64;
    if w_left > 0. {
        let y_left = y.filter(&target_groups).unwrap();
        let ddof_left: u8 = (y_left.len() - 1).try_into().unwrap();
        variance_left = y_left
            .var_as_series(ddof_left)
            .unwrap()
            .f64()
            .expect("not f64")
            .get(0)
            .expect("was null");
    } else {
        variance_left = 0.0;
    }
    if w_right > 0. {
        let y_right = y.filter(&!target_groups).unwrap();
        let ddof_right: u8 = (y_right.len() - 1).try_into().unwrap();
        variance_right = y_right
            .var_as_series(ddof_right)
            .unwrap()
            .f64()
            .expect("not f64")
            .get(0)
            .expect("was null");
    } else {
        variance_right = 0.0;
    }
    let score = (w_left * variance_left) + (w_right * variance_right);
    -score
}

pub fn calc_score(
    y: &Series,
    target_groups: &Series,
    growth_params: &TreeGrowthParameters,
    _g: Option<&Series>,
    _h: Option<&Series>,
    _incrementing_score: Option<f64>,
) -> f64 {
    match growth_params.split_score_metric {
        Some(SplitScoreMetrics::NegEntropy) => neg_entropy_rs(y, target_groups),
        Some(SplitScoreMetrics::NegVariance) => neg_variance_rs(y, target_groups),
        _ => panic!(
            "split_score_metric {:?} not supported",
            growth_params.split_score_metric
        ),
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    // test that gini impurity correctly computes values smaller than zero for a couple of vectors

    #[test]
    fn test_gini_impurity() {
        let values = vec![0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(gini_impurity(values), 0.0);
        let values = vec![0, 1, 0, 1, 0, 1, 0, 1];
        assert_eq!(gini_impurity(values), -0.5);
        let values = vec![0, 1, 2, 3, 4, 5, 6, 7];
        assert_eq!(gini_impurity(values), -0.875);
    }
    // test that entropy correctly computes values smaller than zero for a couple of vectors
    #[test]
    fn test_entropy() {
        let values = vec![0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(entropy(values), 0.0);
        let values = vec![0, 1, 0, 1, 0, 1, 0, 1];
        assert_eq!(entropy(values), -1.0);
        let values = vec![0, 1, 2, 3, 4, 5, 6, 7];
        assert_eq!(entropy(values), -3.0);
    }
    // test count_y_values
    #[test]
    fn test_count_y_values() {
        let values = vec![0, 0, 0, 0, 0, 0, 0, 0];
        let s = Series::new("y", values);
        let counts = count_y_values(&s);
        // assert that counts is a series with one value
        let exp: Vec<u32> = vec![8];
        assert_eq!(counts, Series::new("count", exp));

        let values = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let s = Series::new("y", values);
        let counts = count_y_values(&s);
        let exp: Vec<u32> = vec![4, 4];
        assert_eq!(counts, Series::new("count", exp));

        let values = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let s = Series::new("y", values);
        let counts = count_y_values(&s);
        let exp: Vec<u32> = vec![1, 1, 1, 1, 1, 1, 1, 1];
        assert_eq!(counts, Series::new("count", exp));
    }

    // test calc_probabilities
    #[test]
    fn test_calc_probabilities() {
        let values = vec![0, 0, 0, 0, 0, 0, 0, 0];
        let s = Series::new("y", values);
        let probs = calc_probabilities(&s);
        // assert that counts is a series with one value
        let exp: Vec<f64> = vec![1.0];
        assert_eq!(probs, Series::new("probs", exp));

        let values = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let s = Series::new("y", values);
        let probs = calc_probabilities(&s);
        let exp: Vec<f64> = vec![0.5, 0.5];
        assert_eq!(probs, Series::new("probs", exp));

        let values = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let s = Series::new("y", values);
        let probs = calc_probabilities(&s);
        let exp: Vec<f64> = vec![0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125];
        assert_eq!(probs, Series::new("probs", exp));
    }

    // test calc_neg_entropy_series
    #[test]
    fn test_calc_neg_entropy_series() {
        let values = vec![0, 0, 0, 0, 0, 0, 0, 0];
        let s = Series::new("y", values);
        let probs = calc_probabilities(&s);
        let neg_entropy = calc_neg_entropy_series(&probs);
        assert_eq!(neg_entropy, 0.0);

        let values = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let s = Series::new("y", values);
        let probs = calc_probabilities(&s);
        let neg_entropy = calc_neg_entropy_series(&probs);
        assert_eq!(neg_entropy, -1.0);

        let values = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let s = Series::new("y", values);
        let probs = calc_probabilities(&s);
        let neg_entropy = calc_neg_entropy_series(&probs);
        assert_eq!(neg_entropy, -3.0);
    }

    // test calc_score
    #[test]
    fn test_calc_score() {
        let values = vec![0, 0, 0, 0, 0, 0, 0, 0];
        let s = Series::new("y", values);
        let target_groups = Series::new("target_groups", vec![true; 8]);
        let growth_params = TreeGrowthParameters {
            max_depth: Some(1),
            split_score_metric: Some(SplitScoreMetrics::NegEntropy),
        };
        let score = calc_score(&s, &target_groups, &growth_params, None, None, None);
        assert_eq!(score, 0.0);

        let values = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let s = Series::new("y", values);
        let target_groups = Series::new("target_groups", vec![true; 8]);
        let growth_params = TreeGrowthParameters {
            max_depth: Some(1),
            split_score_metric: Some(SplitScoreMetrics::NegEntropy),
        };
        let score = calc_score(&s, &target_groups, &growth_params, None, None, None);
        assert_eq!(score, -1.0);

        let values = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let s = Series::new("y", values);
        let target_groups = Series::new("target_groups", vec![true; 8]);
        let growth_params = TreeGrowthParameters {
            max_depth: Some(1),
            split_score_metric: Some(SplitScoreMetrics::NegEntropy),
        };
        let score = calc_score(&s, &target_groups, &growth_params, None, None, None);
        assert_eq!(score, -3.0);

        let values = vec![0, 0, 0, 0, 0, 0, 0, 0];
        let s = Series::new("y", values);
        let target_groups = Series::new(
            "target_groups",
            vec![true, true, true, true, false, false, false, false],
        );
        let growth_params = TreeGrowthParameters {
            max_depth: Some(1),
            split_score_metric: Some(SplitScoreMetrics::NegEntropy),
        };
        let score = calc_score(&s, &target_groups, &growth_params, None, None, None);
        assert_eq!(score, 0.0);

        let values = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let s = Series::new("y", values);
        let target_groups = Series::new(
            "target_groups",
            vec![true, true, true, true, false, false, false, false],
        );
        let growth_params = TreeGrowthParameters {
            max_depth: Some(1),
            split_score_metric: Some(SplitScoreMetrics::NegEntropy),
        };
        let score = calc_score(&s, &target_groups, &growth_params, None, None, None);
        assert_eq!(score, -1.0);
    }

    // test entropy_rs
    #[test]
    fn test_entropy_rs() {
        let values = vec![0, 0, 0, 0, 0, 0, 0, 0];
        let s = Series::new("y", values);
        let target_groups = Series::new("target_groups", vec![true; 8]);
        let score = neg_entropy_rs(&s, &target_groups);
        assert_eq!(score, 0.0);

        let values = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let s = Series::new("y", values);
        let target_groups = Series::new("target_groups", vec![true; 8]);
        let score = neg_entropy_rs(&s, &target_groups);
        assert_eq!(score, -1.0);

        let values = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let s = Series::new("y", values);
        let target_groups = Series::new("target_groups", vec![true; 8]);
        let score = neg_entropy_rs(&s, &target_groups);
        assert_eq!(score, -3.0);

        let values = vec![0, 0, 0, 0, 0, 0, 0, 0];
        let s = Series::new("y", values);
        let target_groups = Series::new(
            "target_groups",
            vec![true, true, true, true, false, false, false, false],
        );
        let score = neg_entropy_rs(&s, &target_groups);
        assert_eq!(score, 0.0);

        let values = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let s = Series::new("y", values);
        let target_groups = Series::new(
            "target_groups",
            vec![true, true, true, true, false, false, false, false],
        );
        let score = neg_entropy_rs(&s, &target_groups);
        assert_eq!(score, -1.0);
    }

    // test neg_variance_rs
    #[test]
    fn test_neg_variance_rs() {
        let values = vec![0, 0, 0, 0, 0, 0, 0, 0];
        let s = Series::new("y", values);
        let target_groups = Series::new("target_groups", vec![true; 8]);
        let score = neg_variance_rs(&s, &target_groups);
        assert_eq!(score, 0.0);

        let values = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let s = Series::new("y", values);
        let target_groups = Series::new("target_groups", vec![true; 8]);
        let score = neg_variance_rs(&s, &target_groups);
        assert_eq!(score, -2.);

        let values = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let s = Series::new("y", values);
        let target_groups = Series::new("target_groups", vec![true; 8]);
        let score = neg_variance_rs(&s, &target_groups);
        assert_eq!(score, -42.);

        let values = vec![0, 0, 0, 0, 0, 0, 0, 0];
        let s = Series::new("y", values);
        let target_groups = Series::new(
            "target_groups",
            vec![true, true, true, true, false, false, false, false],
        );
        let score = neg_variance_rs(&s, &target_groups);
        assert_eq!(score, 0.0);

        let values = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let s = Series::new("y", values);
        let target_groups = Series::new(
            "target_groups",
            vec![true, true, true, true, false, false, false, false],
        );
        let score = neg_variance_rs(&s, &target_groups);
        assert_eq!(score, -1.);
    }
}
