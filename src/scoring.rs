use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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

mod tests {
    // test that gini impurity correctly computes values smaller than zero for a couple of vectors
    #[test]
    fn test_gini_impurity() {
        let values = vec![0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(super::gini_impurity(values), 0.0);
        let values = vec![0, 1, 0, 1, 0, 1, 0, 1];
        assert_eq!(super::gini_impurity(values), -0.5);
        let values = vec![0, 1, 2, 3, 4, 5, 6, 7];
        assert_eq!(super::gini_impurity(values), -0.875);
    }
    // test that entropy correctly computes values smaller than zero for a couple of vectors
    #[test]
    fn test_entropy() {
        let values = vec![0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(super::entropy(values), 0.0);
        let values = vec![0, 1, 0, 1, 0, 1, 0, 1];
        assert_eq!(super::entropy(values), -1.0);
        let values = vec![0, 1, 2, 3, 4, 5, 6, 7];
        assert_eq!(super::entropy(values), -3.0);
    }
}
