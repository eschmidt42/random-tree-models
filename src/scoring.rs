use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// compute gini impurity of an array of discrete values
#[pyfunction]
pub fn gini_impurity(values: Vec<usize>) -> PyResult<f64> {
    // if values if empty raise pyvalueerror
    if values.is_empty() {
        return Err(PyValueError::new_err("values cannot be empty"));
    }
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
    Ok(-impurity)
}
