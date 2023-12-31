use pyo3::prelude::*;
mod decisiontree;
mod scoring;
// mod utils;

#[pymodule]
#[pyo3(name = "_rust")]
fn random_tree_models(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    register_scoring_module(py, m)?;
    // m.add_class::<DecisionTree>()?;
    Ok(())
}

/// Scoring module implemented in Rust.
fn register_scoring_module(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let child_module = PyModule::new(py, "scoring_rs")?;
    child_module.add_function(wrap_pyfunction!(scoring::gini_impurity_py, child_module)?)?;
    child_module.add_function(wrap_pyfunction!(scoring::entropy_py, child_module)?)?;
    parent_module.add_submodule(child_module)?;
    Ok(())
}

// #[pyclass]
// struct DecisionTree {
//     num: usize,
// }

// #[pymethods]
// impl DecisionTree {
//     #[new]
//     fn new(num: usize) -> Self {
//         DecisionTree { num }
//     }

//     fn get_num(&self) -> usize {
//         self.num
//     }
// }
