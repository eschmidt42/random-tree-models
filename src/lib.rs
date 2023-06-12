use pyo3::prelude::*;
mod scoring;

#[pymodule]
#[pyo3(name = "_rust")]
fn random_tree_models(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    register_scoring_module(py, m)?;
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
