use polars::{frame::DataFrame, series::Series};
use pyo3::prelude::*;
mod decisiontree;
mod scoring;
use pyo3_polars::{PyDataFrame, PySeries};
mod utils;

#[pymodule]
#[pyo3(name = "_rust")]
fn random_tree_models(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    register_scoring_module(py, m)?;
    m.add_class::<DecisionTree>()?;
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

#[pyclass]
struct DecisionTree {
    max_depth: usize,
    tree_: Option<decisiontree::DecisionTreeTemplate>,
}

#[pymethods]
impl DecisionTree {
    #[new]
    fn new(max_depth: usize) -> Self {
        DecisionTree {
            max_depth,
            tree_: None,
        }
    }

    fn fit(&mut self, X: PyDataFrame, y: PySeries) -> PyResult<()> {
        let mut tree = decisiontree::DecisionTreeTemplate::new(self.max_depth);
        let X: DataFrame = X.into();
        let y: Series = y.into();
        tree.fit(&X, &y);
        self.tree_ = Some(tree);
        Ok(())
    }

    fn predict(&self, X: PyDataFrame) -> PyResult<PySeries> {
        let X: DataFrame = X.into();
        let y_pred = self.tree_.as_ref().unwrap().predict(&X);

        Ok(PySeries(y_pred))
    }
}
