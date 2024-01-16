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
    m.add_class::<DecisionTreeClassifier>()?;
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
struct DecisionTreeClassifier {
    max_depth: usize,
    tree_: Option<decisiontree::DecisionTreeClassifier>,
}

#[pymethods]
impl DecisionTreeClassifier {
    #[new]
    fn new(max_depth: usize) -> Self {
        DecisionTreeClassifier {
            max_depth,
            tree_: None,
        }
    }

    fn fit(&mut self, x: PyDataFrame, y: PySeries) -> PyResult<()> {
        let mut tree = decisiontree::DecisionTreeClassifier::new(self.max_depth);
        let x: DataFrame = x.into();
        let y: Series = y.into();
        tree.fit(&x, &y);
        self.tree_ = Some(tree);
        Ok(())
    }

    fn predict(&self, x: PyDataFrame) -> PyResult<PySeries> {
        let x: DataFrame = x.into();
        let y_pred = self.tree_.as_ref().unwrap().predict(&x);

        Ok(PySeries(y_pred))
    }

    fn predict_proba(&self, x: PyDataFrame) -> PyResult<PyDataFrame> {
        let x: DataFrame = x.into();
        let y_pred = self.tree_.as_ref().unwrap().predict_proba(&x);

        Ok(PyDataFrame(y_pred))
    }
}

// TODO: implement DecisionTreeRegressor
