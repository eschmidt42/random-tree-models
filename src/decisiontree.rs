use polars::{lazy::dsl::GetOutput, prelude::*};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use uuid::Uuid;

use crate::{scoring, utils::TreeGrowthParameters};

#[derive(PartialEq, Debug, Clone)]
pub struct SplitScore {
    pub name: String,
    pub score: f64,
}

impl SplitScore {
    pub fn new(name: String, score: f64) -> Self {
        SplitScore { name, score }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct Node {
    pub column: Option<String>,
    pub column_idx: Option<usize>,
    pub threshold: Option<f64>,
    pub prediction: Option<f64>,
    pub default_is_left: Option<bool>,

    // descendants
    pub left: Option<Box<Node>>,
    pub right: Option<Box<Node>>,

    // misc
    pub measure: Option<SplitScore>,

    pub n_obs: usize,
    pub reason: String,
    pub depth: usize,
    pub node_id: Uuid,
}

impl Node {
    pub fn new(
        column: Option<String>,
        column_idx: Option<usize>,
        threshold: Option<f64>,
        prediction: Option<f64>,
        default_is_left: Option<bool>,
        left: Option<Box<Node>>,
        right: Option<Box<Node>>,
        measure: Option<SplitScore>,
        n_obs: usize,
        reason: String,
        depth: usize,
    ) -> Self {
        let node_id = Uuid::new_v4();
        Node {
            column,
            column_idx,
            threshold,
            prediction,
            default_is_left,
            left,
            right,
            measure,
            n_obs,
            reason,
            depth,
            node_id,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }

    pub fn insert(&mut self, new_node: Node, insert_left: bool) {
        if insert_left {
            match self.left {
                Some(ref mut _left) => {
                    panic!("Something went wrong. The left node is already occupied.")
                } // left.insert(new_node)
                None => self.left = Some(Box::new(new_node)),
            }
        } else {
            match self.right {
                Some(ref mut _right) => {
                    panic!("Something went wrong. The right node is already occupied.")
                }
                //right.insert(new_node),
                None => self.right = Some(Box::new(new_node)),
            }
        }
    }
}

pub fn check_is_baselevel(
    y: &Series,
    depth: usize,
    growth_params: &TreeGrowthParameters,
) -> (bool, String) {
    let n_obs = y.len();
    let n_unique = y
        .n_unique()
        .expect("Something went wrong. Could not get n_unique.");
    let max_depth = growth_params.max_depth;

    if max_depth.is_some() && depth >= max_depth.unwrap() {
        return (true, "max depth reached".to_string());
    } else if n_unique == 1 {
        return (true, "homogenous group".to_string());
    } else if n_obs <= 1 {
        return (true, "<= 1 data point in group".to_string());
    } else {
        (false, "".to_string())
    }
}

pub fn calc_leaf_weight(
    y: &Series,
    growth_params: &TreeGrowthParameters,
    g: Option<&Series>,
    h: Option<&Series>,
) -> f64 {
    let leaf_weight = y.mean().unwrap();

    leaf_weight
}

pub fn calc_leaf_weight_and_split_score(
    y: &Series,
    growth_params: &TreeGrowthParameters,
    g: Option<&Series>,
    h: Option<&Series>,
    incrementing_score: Option<f64>,
) -> (f64, SplitScore) {
    let leaf_weight = calc_leaf_weight(y, growth_params, g, h);

    let target_groups: Series = Series::new("target_groups", vec![true; y.len()]);
    let score = scoring::calc_score(y, &target_groups, growth_params, g, h, incrementing_score);
    let score = SplitScore::new("neg_entropy".to_string(), score);

    (leaf_weight, score)
}

pub struct BestSplit {
    pub score: f64,
    pub column: String,
    pub column_idx: usize,
    pub threshold: f64,
    pub target_groups: Series,
    pub default_is_left: Option<bool>,
}

impl BestSplit {
    pub fn new(
        score: f64,
        column: String,
        column_idx: usize,
        threshold: f64,
        target_groups: Series,
        default_is_left: Option<bool>,
    ) -> Self {
        BestSplit {
            score,
            column,
            column_idx,
            threshold,
            target_groups,
            default_is_left,
        }
    }
}

pub fn find_best_split(
    x: &DataFrame,
    y: &Series,
    growth_params: &TreeGrowthParameters,
    g: Option<&Series>,
    h: Option<&Series>,
    incrementing_score: Option<f64>,
) -> BestSplit {
    if y.len() <= 1 {
        panic!("Something went wrong. The parent_node handed down less than two data points.")
    }

    let mut best_split: Option<BestSplit> = None;

    for (idx, col) in x.get_column_names().iter().enumerate() {
        let mut feature_values = x.select_series(&[col]).unwrap()[0]
            .clone()
            .cast(&DataType::Float64)
            .unwrap();
        feature_values = feature_values.sort(false);

        for value in feature_values.iter() {
            let value: f64 = value.try_extract().unwrap();
            let target_groups = feature_values.lt(value).unwrap();
            let target_groups = Series::new("target_groups", target_groups);

            let score =
                scoring::calc_score(y, &target_groups, growth_params, g, h, incrementing_score);

            match best_split {
                Some(ref mut best_split) => {
                    if score < best_split.score {
                        best_split.score = score;
                        best_split.column = col.to_string();
                        best_split.threshold = value;
                        best_split.target_groups = target_groups;
                        best_split.default_is_left = None;
                    }
                }
                None => {
                    best_split = Some(BestSplit::new(
                        score,
                        col.to_string(),
                        idx,
                        value,
                        target_groups,
                        None,
                    ));
                }
            }
        }
    }

    best_split.unwrap()
}

// Inspirations:
// * https://rusty-ferris.pages.dev/blog/binary-tree-sum-of-values/
// * https://gist.github.com/aidanhs/5ac9088ca0f6bdd4a370
pub fn grow_tree(
    x: &DataFrame,
    y: &Series,
    growth_params: &TreeGrowthParameters,
    parent_node: Option<&Node>,
    depth: usize,
) -> Node {
    let n_obs = x.height();
    if n_obs == 0 {
        panic!("Something went wrong. The parent_node handed down an empty set of data points.")
    }

    let (is_baselevel, reason) = check_is_baselevel(y, depth, growth_params);

    let (leaf_weight, score) = calc_leaf_weight_and_split_score(y, growth_params, None, None, None);

    if is_baselevel {
        let new_node = Node::new(
            None,
            None,
            None,
            Some(leaf_weight),
            None,
            None,
            None,
            Some(score),
            n_obs,
            reason,
            depth,
        );
        return new_node;
    }

    // find best split
    let best = find_best_split(x, y, growth_params, None, None, None);
    // let mut rng = ChaCha20Rng::seed_from_u64(42);

    let mut new_node = Node::new(
        Some(best.column),
        Some(best.column_idx),
        Some(best.threshold),
        Some(leaf_weight),
        match best.default_is_left {
            Some(default_is_left) => Some(default_is_left),
            None => None,
        },
        None,
        None,
        Some(SplitScore::new("neg_entropy".to_string(), best.score)),
        n_obs,
        "leaf node".to_string(),
        depth,
    );

    // check if improvement due to split is below minimum requirement

    // descend left
    let new_left_node = grow_tree(x, y, growth_params, Some(&new_node), &depth + 1); // mut new_node,
    new_node.insert(new_left_node, true);

    // descend right
    let new_right_node = grow_tree(x, y, growth_params, Some(&new_node), depth + 1); // mut new_node,
    new_node.insert(new_right_node, false);

    return new_node;
}

pub fn predict_for_row_with_tree(row: &Series, tree: &Node) -> f64 {
    let mut node = tree;

    let row_f64 = (*row).cast(&DataType::Float64).unwrap();
    let row = row_f64.f64().unwrap();

    while !node.is_leaf() {
        let idx = node.column_idx.unwrap();
        let value: f64 = row.get(idx).expect("Accessing failed.");

        let threshold = node.threshold.unwrap();
        let is_left = if value < threshold {
            node.default_is_left.unwrap()
        } else {
            !node.default_is_left.unwrap()
        };
        if is_left {
            node = node.left.as_ref().unwrap();
        } else {
            node = node.right.as_ref().unwrap();
        }
    }
    node.prediction.unwrap()
}

pub fn udf<'a, 'b>(
    s: Series,
    n_cols: &'a usize,
    tree: &'b Node,
) -> Result<Option<Series>, PolarsError> {
    let mut preds: Vec<f64> = vec![];

    for struct_ in s.iter() {
        let mut row: Vec<f64> = vec![];
        let mut iter = struct_._iter_struct_av();
        for _ in 0..*n_cols {
            let value = iter.next().unwrap().try_extract::<f64>().unwrap();
            row.push(value);
        }
        let row = Series::new("", row);
        let prediction = predict_for_row_with_tree(&row, tree);
        preds.push(prediction);
    }

    Ok(Some(Series::new("predictions", preds)))
}

pub fn predict_with_tree(x: DataFrame, tree: Node) -> Series {
    // use polars to apply predict_for_row_with_tree to get one prediction per row

    let mut columns: Vec<Expr> = vec![];
    let column_names = x.get_column_names();
    for v in column_names {
        columns.push(col(v));
    }
    let n_cols: usize = columns.len();

    let predictions = x
        .lazy()
        .select([as_struct(columns)
            .apply(
                move |s| udf(s, &n_cols, &tree),
                GetOutput::from_type(DataType::Float64),
            )
            .alias("predictions")])
        .collect()
        .unwrap();

    predictions.select_series(&["predictions"]).unwrap()[0].clone()
}

pub struct DecisionTreeCore {
    pub growth_params: TreeGrowthParameters,
    tree: Option<Node>,
}

impl DecisionTreeCore {
    pub fn new(max_depth: usize) -> Self {
        let growth_params = TreeGrowthParameters {
            max_depth: Some(max_depth),
        };
        DecisionTreeCore {
            growth_params,
            tree: None,
        }
    }

    pub fn fit(&mut self, x: &DataFrame, y: &Series) {
        self.tree = Some(grow_tree(x, y, &self.growth_params, None, 0));
    }

    pub fn predict(&self, x: &DataFrame) -> Series {
        let x = x.clone();
        let tree_ = self.tree.clone();
        match tree_ {
            Some(tree) => predict_with_tree(x, tree),
            None => panic!("Something went wrong. The tree is not initialized."),
        }
    }
}

pub struct DecisionTreeClassifier {
    decision_tree_core: DecisionTreeCore,
}

impl DecisionTreeClassifier {
    pub fn new(max_depth: usize) -> Self {
        DecisionTreeClassifier {
            decision_tree_core: DecisionTreeCore::new(max_depth),
        }
    }

    pub fn fit(&mut self, x: &DataFrame, y: &Series) {
        self.decision_tree_core.fit(x, y);
    }

    pub fn predict_proba(&self, x: &DataFrame) -> DataFrame {
        println!("predict_proba for {:?}", x.shape());
        let class1 = self.decision_tree_core.predict(x);
        println!("class1 {:?}", class1.len());
        let y_proba: DataFrame = df!("class_1" => &class1)
            .unwrap()
            .lazy()
            .with_columns([(lit(1.) - col("class_1")).alias("class_0")])
            .collect()
            .unwrap();
        let y_proba = y_proba.select(&["class_0", "class_1"]).unwrap();
        y_proba
    }

    pub fn predict(&self, x: &DataFrame) -> Series {
        let y_proba = self.predict_proba(x);
        // define "y" as a Series that contains the index of the maximum value column per row
        let y = y_proba
            .lazy()
            .select([(col("class_1").gt(0.5)).alias("y")])
            .collect()
            .unwrap();

        y.select_series(&["y"]).unwrap()[0].clone()
    }
}

#[cfg(test)]
mod tests {
    // use rand_chacha::ChaCha20Rng;
    // use rand::SeedableRng;

    use super::*;

    #[test]
    fn test_split_score() {
        let split_score = SplitScore::new("test".to_string(), 0.5);
        assert_eq!(split_score.name, "test");
        assert_eq!(split_score.score, 0.5);
    }

    #[test]
    fn test_node_init() {
        let node = Node::new(
            Some("column".to_string()),
            Some(0),
            Some(0.0),
            Some(1.0),
            Some(true),
            None,
            None,
            Some(SplitScore::new("score".to_string(), 0.5)),
            10,
            "leaf node".to_string(),
            0,
        );
        assert_eq!(node.column.unwrap(), "column".to_string());
        assert_eq!(node.column_idx.unwrap(), 0);
        assert_eq!(node.threshold.unwrap(), 0.0);
        assert_eq!(node.prediction.unwrap(), 1.0);
        assert_eq!(node.default_is_left.unwrap(), true);
        assert_eq!(node.left, None);
        assert_eq!(node.right, None);
        let m = node.measure.unwrap();
        assert_eq!(m.name, "score");
        assert_eq!(m.score, 0.5);
        assert_eq!(node.n_obs, 10);
        assert_eq!(node.reason, "leaf node".to_string());
        assert_eq!(node.depth, 0);
    }

    #[test]
    fn test_child_node_assignment() {
        let mut node = Node::new(
            Some("column".to_string()),
            Some(0),
            Some(0.0),
            Some(1.0),
            Some(true),
            None,
            None,
            Some(SplitScore::new("score".to_string(), 0.5)),
            10,
            "leaf node".to_string(),
            0,
        );
        let child_node = Node::new(
            Some("column".to_string()),
            Some(0),
            Some(0.0),
            Some(1.0),
            Some(true),
            None,
            None,
            Some(SplitScore::new("score".to_string(), 0.5)),
            10,
            "leaf node".to_string(),
            0,
        );
        node.left = Some(Box::new(child_node));
        assert_eq!(node.left.is_some(), true);
        assert_eq!(node.right.is_none(), true);
    }

    #[test]
    fn test_grandchild_node_assignment() {
        let mut node = Node::new(
            Some("column".to_string()),
            Some(0),
            Some(0.0),
            Some(1.0),
            Some(true),
            None,
            None,
            Some(SplitScore::new("score".to_string(), 0.5)),
            10,
            "leaf node".to_string(),
            0,
        );
        let child_node = Node::new(
            Some("column".to_string()),
            Some(0),
            Some(0.0),
            Some(1.0),
            Some(true),
            None,
            None,
            Some(SplitScore::new("score".to_string(), 0.5)),
            10,
            "leaf node".to_string(),
            0,
        );
        let grandchild_node = Node::new(
            Some("column".to_string()),
            Some(0),
            Some(0.0),
            Some(1.0),
            Some(true),
            None,
            None,
            Some(SplitScore::new("score".to_string(), 0.5)),
            10,
            "leaf node".to_string(),
            0,
        );
        node.left = Some(Box::new(child_node));
        node.left.as_mut().unwrap().left = Some(Box::new(grandchild_node));
        assert_eq!(node.left.is_some(), true);
        assert_eq!(node.right.is_none(), true);
        assert_eq!(node.left.as_ref().unwrap().left.is_some(), true);
        assert_eq!(node.left.as_ref().unwrap().right.is_none(), true);
    }

    #[test]
    fn test_node_is_leaf() {
        let node = Node {
            column: Some("column".to_string()),
            column_idx: Some(0),
            threshold: Some(0.0),
            prediction: Some(1.0),
            default_is_left: Some(true),
            left: None,
            right: None,
            measure: Some(SplitScore::new("score".to_string(), 0.5)),
            n_obs: 10,
            reason: "leaf node".to_string(),
            depth: 1,
            node_id: Uuid::new_v4(),
        };
        assert_eq!(node.is_leaf(), true);
    }

    // test calc_leaf_weight_and_split_score
    #[test]
    fn test_calc_leaf_weight_and_split_score() {
        let y = Series::new("y", &[1, 1, 1]);
        let growth_params = TreeGrowthParameters { max_depth: Some(2) };
        let (leaf_weight, score) =
            calc_leaf_weight_and_split_score(&y, &growth_params, None, None, None);
        assert_eq!(leaf_weight, 1.0);
        assert_eq!(score.name, "neg_entropy");
        assert_eq!(score.score, 0.0);
    }

    #[test]
    fn test_grow_tree() {
        let df = DataFrame::new(vec![
            Series::new("a", &[1, 2, 3]),
            Series::new("b", &[1, 2, 3]),
            Series::new("c", &[1, 2, 3]),
        ])
        .unwrap();
        let y = Series::new("y", &[1, 1, 2]);
        let growth_params = TreeGrowthParameters { max_depth: Some(1) };

        let tree = grow_tree(&df, &y, &growth_params, None, 0);

        assert!(tree.is_leaf() == false);
        assert_eq!(tree.left.is_some(), true);
        assert_eq!(tree.right.is_some(), true);
        assert_eq!(tree.left.as_ref().unwrap().is_leaf(), true);
        assert_eq!(tree.right.as_ref().unwrap().is_leaf(), true);
    }

    #[test]
    fn test_predict_for_row_with_tree() {
        let df = DataFrame::new(vec![
            Series::new("a", &[1, 2, 3]),
            Series::new("b", &[1, 2, 3]),
            Series::new("c", &[1, 2, 3]),
        ])
        .unwrap();
        let y = Series::new("y", &[1, 1, 1]);
        let growth_params = TreeGrowthParameters { max_depth: Some(2) };

        let tree = grow_tree(&df, &y, &growth_params, None, 0);

        let row = df.select_at_idx(0).unwrap();
        let prediction = predict_for_row_with_tree(&row, &tree);
        assert_eq!(prediction, 1.0);
    }

    #[test]
    fn test_predict_with_tree() {
        let df = DataFrame::new(vec![
            Series::new("a", &[1, 2, 3, 4]),
            Series::new("b", &[1, 2, 3, 4]),
            Series::new("c", &[1, 2, 3, 4]),
        ])
        .unwrap();
        let y = Series::new("y", &[1, 1, 1, 1]);
        let growth_params = TreeGrowthParameters { max_depth: Some(2) };
        let tree = grow_tree(&df, &y, &growth_params, None, 0);

        let predictions = predict_with_tree(df, tree);
        assert_eq!(
            predictions,
            Series::new("predictions", &[1.0, 1.0, 1.0, 1.0])
        );
    }

    #[test]
    fn test_decision_tree_core() {
        let df = DataFrame::new(vec![
            Series::new("a", &[1, 2, 3]),
            Series::new("b", &[1, 2, 3]),
            Series::new("c", &[1, 2, 3]),
        ])
        .unwrap();
        let y = Series::new("y", &[1, 1, 1]);

        let mut dtree = DecisionTreeCore::new(2);
        dtree.fit(&df, &y);
        let predictions = dtree.predict(&df);
        assert_eq!(predictions, Series::new("predictions", &[1.0, 1.0, 1.0]));
    }

    #[test]
    fn test_decision_tree_classifier() {
        let df = DataFrame::new(vec![
            Series::new("a", &[1, 2, 3]),
            Series::new("b", &[1, 2, 3]),
            Series::new("c", &[1, 2, 3]),
        ])
        .unwrap();
        let y = Series::new("y", &[1, 1, 1]);

        let mut dtree = DecisionTreeClassifier::new(2);
        dtree.fit(&df, &y);
        let predictions = dtree.predict(&df);
        assert_eq!(predictions, Series::new("y", &[1, 1, 1]));

        let y_proba = dtree.predict_proba(&df);
        assert_eq!(y_proba.shape(), (3, 2));
        assert_eq!(y_proba.get_column_names(), &["class_0", "class_1"]);
        // assert that y_proba sums to 1 per row
        let y_proba_sum = y_proba
            .sum_horizontal(polars::frame::NullStrategy::Propagate)
            .unwrap()
            .unwrap();
        assert_eq!(y_proba_sum, Series::new("class_0", &[1.0, 1.0, 1.0]));
    }
}
