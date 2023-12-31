use polars::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use uuid::Uuid;

#[derive(PartialEq, Debug)]
pub struct SplitScore {
    pub name: String,
    pub score: f64,
}

impl SplitScore {
    pub fn new(name: String, score: f64) -> Self {
        SplitScore { name, score }
    }
}

#[derive(PartialEq, Debug)]
pub struct Node {
    pub array_column: usize,
    pub threshold: f64,
    pub prediction: f64,
    pub default_is_left: bool,

    // descendants
    pub left: Option<Box<Node>>,
    pub right: Option<Box<Node>>,

    // misc
    pub measure: SplitScore,

    pub n_obs: usize,
    pub reason: String,
    pub depth: usize,
    pub node_id: Uuid,
}

impl Node {
    pub fn new(
        array_column: usize,
        threshold: f64,
        prediction: f64,
        default_is_left: bool,
        left: Option<Box<Node>>,
        right: Option<Box<Node>>,
        measure: SplitScore,
        n_obs: usize,
        reason: String,
        depth: usize,
    ) -> Self {
        let node_id = Uuid::new_v4();
        Node {
            array_column,
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
// Inspirations:
// * https://rusty-ferris.pages.dev/blog/binary-tree-sum-of-values/
// * https://gist.github.com/aidanhs/5ac9088ca0f6bdd4a370
pub fn grow_tree(x: &DataFrame, y: &Series, parent_node: Option<&Node>, depth: usize) -> Node {
    // TODO: implement check_is_baselevel and such
    let n_obs = x.height();
    if n_obs == 0 {
        panic!("Something went wrong. The parent_node handed down an empty set of data points.")
    }

    let is_baselevel: bool = depth == 1;
    if is_baselevel {
        let new_node = Node::new(
            0,
            0.0,
            1.0,
            true,
            None,
            None,
            SplitScore::new("score".to_string(), 0.5),
            10,
            "leaf node".to_string(),
            0,
        );
        return new_node;
    }

    let mut rng = ChaCha20Rng::seed_from_u64(42);

    let mut new_node = Node::new(
        0,
        0.0,
        1.0,
        true,
        None,
        None,
        SplitScore::new("score".to_string(), 0.5),
        10,
        "leaf node".to_string(),
        0,
    );

    // descend left
    let new_left_node = grow_tree(x, y, Some(&new_node), &depth + 1); // mut new_node,
    new_node.insert(new_left_node, true);

    // descend right
    let new_right_node = grow_tree(x, y, Some(&new_node), depth + 1); // mut new_node,
    new_node.insert(new_right_node, false);

    return new_node;
}

pub fn predict_for_row_with_tree(row: &Series, tree: &Node) -> f64 {
    let mut node = tree;

    let row_f64 = (*row).cast(&DataType::Float64).unwrap();
    let row = row_f64.f64().unwrap();

    while !node.is_leaf() {
        let value: f64 = row.get(node.array_column).expect("Accessing failed.");

        let is_left = if value < node.threshold {
            node.default_is_left
        } else {
            !node.default_is_left
        };
        if is_left {
            node = node.left.as_ref().unwrap();
        } else {
            node = node.right.as_ref().unwrap();
        }
    }
    node.prediction
}

pub fn predict_with_tree(x: &DataFrame, tree: &Node) -> Series {
    // use polars to apply predict_for_row_with_tree to get one prediction per row
    let predictions: Series = x
        .iter()
        .map(|row| predict_for_row_with_tree(row, tree))
        .collect();

    let predictions = Series::new("predictions", predictions);

    predictions
}

struct DecisionTreeTemplate {
    pub max_depth: usize,
    tree: Option<Node>,
}

impl DecisionTreeTemplate {
    pub fn new(max_depth: usize) -> Self {
        DecisionTreeTemplate {
            max_depth,
            tree: None,
        }
    }

    pub fn fit(&mut self, x: &DataFrame, y: &Series) {
        self.tree = Some(grow_tree(x, y, None, 0));
    }

    pub fn predict(&self, x: &DataFrame) -> Series {
        match &self.tree {
            Some(tree) => predict_with_tree(x, tree),
            None => panic!("Something went wrong. The tree is not initialized."),
        }
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
            0,
            0.0,
            1.0,
            true,
            None,
            None,
            SplitScore::new("score".to_string(), 0.5),
            10,
            "leaf node".to_string(),
            0,
        );
        assert_eq!(node.array_column, 0);
        assert_eq!(node.threshold, 0.0);
        assert_eq!(node.prediction, 1.0);
        assert_eq!(node.default_is_left, true);
        assert_eq!(node.left, None);
        assert_eq!(node.right, None);
        assert_eq!(node.measure.name, "score");
        assert_eq!(node.measure.score, 0.5);
        assert_eq!(node.n_obs, 10);
        assert_eq!(node.reason, "leaf node".to_string());
        assert_eq!(node.depth, 0);
    }

    #[test]
    fn test_child_node_assignment() {
        let mut node = Node::new(
            0,
            0.0,
            1.0,
            true,
            None,
            None,
            SplitScore::new("score".to_string(), 0.5),
            10,
            "leaf node".to_string(),
            0,
        );
        let child_node = Node::new(
            0,
            0.0,
            1.0,
            true,
            None,
            None,
            SplitScore::new("score".to_string(), 0.5),
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
            0,
            0.0,
            1.0,
            true,
            None,
            None,
            SplitScore::new("score".to_string(), 0.5),
            10,
            "leaf node".to_string(),
            0,
        );
        let child_node = Node::new(
            0,
            0.0,
            1.0,
            true,
            None,
            None,
            SplitScore::new("score".to_string(), 0.5),
            10,
            "leaf node".to_string(),
            0,
        );
        let grandchild_node = Node::new(
            0,
            0.0,
            1.0,
            true,
            None,
            None,
            SplitScore::new("score".to_string(), 0.5),
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
            array_column: 0,
            threshold: 0.0,
            prediction: 1.0,
            default_is_left: true,
            left: None,
            right: None,
            measure: SplitScore::new("score".to_string(), 0.5),
            n_obs: 10,
            reason: "leaf node".to_string(),
            depth: 1,
            node_id: Uuid::new_v4(),
        };
        assert_eq!(node.is_leaf(), true);
    }

    #[test]
    fn test_grow_tree() {
        let df = DataFrame::new(vec![
            Series::new("a", &[1, 2, 3]),
            Series::new("b", &[1, 2, 3]),
            Series::new("c", &[1, 2, 3]),
        ])
        .unwrap();
        let y = Series::new("y", &[1, 2, 3]);

        let tree = grow_tree(&df, &y, None, 0);

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
        let y = Series::new("y", &[1, 2, 3]);

        let tree = grow_tree(&df, &y, None, 0);

        let row = df.select_at_idx(0).unwrap();
        let prediction = predict_for_row_with_tree(&row, &tree);
        assert_eq!(prediction, 1.0);
    }

    // test predict_with_tree
    #[test]
    fn test_predict_with_tree() {
        let df = DataFrame::new(vec![
            Series::new("a", &[1, 2, 3]),
            Series::new("b", &[1, 2, 3]),
            Series::new("c", &[1, 2, 3]),
        ])
        .unwrap();
        let y = Series::new("y", &[1, 2, 3]);

        let tree = grow_tree(&df, &y, None, 0);

        let predictions = predict_with_tree(&df, &tree);
        assert_eq!(predictions, Series::new("predictions", &[1.0, 1.0, 1.0]));
    }

    // test DecisionTreeTemplate
    #[test]
    fn test_decision_tree_template() {
        let df = DataFrame::new(vec![
            Series::new("a", &[1, 2, 3]),
            Series::new("b", &[1, 2, 3]),
            Series::new("c", &[1, 2, 3]),
        ])
        .unwrap();
        let y = Series::new("y", &[1, 2, 3]);

        let mut dtree = DecisionTreeTemplate::new(2);
        dtree.fit(&df, &y);
        let predictions = dtree.predict(&df);
        assert_eq!(predictions, Series::new("predictions", &[1.0, 1.0, 1.0]));
    }
}
