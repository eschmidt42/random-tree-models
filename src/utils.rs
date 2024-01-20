use std::fmt;

#[derive(Debug)]
pub enum SplitScoreMetrics {
    NegVariance,
    NegEntropy,
}

impl fmt::Display for SplitScoreMetrics {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
        // or, alternatively:
        // fmt::Debug::fmt(self, f)
    }
}

pub struct TreeGrowthParameters {
    pub max_depth: Option<usize>,
    pub split_score_metric: Option<SplitScoreMetrics>,
}
