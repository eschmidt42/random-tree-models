pub enum SplitScoreMetrics {
    Variance,
    Entropy,
}

pub struct TreeGrowthParameters {
    pub max_depth: Option<usize>,
    pub split_score_metric: Option<SplitScoreMetrics>,
}
