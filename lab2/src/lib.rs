use burn::data::dataset::InMemDataset;

use crate::data::RegressionItem;

pub mod config;
pub mod data;
pub mod models;

#[inline]
pub fn goal_fn(x: f32, y: f32) -> f32 {
    x.sin() + y
}

pub fn generate_data(x_start: f32, x_end: f32, num_points: usize) -> InMemDataset<RegressionItem> {
    let mut items = Vec::with_capacity(num_points);

    let step = if num_points > 1 {
        (x_end - x_start) / (num_points - 1) as f32
    } else {
        0.0
    };

    let x_norm_factor = models::TEST_END;
    let z_norm_factor = 2.0;

    for i in 0..num_points {
        let x_raw = x_start + step * (i as f32);
        let y_raw = (x_raw * 0.5).cos();

        let z_raw = goal_fn(x_raw, y_raw);

        items.push(RegressionItem {
            input: [x_raw / x_norm_factor, y_raw],
            target: [z_raw / z_norm_factor],
        });
    }

    InMemDataset::new(items)
}
fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}
