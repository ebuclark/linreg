#![allow(non_snake_case)]
// Use the linreg module that we've built
mod linreg;

// Linear Algebra goodies
use linreg::LinearRegression;
use ndarray::{array, Array, Array1, Array2};

use ndarray_rand::RandomExt;
// Randomness
use rand::{SeedableRng, rngs::StdRng};
// Distributions
use ndarray_rand::rand_distr::{Uniform, Normal};

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    let n: usize = 100;
    let p: usize = 2;
    
    // Simulate random data
    let mut X: Array2<f64> = Array::random_using(
        (n,p+1), 
        Uniform::new(0.0, 1.0).unwrap(), 
        &mut rng
    ) * 2.0;
    // Fill bias column with 1
    X.column_mut(0).fill(1.0);

    // Epsilon noise generated from a normal distribution
    let e: Array1<f64> = Array::random_using(
        n, 
        Normal::new(0.0,1.0).unwrap(), 
        &mut rng
    );

    // Generate y from some ground truth data
    let true_weights = array![4.0, 3.0, 5.0];
    let y: Array1<f64> = X.dot(&true_weights) + &e;

    // Construct and fit out linear regressor
    let mut regressor = LinearRegression::new(0.01, 1000);
    regressor.fit(&X,&y);

    // Print weights and MSE
    println!("Regressor weights {:#?}", regressor.weights);

    println!("MSE: {:.3}", regressor.mse(&X, &y));
}
