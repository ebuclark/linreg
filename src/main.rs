#![allow(non_snake_case)]

mod linreg;

use linreg::LinearRegression;
use ndarray::{array, Array, Array1, Array2};

use ndarray_rand::RandomExt;
// randomness
use rand::{SeedableRng, rngs::StdRng};
// distributions
use ndarray_rand::rand_distr::{Uniform, Normal};

fn main() {
    // Simulate random data
    let mut rng = StdRng::seed_from_u64(42);
    let n: usize = 100;
    let p: usize = 2;
    
    let mut X: Array2<f64> = Array::random_using(
        (n,p+1), 
        Uniform::new(0.0, 1.0).unwrap(), 
        &mut rng
    ) * 2.0;
    // fill bias column with 1
    X.column_mut(0).fill(1.0);

    // epsilon noise generated from a normal distribution
    let e: Array1<f64> = Array::random_using(
        n, 
        Normal::new(0.0,1.0).unwrap(), 
        &mut rng
    );

    let true_weights = array![4.0, 3.0, 5.0];
    let y: Array1<f64> = X.dot(&true_weights) + &e;

    let mut regressor = LinearRegression::new(0.01, 1000);
    regressor.fit(X,y);
    println!("Regressor weights {:#?}", regressor.weights);
}
