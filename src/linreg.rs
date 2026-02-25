use ndarray::{Array1, Array2};

pub struct LinearRegression {
    pub lr: f64,
    pub weights: Option<Array1<f64>>, // None before fit - Some after fit
    pub n_iters: i32,
}

impl LinearRegression {
    // Static Method - e.g. constructor
    pub fn new(lr: f64, n_iters: i32) -> Self {
        // let weights = Array1::zeros()
        Self {
            // attributes
            lr,
            weights: None,
            n_iters,
        }
    }

    pub fn fit(&mut self, X: Array2<f64>, y: Array1<f64>){
        // Given X and Y, fit a linear regression model
        
        // Get shape of X matrix
        let (n,p) = X.dim();

        self.weights = Some(Array1::zeros(p));

        // Extract weights as 'w' from Option
        // let's work with a local mutable copy
        let mut w = self.weights.take().unwrap();

        // Gradient Descent Loop
        for _ in 0..self.n_iters {
            // generate y prediction
            let y_pred = X.dot(&w);

            // compute gradient
            let residuals = &y_pred - &y;
            let dw = (2.0 / n as f64) *X.t().dot(&residuals);

            w = w - self.lr*dw;
        }
        self.weights = Some(w);
    }

    pub fn predict(&mut self, X: Array2<f64>) -> Array1<f64> {
        let w = self.weights.as_ref().expect(
            "Model has not been trained. Call .fit(X,y) first."
        );
        
        X.dot(w)
    }
}
