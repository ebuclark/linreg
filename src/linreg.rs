use ndarray::{Array1, Array2};

pub struct LinearRegression {
    pub lr: f64,
    pub weights: Option<Array1<f64>>, // None before fit - Some after fit
    pub n_iters: i32,
}

impl LinearRegression {
    pub fn new(lr: f64, n_iters: i32) -> Self {
        Self {
            lr,
            weights: None,
            n_iters,
        }
    }

    pub fn fit(&mut self, X: &Array2<f64>, y: &Array1<f64>){
        // Given X and Y, fit a linear regression model
        
        // Get shape of X matrix
        let (n,p) = X.dim();

        self.weights = Some(Array1::zeros(p));

        // Extract weights as 'w' from Option
        // let's work with a local mutable copy
        let mut w = self.weights.take().unwrap();

        // Gradient Descent Loop
        for _ in 0..self.n_iters {
            // Generate y prediction
            let y_hat = X.dot(&w);

            // Compute gradient
            let residuals = &y_hat - y;
            let dw = (2.0 / n as f64) *X.t().dot(&residuals);

            w = w - self.lr*dw;
        }
        self.weights = Some(w);
    }

    pub fn predict(&mut self, X: &Array2<f64>) -> Array1<f64> {
        let w = self.weights.as_ref().expect(
            "Model has not been trained. Call .fit(X,y) first."
        );

        X.dot(w)
    }

    pub fn mse(&mut self, X: &Array2<f64>, y: &Array1<f64>) -> f64 {
        let y_hat = self.predict(X);
        let residuals = y - &y_hat;

        residuals.mapv(|err| err.powi(2)).mean().unwrap_or(0.0)
    }
}
