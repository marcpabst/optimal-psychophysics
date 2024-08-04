pub use bernoulli_logit::BernoulliLogit;
use num_traits::Float;
use rand::Rng;

mod bernoulli_logit;

// re-export distributions from rand_distr
pub use rand_distr::Normal;

/// Marker trait for continuous distributions.
pub trait ContinuousDistribution {}
/// Marker trait for discrete distributions.
pub trait DiscreteDistribution {}
/// Marker trait for univariate distributions.
pub trait UnivariateDistribution {}

/// Samplable trait.
pub trait Samplable<S> {
    fn sample<R>(&self, rng: &mut R) -> S
    where
        R: rand::Rng + ?Sized;
}

#[allow(unused)]
pub trait ContinuousUnivariateDistribution<F: Float, S: Float, const P: usize>:
    ContinuousDistribution + UnivariateDistribution + Samplable<S>
{
    /// Return the log probability density of the distribution at `x` given the parameters.
    fn logpdf(params: &[F; P], x: S) -> F;

    /// Return the log probability of the distribution at `x`.
    fn logp(&self, x: S) -> F {
        Self::logpdf(&self.params(), x)
    }

    /// Returns the support of the distribution.
    fn support(&self) -> (F, F);

    /// Returns the parameters of the distribution.
    fn params(&self) -> [F; P];

    /// Returns the location parameter of the distribution (if meaningful).
    fn loc(&self) -> Option<F>;

    /// Returns the scale parameter of the distribution (if meaningful).
    fn scale(&self) -> Option<F>;
}

#[allow(unused)]
pub trait DiscreteUnivariateDistribution<F: Float, S, const P: usize>:
    DiscreteDistribution + UnivariateDistribution + Samplable<S>
{
    /// Return the log probability density of the distribution at `x` given the parameters.
    fn logpmf(params: &[F; P], x: S) -> F;

    /// Return the log probability of the distribution at `x`.
    fn logp(&self, x: S) -> F {
        Self::logpmf(&self.params(), x)
    }

    /// Returns the support of the distribution.
    fn support(&self) -> (F, F);

    /// Returns the parameters of the distribution.
    fn params(&self) -> [F; P];

    /// Returns the location parameter of the distribution (if meaningful).
    fn loc(&self) -> Option<F>;

    /// Returns the scale parameter of the distribution (if meaningful).
    fn scale(&self) -> Option<F>;
}

// The Normal distribution

impl ContinuousDistribution for rand_distr::Normal<f64> {}
impl UnivariateDistribution for rand_distr::Normal<f64> {}

impl ContinuousUnivariateDistribution<f64, f64, 2> for rand_distr::Normal<f64> {
    fn logpdf(params: &[f64; 2], x: f64) -> f64 {
        let ln_2_pi_sqrt = (2.0 * std::f64::consts::PI).sqrt().ln();
        -0.5 * ((x - params[0]) / params[1]).powi(2) - ln_2_pi_sqrt - params[1].ln()
    }

    fn support(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }

    fn params(&self) -> [f64; 2] {
        [self.mean(), self.std_dev()]
    }

    fn loc(&self) -> Option<f64> {
        Some(self.mean())
    }

    fn scale(&self) -> Option<f64> {
        Some(self.std_dev())
    }
}

impl Samplable<f64> for rand_distr::Normal<f64> {
    fn sample<R>(&self, rng: &mut R) -> f64
    where
        R: Rng + ?Sized,
    {
        rand_distr::Distribution::<f64>::sample(self, rng)
    }
}

// The Bernoulli distribution

impl DiscreteDistribution for rand_distr::Bernoulli {}
impl UnivariateDistribution for rand_distr::Bernoulli {}

impl DiscreteUnivariateDistribution<f64, bool, 1> for rand_distr::Bernoulli {
    #[inline]
    fn logpmf(params: &[f64; 1], x: bool) -> f64 {
        let p = params[0];
        if x {
            p.ln()
        } else {
            (1.0 - p).ln()
        }
    }

    fn support(&self) -> (f64, f64) {
        (0.0, 1.0)
    }

    fn params(&self) -> [f64; 1] {
        todo!()
    }

    fn loc(&self) -> Option<f64> {
        None
    }

    fn scale(&self) -> Option<f64> {
        None
    }
}

impl Samplable<bool> for rand_distr::Bernoulli {
    fn sample<R>(&self, rng: &mut R) -> bool
    where
        R: Rng + ?Sized,
    {
        rand_distr::Distribution::<bool>::sample(self, rng)
    }
}

impl DiscreteDistribution for BernoulliLogit {}
impl UnivariateDistribution for BernoulliLogit {}

impl DiscreteUnivariateDistribution<f64, bool, 1> for BernoulliLogit {
    #[inline]
    fn logpmf(params: &[f64; 1], x: bool) -> f64 {
        let p_logit = params[0];

        // np.log(
        // (np.exp(p_logit) / (np.exp(p_logit) + 1))
        // **(1 - y) * (1 / (np.exp(p_logit) + 1))**y)
        let y = if x { 1.0 } else { 0.0 };

        let p_logit_exp = p_logit.exp();
        ((p_logit_exp / (p_logit_exp + 1.0)).powf(1.0 - y) * (1.0 / (p_logit_exp + 1.0)).powf(y))
            .ln()
    }

    fn support(&self) -> (f64, f64) {
        (0.0, 1.0)
    }

    fn params(&self) -> [f64; 1] {
        [self.logit_p()]
    }

    fn loc(&self) -> Option<f64> {
        None
    }

    fn scale(&self) -> Option<f64> {
        None
    }
}

impl Samplable<bool> for BernoulliLogit {
    #[inline]
    fn sample<R>(&self, rng: &mut R) -> bool
    where
        R: Rng + ?Sized,
    {
        let logit_p = self.logit_p();
        let p = logit_p.exp() / (1.0 + logit_p.exp());
        let u = rng.gen::<f64>();
        let logit_p = p.ln() - (-p).ln_1p();
        u < logit_p.exp() / (1.0 + logit_p.exp())
    }
}
