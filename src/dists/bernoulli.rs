//! Benroulli logit distribution. Identical to the Bernoulli distribution, but with an internal logit transformation for numerical stability.

//! The Bernoulli distribution `Bernoulli(p)`.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BernoulliLogit(rand::distributions::Bernoulli);

impl BernoulliLogit {
    /// Construct a new `BernoulliLogit` distribution with the given probability `p`.
    ///
    /// # Panics
    ///
    /// If `p` is not in the interval `[0, 1]`.
    #[inline]
    pub fn new(logit_p: f64) -> Self {
        let p = 1.0 / (1.0 + (-logit_p).exp());
        Self(rand::distributions::Bernoulli::new(p).unwrap())
    }
}

impl core::ops::Deref for BernoulliLogit {
    type Target = rand::distributions::Bernoulli;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
