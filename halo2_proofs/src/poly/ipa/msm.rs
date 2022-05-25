// use super::{ParamsIPA, IPACommitmentScheme};
use crate::{
    arithmetic::{best_multiexp, parallelize, CurveAffine},
    poly::{
        commitment::{CommitmentScheme, Params, MSM},
        ipa::commitment::ParamsVerifierIPA,
    },
};
use ff::Field;
use group::Group;

use super::commitment::{IPACommitmentScheme, ParamsIPA};

// /// A multiscalar multiplication in the polynomial commitment scheme
// #[derive(Debug, Clone)]
// pub struct MSMIPA<C: CurveAffine> {
//     scalars: Vec<C::Scalar>,
//     bases: Vec<C::Curve>,
// }

// impl<'a, C: CurveAffine> MSMIPA<C> {
//     pub fn new() -> Self {
//         Self {
//             scalars: vec![],
//             bases: vec![],
//         }
//     }

//     /// Add another multiexp into this one
//     pub fn add_msm(&mut self, other: &dyn MSM<C>) {
//         self.scalars.extend(other.scalars().iter());
//         self.bases.extend(other.bases().iter());
//     }
// }

/// A multiscalar multiplication in the polynomial commitment scheme
#[derive(Debug, Clone)]
pub struct MSMIPA<'params, C: CurveAffine> {
    pub(crate) params: &'params ParamsVerifierIPA<C>,
    g_scalars: Option<Vec<C::Scalar>>,
    w_scalar: Option<C::Scalar>,
    u_scalar: Option<C::Scalar>,
    other_scalars: Vec<C::Scalar>,
    other_bases: Vec<C::Curve>,
}

impl<'a, C: CurveAffine> MSMIPA<'a, C> {
    /// Given verifier parameters Creates an empty multi scalar engine
    pub fn new(params: &'a ParamsVerifierIPA<C>) -> Self {
        let g_scalars = None;
        let w_scalar = None;
        let u_scalar = None;
        let other_scalars = vec![];
        let other_bases = vec![];

        Self {
            g_scalars,
            w_scalar,
            u_scalar,
            other_scalars,
            other_bases,

            params,
        }
    }

    /// Add another multiexp into this one
    pub fn add_msm(&mut self, other: &Self) {
        self.other_scalars.extend(other.other_scalars.iter());
        self.other_bases.extend(other.other_bases.iter());

        if let Some(g_scalars) = &other.g_scalars {
            self.add_to_g_scalars(g_scalars);
        }

        if let Some(w_scalar) = &other.w_scalar {
            self.add_to_w_scalar(*w_scalar);
        }

        if let Some(u_scalar) = &other.u_scalar {
            self.add_to_u_scalar(*u_scalar);
        }
    }
}

impl<'a, C: CurveAffine> MSM<C> for MSMIPA<'a, C> {
    fn append_term(&mut self, scalar: C::Scalar, point: C::Curve) {
        self.other_scalars.push(scalar);
        self.other_bases.push(point);
    }

    fn add_msm(&mut self, other: &dyn MSM<C>) {
        self.other_scalars.extend(other.scalars().iter());
        self.other_bases.extend(other.bases().iter());
    }

    fn scale(&mut self, factor: C::Scalar) {
        if let Some(g_scalars) = &mut self.g_scalars {
            parallelize(g_scalars, |g_scalars, _| {
                for g_scalar in g_scalars {
                    *g_scalar *= &factor;
                }
            })
        }

        if !self.other_scalars.is_empty() {
            parallelize(&mut self.other_scalars, |other_scalars, _| {
                for other_scalar in other_scalars {
                    *other_scalar *= &factor;
                }
            })
        }

        self.w_scalar = self.w_scalar.map(|a| a * &factor);
        self.u_scalar = self.u_scalar.map(|a| a * &factor);
    }

    fn check(&self) -> bool {
        bool::from(self.eval().is_identity())
    }

    fn eval(&self) -> C::Curve {
        let len = self.g_scalars.as_ref().map(|v| v.len()).unwrap_or(0)
            + self.w_scalar.map(|_| 1).unwrap_or(0)
            + self.u_scalar.map(|_| 1).unwrap_or(0)
            + self.other_scalars.len();
        let mut scalars: Vec<C::Scalar> = Vec::with_capacity(len);
        let mut bases: Vec<C::Curve> = Vec::with_capacity(len);

        scalars.extend(&self.other_scalars);
        bases.extend(&self.other_bases);

        if let Some(w_scalar) = self.w_scalar {
            scalars.push(w_scalar);
            bases.push(self.params.w.into());
        }

        if let Some(u_scalar) = self.u_scalar {
            scalars.push(u_scalar);
            bases.push(self.params.u.into());
        }

        if let Some(g_scalars) = &self.g_scalars {
            scalars.extend(g_scalars);
            let g: Vec<C::CurveExt> = self.params.g.iter().map(|g| (*g).into()).collect();
            bases.extend(g);
        }

        assert_eq!(scalars.len(), len);

        use group::Curve;
        let mut normalized = vec![C::identity(); scalars.len()];
        C::Curve::batch_normalize(&bases[..], &mut normalized);
        best_multiexp(&scalars, &normalized[..])
    }

    fn bases(&self) -> Vec<C::CurveExt> {
        self.other_bases.clone()
    }

    fn scalars(&self) -> Vec<C::Scalar> {
        self.other_scalars.clone()
    }
}

impl<'a, C: CurveAffine> MSMIPA<'a, C> {
    /// Add a value to the first entry of `g_scalars`.
    pub fn add_constant_term(&mut self, constant: C::Scalar) {
        if let Some(g_scalars) = self.g_scalars.as_mut() {
            g_scalars[0] += &constant;
        } else {
            let mut g_scalars = vec![C::Scalar::zero(); self.params.n as usize];
            g_scalars[0] += &constant;
            self.g_scalars = Some(g_scalars);
        }
    }

    /// Add a vector of scalars to `g_scalars`. This function will panic if the
    /// caller provides a slice of scalars that is not of length `params.n`.
    pub fn add_to_g_scalars(&mut self, scalars: &[C::Scalar]) {
        assert_eq!(scalars.len(), self.params.n as usize);
        if let Some(g_scalars) = &mut self.g_scalars {
            parallelize(g_scalars, |g_scalars, start| {
                for (g_scalar, scalar) in g_scalars.iter_mut().zip(scalars[start..].iter()) {
                    *g_scalar += scalar;
                }
            })
        } else {
            self.g_scalars = Some(scalars.to_vec());
        }
    }

    /// Add to `w_scalar`
    pub fn add_to_w_scalar(&mut self, scalar: C::Scalar) {
        self.w_scalar = self.w_scalar.map_or(Some(scalar), |a| Some(a + &scalar));
    }

    /// Add to `u_scalar`
    pub fn add_to_u_scalar(&mut self, scalar: C::Scalar) {
        self.u_scalar = self.u_scalar.map_or(Some(scalar), |a| Some(a + &scalar));
    }
}
