
use super::error::{GPUError, GPUResult};

use ff::Field;
use ff::PrimeField;
use ff::PrimeFieldBits;
use group::{prime::PrimeCurveAffine, Group};
use group::prime::PrimeCurve;
use crate::arithmetic::Engine;


use bit_vec::{self, BitVec};
use log::{info, warn};
use rayon::prelude::*;
use std::io;
use std::iter;
use std::ops::AddAssign;
use std::sync::Arc;
use futures::SinkExt;

use crate::worker::{Waiter, Worker};
use crate::gpu;

/// An object that builds a source of bases.
pub trait SourceBuilder<G: PrimeCurveAffine>: Send + Sync + 'static + Clone {
    type Source: Source<G>;

    fn new(self) -> Self::Source;
    fn get(self) -> (Arc<Vec<G>>, usize);
}

/// A source of bases, like an iterator.
pub trait Source<G: PrimeCurveAffine> {
    /// Parses the element from the source. Fails if the point is at infinity.
    fn add_assign_mixed(
        &mut self,
        to: &mut <G as PrimeCurveAffine>::Curve,
    ) -> Result<(), GPUError>;

    /// Skips `amt` elements from the source, avoiding deserialization.
    fn skip(&mut self, amt: usize) -> Result<(), GPUError>;
}

impl<G: PrimeCurveAffine> SourceBuilder<G> for (Arc<Vec<G>>, usize) {
    type Source = (Arc<Vec<G>>, usize);

    fn new(self) -> (Arc<Vec<G>>, usize) {
        (self.0.clone(), self.1)
    }

    fn get(self) -> (Arc<Vec<G>>, usize) {
        (self.0.clone(), self.1)
    }
}

impl<G: PrimeCurveAffine> Source<G> for (Arc<Vec<G>>, usize) {
    fn add_assign_mixed(
        &mut self,
        to: &mut <G as PrimeCurveAffine>::Curve,
    ) -> Result<(), GPUError> {
        if self.0.len() <= self.1 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "expected more bases from source",
            )
                .into());
        }

        if self.0[self.1].is_zero() {
            return Err(GPUError::GPUDisabled);
        }

        to.add_assign_mixed(&self.0[self.1]);

        self.1 += 1;

        Ok(())
    }

    fn skip(&mut self, amt: usize) -> Result<(), GPUError> {
        if self.0.len() <= self.1 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "expected more bases from source",
            )
                .into());
        }

        self.1 += amt;

        Ok(())
    }
}

pub trait QueryDensity {
    /// Returns whether the base exists.
    type Iter: Iterator<Item = bool>;

    fn iter(self) -> Self::Iter;
    fn get_query_size(self) -> Option<usize>;
}

#[derive(Clone)]
pub struct FullDensity;

impl AsRef<FullDensity> for FullDensity {
    fn as_ref(&self) -> &FullDensity {
        self
    }
}

impl<'a> QueryDensity for &'a FullDensity {
    type Iter = iter::Repeat<bool>;

    fn iter(self) -> Self::Iter {
        iter::repeat(true)
    }

    fn get_query_size(self) -> Option<usize> {
        None
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DensityTracker {
    pub bv: BitVec,
    pub total_density: usize,
}

impl<'a> QueryDensity for &'a DensityTracker {
    type Iter = bit_vec::Iter<'a>;

    fn iter(self) -> Self::Iter {
        self.bv.iter()
    }

    fn get_query_size(self) -> Option<usize> {
        Some(self.bv.len())
    }
}

impl DensityTracker {
    pub fn new() -> DensityTracker {
        DensityTracker {
            bv: BitVec::new(),
            total_density: 0,
        }
    }

    pub fn add_element(&mut self) {
        self.bv.push(false);
    }

    pub fn inc(&mut self, idx: usize) {
        if !self.bv.get(idx).unwrap() {
            self.bv.set(idx, true);
            self.total_density += 1;
        }
    }

    pub fn get_total_density(&self) -> usize {
        self.total_density
    }

    /// Extend by concatenating `other`. If `is_input_density` is true, then we are tracking an input density,
    /// and other may contain a redundant input for the `One` element. Coalesce those as needed and track the result.
    pub fn extend(&mut self, other: Self, is_input_density: bool) {
        if other.bv.is_empty() {
            // Nothing to do if other is empty.
            return;
        }

        if self.bv.is_empty() {
            // If self is empty, assume other's density.
            self.total_density = other.total_density;
            self.bv = other.bv;
            return;
        }

        if is_input_density {
            // Input densities need special handling to coalesce their first inputs.

            if other.bv[0] {
                // If other's first bit is set,
                if self.bv[0] {
                    // And own first bit is set, then decrement total density so the final sum doesn't overcount.
                    self.total_density -= 1;
                } else {
                    // Otherwise, set own first bit.
                    self.bv.set(0, true);
                }
            }
            // Now discard other's first bit, having accounted for it above, and extend self by remaining bits.
            self.bv.extend(other.bv.iter().skip(1));
        } else {
            // Not an input density, just extend straightforwardly.
            self.bv.extend(other.bv);
        }

        // Since any needed adjustments to total densities have been made, just sum the totals and keep the sum.
        self.total_density += other.total_density;
    }
}

fn dense_multiexp_inner<G: PrimeCurveAffine>(
    bases: Arc<Vec<G>>,
    exponents: Arc<Vec<<G::Scalar as PrimeField>::Repr>>,
    c: u32,
    handle_trivial: bool,
) -> Result<<G as PrimeCurveAffine>::Curve, GPUError> {
    // Perform this region of the multiexp
    let this = move |bases: Arc<Vec<G>>,
                     exponents: Arc<Vec<<G::Scalar as PrimeField>::Repr>>,
                     skip: u32|
                     -> Result<_, GPUError> {
        // Accumulate the result
        let mut acc = G::Curve::identity();

        // Create space for the buckets
        let mut buckets = vec![<G as PrimeCurveAffine>::Curve::identity(); (1 << c) - 1];

        let zero = G::Scalar::zero().into_repr();
        let one = G::Scalar::one().into_repr();

        for (base, &exp) in bases.iter().zip(exponents.iter()) {
            if exp != zero {
                if exp == one {
                    if handle_trivial {
                        acc.add_assign_mixed(base);
                    }
                } else {
                    let mut exp = exp;
                    exp.shr(skip);
                    let exp = exp.as_ref()[0] % (1 << c);
                    if exp != 0 {
                        buckets[(exp - 1) as usize].add_assign_mixed(base);
                    }
                }
            }
        }

        // Summation by parts
        // e.g. 3a + 2b + 1c = a +
        //                    (a) + b +
        //                    ((a) + b) + c
        let mut running_sum = G::Curve::identity();
        for exp in buckets.into_iter().rev() {
            running_sum.add_assign(&exp);
            acc.add_assign(&running_sum);
        }

        Ok(acc)
    };

    let parts = (0..G::Scalar::NUM_BITS)
        .into_par_iter()
        .step_by(c as usize)
        .map(|skip| this(bases.clone(), exponents.clone(), skip))
        .collect::<Vec<Result<_, _>>>();

    parts
        .into_iter()
        .rev()
        .try_fold(<G as PrimeCurveAffine>::Curve::identity(), |mut acc, part| {
            for _ in 0..c {
                acc.double();
            }

            acc.add_assign(&part?);
            Ok(acc)
        })
}

/// Perform multi-exponentiation. The caller is responsible for ensuring the
/// query size is the same as the number of exponents.
pub fn dense_multiexp<G: PrimeCurveAffine>(
    pool: &Worker,
    bases: Arc<Vec<G>>,
    exponents: Arc<Vec<<G::Scalar as PrimeField>::Repr>>,
    kern: &mut Option<gpu::LockedMultiexpKernel<G::Engine>>,
) -> Waiter<Result<<G as PrimeCurveAffine>::Curve, GPUError>> {
    assert_eq!(bases.len(), exponents.len(), "bases must be equal to exps.");
    let n = bases.len();
    if let Some(ref mut kern) = kern {
        if let Ok(p) = kern.with(|k: &mut gpu::MultiexpKernel<G::Engine>| {
            let bss = bases.clone();
            let exps = exponents.clone();
            k.dense_multiexp(pool, bss, exps, n)
        }) {
            return Waiter::done(Ok(p));
        }
    }

    let c = if exponents.len() < 32 {
        3u32
    } else {
        (f64::from(exponents.len() as u32)).ln().ceil() as u32
    };

    let result = pool.compute(move || dense_multiexp_inner(bases, exponents, c, true));
    #[cfg(feature = "gpu")]
        {
            // Do not give the control back to the caller till the
            // multiexp is done. We may want to reacquire the GPU again
            // between the multiexps.
            let result = result.wait();
            Waiter::done(result)
        }
}

pub fn create_multiexp_kernel<E>(_log_d: usize, priority: bool) -> Option<gpu::MultiexpKernel<E>>
    where
        E: crate::pairing::Engine,
{
    match gpu::MultiexpKernel::<E>::create(priority) {
        Ok(k) => {
            info!("GPU Multiexp kernel instantiated!");
            Some(k)
        }
        Err(e) => {
            warn!("Cannot instantiate GPU Multiexp kernel! Error: {}", e);
            None
        }
    }
}

#[cfg(feature = "gpu")]
#[test]
pub fn gpu_multiexp_consistency() {
    use crate::pairing::bn256::Bn256;
    use std::time::Instant;

    let _ = env_logger::try_init();
    gpu::dump_device_list();

    const MAX_LOG_D: usize = 26;
    const START_LOG_D: usize = 22;

    let mut kern = Some(gpu::LockedMultiexpKernel::<Bn256>::new(MAX_LOG_D, false));

    let pool = Worker::new();

    let rng = &mut rand::thread_rng();

    let mut bases = (0..(1 << 10))
        .map(|_| <Bn256 as crate::arithmetic::Engine>::G1::random(rng).into_affine())
        .collect::<Vec<_>>();
    for _ in 10..START_LOG_D {
        bases = [bases.clone(), bases.clone()].concat();
    }

    for log_d in START_LOG_D..=MAX_LOG_D {
        let samples = 1 << log_d;
        println!("Testing Multiexp for {} elements...", samples);

        let v = (0..samples)
            .map(|_| {
                // <Bn256 as ScalarEngine>::Fr::from_str("1000")
                <Bn256::Scalar as PrimeField>::from_repr()
                    .unwrap()
                    .into_repr()
            })
            .collect::<Vec<_>>();

        let mut now = Instant::now();
        let gpu = dense_multiexp(
            &pool,
            Arc::new(bases.clone()),
            Arc::new(v.clone()),
            &mut kern,
        )
            .wait()
            .unwrap();

        let gpu_dur = now.elapsed().as_secs() * 1000 as u64 + now.elapsed().subsec_millis() as u64;
        println!("GPU took {}ms.", gpu_dur);

        now = Instant::now();
        let cpu = dense_multiexp(
            &pool,
            Arc::new(bases.clone()),
            Arc::new(v.clone()),
            &mut None,
        )
            .wait()
            .unwrap();
        let cpu_dur = now.elapsed().as_secs() * 1000 as u64 + now.elapsed().subsec_millis() as u64;
        println!("CPU took {}ms.", cpu_dur);

        println!("Speedup: x{}", cpu_dur as f32 / gpu_dur as f32);

        assert_eq!(cpu, gpu);

        println!("============================");

        bases = [bases.clone(), bases.clone()].concat();
    }
}
