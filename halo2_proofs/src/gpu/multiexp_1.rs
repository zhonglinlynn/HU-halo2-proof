use pairing_bn256::{
    ff::{Field, PrimeField, PrimeFieldRepr, ScalarEngine},
    CurveAffine, CurveProjective, Engine,
};

use bit_vec::{self, BitVec};
use log::{info, warn};
use rayon::prelude::*;
use std::io;
use std::iter;
use std::sync::Arc;

use super::worker::{Waiter, Worker};
use super::error::GPUError;
use crate::gpu;

/// An object that builds a source of bases.
pub trait SourceBuilder<G: CurveAffine>: Send + Sync + 'static + Clone {
    type Source: Source<G>;

    fn new(self) -> Self::Source;
    fn get(self) -> (Arc<Vec<G>>, usize);
}

/// A source of bases, like an iterator.
pub trait Source<G: CurveAffine> {
    /// Parses the element from the source. Fails if the point is at infinity.
    fn add_assign_mixed(
        &mut self,
        to: &mut <G as CurveAffine>::Projective,
    ) -> Result<(), GPUError>;

    /// Skips `amt` elements from the source, avoiding deserialization.
    fn skip(&mut self, amt: usize) -> Result<(), GPUError>;
}

impl<G: CurveAffine> SourceBuilder<G> for (Arc<Vec<G>>, usize) {
    type Source = (Arc<Vec<G>>, usize);

    fn new(self) -> (Arc<Vec<G>>, usize) {
        (self.0.clone(), self.1)
    }

    fn get(self) -> (Arc<Vec<G>>, usize) {
        (self.0.clone(), self.1)
    }
}

impl<G: CurveAffine> Source<G> for (Arc<Vec<G>>, usize) {
    fn add_assign_mixed(
        &mut self,
        to: &mut <G as CurveAffine>::Projective,
    ) -> Result<(), GPUError> {
        if self.0.len() <= self.1 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "expected more bases from source",
            )
                .into());
        }

        if self.0[self.1].is_zero() {
            return Err(GPUError::UnexpectedIdentity);
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

fn dense_multiexp_inner<G: CurveAffine>(
    bases: Arc<Vec<G>>,
    exponents: Arc<Vec<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>>,
    c: u32,
    handle_trivial: bool,
) -> Result<<G as CurveAffine>::Projective, GPUError> {
    // Perform this region of the multiexp
    let this = move |bases: Arc<Vec<G>>,
                     exponents: Arc<Vec<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>>,
                     skip: u32|
                     -> Result<_, GPUError> {
        // Accumulate the result
        let mut acc = G::Projective::zero();

        // Create space for the buckets
        let mut buckets = vec![<G as CurveAffine>::Projective::zero(); (1 << c) - 1];

        let zero = <G::Engine as ScalarEngine>::Fr::zero().into_repr();
        let one = <G::Engine as ScalarEngine>::Fr::one().into_repr();

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
        let mut running_sum = G::Projective::zero();
        for exp in buckets.into_iter().rev() {
            running_sum.add_assign(&exp);
            acc.add_assign(&running_sum);
        }

        Ok(acc)
    };

    let parts = (0..<G::Engine as ScalarEngine>::Fr::NUM_BITS)
        .into_par_iter()
        .step_by(c as usize)
        .map(|skip| this(bases.clone(), exponents.clone(), skip))
        .collect::<Vec<Result<_, _>>>();

    parts
        .into_iter()
        .rev()
        .try_fold(<G as CurveAffine>::Projective::zero(), |mut acc, part| {
            for _ in 0..c {
                acc.double();
            }

            acc.add_assign(&part?);
            Ok(acc)
        })
}

/// Perform multi-exponentiation. The caller is responsible for ensuring the
/// query size is the same as the number of exponents.
pub fn dense_multiexp<G: CurveAffine>(
    pool: &Worker,
    bases: Arc<Vec<G>>,
    exponents: Arc<Vec<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>>,
    kern: &mut Option<gpu::LockedMultiexpKernel<G::Engine>>,
) -> Waiter<Result<<G as CurveAffine>::Projective, GPUError>> {
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
    #[cfg(not(feature = "gpu"))]
        result
}

#[cfg(any(feature = "pairing", feature = "blst"))]
#[test]
fn test_with_bls12() {
    fn naive_multiexp<G: CurveAffine>(
        bases: Arc<Vec<G>>,
        exponents: Arc<Vec<<G::Scalar as PrimeField>::Repr>>,
    ) -> G::Projective {
        assert_eq!(bases.len(), exponents.len());

        let mut acc = G::Projective::zero();

        for (base, exp) in bases.iter().zip(exponents.iter()) {
            acc.add_assign(&base.mul(*exp));
        }

        acc
    }

    use crate::pairing::bls12_381::Bls12;
    use crate::pairing::Engine;
    use rand;

    const SAMPLES: usize = 1 << 14;

    let rng = &mut rand::thread_rng();
    let v = Arc::new(
        (0..SAMPLES)
            .map(|_| <Bls12 as ScalarEngine>::Fr::random(rng).into_repr())
            .collect::<Vec<_>>(),
    );
    let g = Arc::new(
        (0..SAMPLES)
            .map(|_| <Bls12 as Engine>::G1::random(rng).into_affine())
            .collect::<Vec<_>>(),
    );

    let now = std::time::Instant::now();
    let naive = naive_multiexp(g.clone(), v.clone());
    println!("Naive: {}", now.elapsed().as_millis());

    let now = std::time::Instant::now();
    let pool = Worker::new();

    let fast = multiexp(&pool, (g, 0), FullDensity, v, &mut None)
        .wait()
        .unwrap();

    println!("Fast: {}", now.elapsed().as_millis());

    assert_eq!(naive, fast);
}

pub fn create_multiexp_kernel<E>(_log_d: usize, priority: bool) -> Option<gpu::MultiexpKernel<E>>
    where
        E: pairing_bn256::Engine,
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
    use crate::rand::Rand;
    use std::time::Instant;

    let _ = env_logger::try_init();
    gpu::dump_device_list();

    const MAX_LOG_D: usize = 26;
    const START_LOG_D: usize = 22;

    let mut kern = Some(gpu::LockedMultiexpKernel::<Bn256>::new(MAX_LOG_D, false));

    let pool = Worker::new();

    let rng = &mut rand::thread_rng();

    let mut bases = (0..(1 << 10))
        .map(|_| <Bn256 as crate::pairing::Engine>::G1::rand(rng).into_affine())
        .collect::<Vec<_>>();
    for _ in 10..START_LOG_D {
        bases = [bases.clone(), bases.clone()].concat();
    }

    for log_d in START_LOG_D..=MAX_LOG_D {
        let samples = 1 << log_d;
        println!("Testing Multiexp for {} elements...", samples);

        let v = (0..samples)
            .map(|_| {
                <Bn256 as ScalarEngine>::Fr::from_str("1000")
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

// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     use rand::Rng;
//     use rand_core::SeedableRng;
//     use rand_xorshift::XorShiftRng;
//
//     #[test]
//     fn test_extend_density_regular() {
//         let mut rng = XorShiftRng::from_seed([
//             0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
//             0xbc, 0xe5,
//         ]);
//
//         for k in &[2, 4, 8] {
//             for j in &[10, 20, 50] {
//                 let count: usize = k * j;
//
//                 let mut tracker_full = DensityTracker::new();
//                 let mut partial_trackers: Vec<DensityTracker> = Vec::with_capacity(count / k);
//                 for i in 0..count {
//                     if i % k == 0 {
//                         partial_trackers.push(DensityTracker::new());
//                     }
//
//                     let index: usize = i / k;
//                     if rng.gen() {
//                         tracker_full.add_element();
//                         partial_trackers[index].add_element();
//                     }
//
//                     if !partial_trackers[index].bv.is_empty() {
//                         let idx = rng.gen_range(0, partial_trackers[index].bv.len());
//                         let offset: usize = partial_trackers
//                             .iter()
//                             .take(index)
//                             .map(|t| t.bv.len())
//                             .sum();
//                         tracker_full.inc(offset + idx);
//                         partial_trackers[index].inc(idx);
//                     }
//                 }
//
//                 let mut tracker_combined = DensityTracker::new();
//                 for tracker in partial_trackers.into_iter() {
//                     tracker_combined.extend(tracker, false);
//                 }
//                 assert_eq!(tracker_combined, tracker_full);
//             }
//         }
//     }
//
//     #[test]
//     fn test_extend_density_input() {
//         let mut rng = XorShiftRng::from_seed([
//             0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
//             0xbc, 0xe5,
//         ]);
//         let trials = 10;
//         let max_bits = 10;
//         let max_density = max_bits;
//
//         // Create an empty DensityTracker.
//         let empty = || DensityTracker::new();
//
//         // Create a random DensityTracker with first bit unset.
//         let unset = |rng: &mut XorShiftRng| {
//             let mut dt = DensityTracker::new();
//             dt.add_element();
//             let n = rng.gen_range(1, max_bits);
//             let target_density = rng.gen_range(0, max_density);
//             for _ in 1..n {
//                 dt.add_element();
//             }
//
//             for _ in 0..target_density {
//                 if n > 1 {
//                     let to_inc = rng.gen_range(1, n);
//                     dt.inc(to_inc);
//                 }
//             }
//             assert!(!dt.bv[0]);
//             assert_eq!(n, dt.bv.len());
//             dbg!(&target_density, &dt.total_density);
//
//             dt
//         };
//
//         // Create a random DensityTracker with first bit set.
//         let set = |mut rng: &mut XorShiftRng| {
//             let mut dt = unset(&mut rng);
//             dt.inc(0);
//             dt
//         };
//
//         for _ in 0..trials {
//             {
//                 // Both empty.
//                 let (mut e1, e2) = (empty(), empty());
//                 e1.extend(e2, true);
//                 assert_eq!(empty(), e1);
//             }
//             {
//                 // First empty, second unset.
//                 let (mut e1, u1) = (empty(), unset(&mut rng));
//                 e1.extend(u1.clone(), true);
//                 assert_eq!(u1, e1);
//             }
//             {
//                 // First empty, second set.
//                 let (mut e1, s1) = (empty(), set(&mut rng));
//                 e1.extend(s1.clone(), true);
//                 assert_eq!(s1, e1);
//             }
//             {
//                 // First set, second empty.
//                 let (mut s1, e1) = (set(&mut rng), empty());
//                 let s2 = s1.clone();
//                 s1.extend(e1, true);
//                 assert_eq!(s1, s2);
//             }
//             {
//                 // First unset, second empty.
//                 let (mut u1, e1) = (unset(&mut rng), empty());
//                 let u2 = u1.clone();
//                 u1.extend(e1, true);
//                 assert_eq!(u1, u2);
//             }
//             {
//                 // First unset, second unset.
//                 let (mut u1, u2) = (unset(&mut rng), unset(&mut rng));
//                 let expected_total = u1.total_density + u2.total_density;
//                 u1.extend(u2, true);
//                 assert_eq!(expected_total, u1.total_density);
//                 assert!(!u1.bv[0]);
//             }
//             {
//                 // First unset, second set.
//                 let (mut u1, s1) = (unset(&mut rng), set(&mut rng));
//                 let expected_total = u1.total_density + s1.total_density;
//                 u1.extend(s1, true);
//                 assert_eq!(expected_total, u1.total_density);
//                 assert!(u1.bv[0]);
//             }
//             {
//                 // First set, second unset.
//                 let (mut s1, u1) = (set(&mut rng), unset(&mut rng));
//                 let expected_total = s1.total_density + u1.total_density;
//                 s1.extend(u1, true);
//                 assert_eq!(expected_total, s1.total_density);
//                 assert!(s1.bv[0]);
//             }
//             {
//                 // First set, second set.
//                 let (mut s1, s2) = (set(&mut rng), set(&mut rng));
//                 let expected_total = s1.total_density + s2.total_density - 1;
//                 s1.extend(s2, true);
//                 assert_eq!(expected_total, s1.total_density);
//                 assert!(s1.bv[0]);
//             }
//         }
//     }
// }
