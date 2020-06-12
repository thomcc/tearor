//! `tearor` provides the [`TearCell`], a (barely) thread-safe lock-free cell
//! type providing tearing access to any type which is [`Pod`].
//!
//! Tearing access refers to when multiple smaller, separate read or write
//! operations are used to perform a larger unit of work. For example, if you
//! wrote to a &mut u32 by performing 4 writes, one to each byte, or vice-versa.
//!
//! TearCell uses the same idea, but with atomics. If your `T` is too large to
//! fit inside an atomic, then `TearCell` will split it over a few operations.
//!
//! Needless to say, this means calls to `TearCell::load`, `TearCell::store`,
//! (etc) are *not* atomic (nor do they provide *any* guarantees about
//! ordering), however every individual operation the TearCell performs *is*
//! atomic (with the weakest ordering we can get our hands on), which is enough
//! to avoid data races.
//!
//! It's essentially a tool for turning data races into data corruption. If the
//! lack of synchronization would cause a data race (e.g. with UnsafeCell), then
//! `TearCell` is very likely to corrupt your data.
//!
//! However, if this does not matter to you for one reason or another (examples:
//! your synchronization is performed externally, you want to perform an
//! optimistic read, all threads are writing the same value, or you miss the fun
//! you had debugging data corruption issues in C++), then `TearCell` might be
//! what you want.
//!
//! # This library might not be for you if...
//!
//! - You aren't sure if tearing is acceptable for your use case, or aren't sure
//!   you understand what it means. You should just use a `Mutex<T>` or
//!   `RwLock<T>` then. These are much harder to misuse, and can't corrupt your
//!   data.
//!
//! - You don't require `Sync` -- e.g. `Cell<T>` would be acceptable for your
//!   use case. In this case, use `Cell<T>`.
//!
//! - Your `T` is not `Send`. In this case, `TearCell<T>` will be neither `Send`
//!   nor `Sync`, which makes it pretty useless.
//!
//! - Your `T` is not [plain-old-data]([`Pod`]) -- e.g. it has padding bytes,
//!   isn't `Copy`, contains references, has initialized bit patterns which
//!   cause undefined behavior, etc. (Example: #[repr(Rust)] types, bools,
//!   enums, char, references...).
//!
//! - Your `T` (is POD and) fits inside one of the atomic types in
//!   `core::sync::atomic`. TearCell will literally be worse in every way than
//!   just performing `Relaxed` loads/stores to that type.
//!
//! - You need to compile for a target which doesn't support `AtomicU8` or
//!   `AtomicUsize`. Currently this is unsupported, but I'd accept a PR adding
//!   the `#[cfg]` needed.
//!
//! - `T` is fairly large, but only has a 1-byte alignment, or has an odd size.
//!   (For example, `[u8; 127]`). For these the performance will be suboptimal,
//!   but this may be improved in the future.

#![no_std]

#[cfg(test)]
extern crate alloc;

// Re-export relevant types from bytemuck
pub use bytemuck::{Pod, Zeroable};

use core::cell::UnsafeCell;
use core::mem::{align_of, size_of};
use core::sync::atomic::*;

/// `TearCell` is a minimally thread-safe cell type for anything that can meet the
/// requirements of `Pod`.
///
/// By "minimally thread-safe", I mean that it's completely possible for
/// `TearCell` to corrupt whatever you store in it, but that *strictly speaking*
/// all reads and writes are atomic, and thus you have no undefined behavior.
/// That said, writes and reads from the `TearCell` itself are *not* atomic.
///
/// It's intended for cases where `T` does not fit into an atomic type, you
/// don't want to use a Mutex, and you happen to know that *in practice* data
/// races aren't a concern. (For example, if synchronization is provided
/// externally but in a way you can't prove, or you just don't really care).
///
/// ## Implementation details
///
/// **Note: These are not part of TearCell's stability guarantee.**
///
/// `TearCell<T>` wraps an `UnsafeCell<T>` and implements reads and writes by
/// interpreting it as either a `&[AtomicU$N]`, where `$N` is the largest
/// implemented integer size available that meets `T`'s size and alignment
/// requirements.
///
/// For loads, we zero-init a target `T`, interpret it as a `&mut [u$N]`, and
/// perform relaxed loads from our atomics, and write the result into the
/// buffer.
///
/// For stores, we interpret the provided value as a `&[u$N]`, and perform a
/// relaxed store into our atomic buffer for each value we read from it.
#[repr(transparent)]
pub struct TearCell<T>(UnsafeCell<T>);

unsafe impl<T: Pod + Send> Sync for TearCell<T> {}
unsafe impl<T: Pod + Send> Send for TearCell<T> {}

impl<T> TearCell<T> {
    /// Create a new TearCell, wrapping `v`.
    ///
    /// This should probably only be done if `T: Pod`, as it's almost entirely
    /// useless without it. However, putting a `where T: Pod` bound currently
    /// would mean `TearCell::new` can no longer be a `const fn`, which makes it
    /// drastically less useful.
    #[inline]
    pub const fn new(v: T) -> Self {
        Self(UnsafeCell::new(v))
    }

    /// Unwraps the stored value.
    #[inline]
    pub fn into_inner(self) -> T {
        self.0.into_inner()
    }
}

#[inline]
fn tearcell_can_use_atom<T, U: Atom>() -> bool {
    (align_of::<T>() >= align_of::<U>())
        && (size_of::<T>() >= size_of::<U>())
        && (size_of::<T>() % size_of::<U>()) == 0
}

impl<T: Pod> TearCell<T> {
    /// Store `value` into the cell. No guarantees of ordering or atomicity is
    /// provided for this write, only that it will not cause undefined
    /// behavior.
    ///
    /// See also `store_ref` if `T` is large enough that you'd rather not copy
    /// it on the stack when you can avoid it.
    #[inline]
    pub fn store(&self, value: T) {
        self.store_ref(&value)
    }

    /// Read the value out of this cell. No guarantees of ordering or atomicity
    /// is provided for this write, only that it will not cause undefined
    /// behavior.
    ///
    /// Specifically, the `T` returned may not be one ever written. See docs for
    /// more info.
    #[inline]
    pub fn load(&self) -> T {
        if size_of::<T>() == 0 {
            T::zeroed()
        } else if tearcell_can_use_atom::<T, AtomicU64>() {
            self.do_load::<AtomicU64>()
        } else if tearcell_can_use_atom::<T, AtomicU32>() {
            self.do_load::<AtomicU32>()
        } else if tearcell_can_use_atom::<T, AtomicU16>() {
            self.do_load::<AtomicU16>()
        } else if tearcell_can_use_atom::<T, AtomicU8>() {
            self.do_load::<AtomicU8>()
        } else {
            unreachable!();
        }
    }

    /// Equivalent to `store` but takes the value by reference. No guarantees of
    /// ordering or atomicity is provided for this write, only that it will not
    /// cause undefined behavior.
    #[inline]
    pub fn store_ref(&self, value: &T) {
        if size_of::<T>() == 0 {
            return;
        } else if tearcell_can_use_atom::<T, AtomicU64>() {
            self.do_store::<AtomicU64>(value)
        } else if tearcell_can_use_atom::<T, AtomicU32>() {
            self.do_store::<AtomicU32>(value)
        } else if tearcell_can_use_atom::<T, AtomicU16>() {
            self.do_store::<AtomicU16>(value)
        } else if tearcell_can_use_atom::<T, AtomicU8>() {
            self.do_store::<AtomicU8>(value)
        } else {
            unreachable!();
        }
    }

    #[inline]
    fn atom_slice<A: Atom>(&self) -> &[A] {
        let size = size_of::<T>() / size_of::<A>();
        assert!(size != 0);
        assert!(size * size_of::<A>() == size_of::<T>());
        unsafe { core::slice::from_raw_parts(self.0.get() as *const A, size) }
    }

    #[inline]
    fn do_load<A: Atom>(&self) -> T {
        use core::mem::MaybeUninit;
        let mut result: MaybeUninit<T> = MaybeUninit::uninit();
        let src: &[A] = self.atom_slice();

        let dst: &mut [MaybeUninit<A::Prim>] = unsafe {
            core::slice::from_raw_parts_mut(
                result.as_mut_ptr() as *mut MaybeUninit<A::Prim>,
                size_of::<T>() / size_of::<A::Prim>(),
            )
        };
        assert_eq!(dst.len() * size_of::<A::Prim>(), size_of::<T>());

        assert_eq!(src.len(), dst.len());
        for (db, sb) in dst.iter_mut().zip(src.iter()) {
            // MaybeUninit::write() is unstable...
            unsafe {
                db.as_mut_ptr().write(sb.get());
            }
        }
        unsafe { result.assume_init() }
    }

    #[inline]
    fn do_store<A: Atom>(&self, v: &T) {
        let src: &[A::Prim] = bytemuck::try_cast_slice(core::slice::from_ref(v)).unwrap();
        let dst: &[A] = self.atom_slice();
        assert_eq!(src.len(), dst.len());
        for (d, s) in dst.iter().zip(src.iter()) {
            d.set(*s);
        }
    }
}

/// Used to impl the underlying tearing loads/stores.
unsafe trait Atom: Sync + Send + 'static + Sized {
    type Prim: Pod;
    fn get(&self) -> Self::Prim;
    fn set(&self, p: Self::Prim);
}
macro_rules! def_atom {
    ($Atom:ty, $prim:ty) => {
        unsafe impl Atom for $Atom {
            type Prim = $prim;
            #[inline]
            fn get(&self) -> Self::Prim {
                self.load(Ordering::Relaxed)
            }
            #[inline]
            fn set(&self, v: Self::Prim) {
                self.store(v, Ordering::Relaxed);
            }
        }
    };
}
def_atom!(AtomicU64, u64);
// def_atom!(AtomicUsize, usize);
def_atom!(AtomicU32, u32);
def_atom!(AtomicU16, u16);
def_atom!(AtomicU8, u8);

// Don't bother with u16/u32/etc -- we're only bothering with u8 because we
// can't statically prevent it.

#[cfg(test)]
mod test {
    use super::*;
    use core::convert::*;
    #[test]
    fn test_basic() {
        let v: TearCell<[u8; 0]> = TearCell::new([]);
        assert_eq!(v.load(), []);
        v.store([]);
        v.store_ref(&[]);

        let v: TearCell<[usize; 0]> = TearCell::new([]);
        assert_eq!(v.load(), []);
        v.store([]);
        v.store_ref(&[]);

        let v: TearCell<[u8; 0]> = TearCell::new([]);
        assert_eq!(v.load(), []);
        v.store([]);
        v.store_ref(&[]);

        let v: TearCell<[u8; 1]> = TearCell::new([0u8; 1]);
        assert_eq!(v.load(), [0]);
        v.store([1]);
        assert_eq!(v.load(), [1]);
        v.store_ref(&[2]);
        assert_eq!(v.load(), [2]);
    }
    macro_rules! test_arr {
        ($($n:expr),+) => {$(
            do_test_arr::<u8, [u8; $n]>();
            do_test_arr::<u16, [u16; $n]>();
            do_test_arr::<u32, [u32; $n]>();
            do_test_arr::<u64, [u64; $n]>();
            do_test_arr::<usize, [usize; $n]>();
            do_test_arr::<u128, [u128; $n]>();
        )+};
    }

    #[test]
    fn test_many() {
        test_arr![2, 3, 4, 5, 6, 7, 8, 9, 10];
        // slows down compile lots..
        //  11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        //     26, 27, 28, 29, 30, 31, 32
    }

    #[test]
    fn test_overaligned() {
        #[derive(Copy, Clone, PartialEq, Default, Debug)]
        #[repr(C, align(16))]
        struct Overaligned([u8; 16]);

        unsafe impl bytemuck::Zeroable for Overaligned {}
        unsafe impl bytemuck::Pod for Overaligned {}

        impl AsRef<[u8]> for Overaligned {
            fn as_ref(&self) -> &[u8] {
                &self.0
            }
        }

        do_test::<u8, Overaligned>(16, move |s: &[u8]| Overaligned(s.try_into().unwrap()));
    }

    fn do_test_arr<T, Arr>()
    where
        T: Copy + From<u8> + Default + core::ops::Not<Output = T> + PartialEq + core::fmt::Debug,
        Arr: Pod + AsRef<[T]> + PartialEq + core::fmt::Debug,
        for<'a> Arr: TryFrom<&'a [T]>,
    {
        do_test(
            core::mem::size_of::<Arr>() / core::mem::size_of::<T>(),
            move |v: &[T]| -> Arr { v.try_into().ok().unwrap() },
        )
    }
    fn do_test<T, Arr>(size: usize, make: fn(&[T]) -> Arr)
    where
        T: Copy + From<u8> + Default + core::ops::Not<Output = T> + PartialEq + core::fmt::Debug,
        Arr: Pod + AsRef<[T]> + PartialEq + core::fmt::Debug,
    {
        let mut v0 = alloc::vec![Default::default(); size];
        let a0: Arr = make(&v0);
        let tc0: TearCell<Arr> = TearCell::new(a0);
        assert_eq!(tc0.load(), a0);
        for i in 0..size {
            v0[i] = ((i + 1) as u8).into();
        }

        let a0: Arr = make(&v0);
        let tc1 = TearCell::new(a0);
        assert_eq!(&v0[..], tc1.load().as_ref());
        tc0.store(a0);
        assert_eq!(&v0[..], tc0.load().as_ref());
        v0.reverse();

        let ar0: Arr = make(&v0);
        tc0.store_ref(&ar0);
        assert_eq!(&v0[..], tc0.load().as_ref());

        for i in 0..size {
            v0[i] = !v0[i]
        }
        let a0: Arr = make(&v0);
        tc0.store(a0);
        assert_eq!(&v0[..], tc0.load().as_ref());

        tc1.store_ref(&a0);
        assert_eq!(&v0[..], tc0.load().as_ref());
    }
}
