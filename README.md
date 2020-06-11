# `tearor`

`tearor` provides the `TearCell`, a (barely) thread-safe lock-free cell
type providing tearing access to any type which is `Pod`.

Tearing access refers to when multiple smaller, separate read or write
operations are used to perform a larger unit of work. For example, if you
wrote to a &mut u32 by performing 4 writes, one to each byte, or vice-versa.

TearCell uses the same idea, but with atomics. If your `T` is too large to
fit inside an atomic, then `TearCell` will split it over a few operations.

Needless to say, this means calls to `TearCell::load`, `TearCell::store`,
(etc) are *not* atomic (nor do they provide *any* guarantees about
ordering), however every individual operation the TearCell performs *is*
atomic (with the weakest ordering we can get our hands on), which is enough
to avoid data races.

It's essentially a tool for turning data races into data corruption. If the
lack of synchronization would cause a data race (e.g. with UnsafeCell), then
`TearCell` is very likely to corrupt your data.

However, if this does not matter to you for one reason or another (examples:
your synchronization is performed externally, you want to perform an
optimistic read, all threads are writing the same value, or you miss the fun
you had debugging data corruption issues in C++), then `TearCell` might be
what you want.

# License

Public domain, as explained [here](https://creativecommons.org/share-your-work/public-domain/cc0/)

