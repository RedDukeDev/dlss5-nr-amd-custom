# Native kernels beside the translation

Tools for replacing the heaviest DLSS-NR kernels with hand-written HIP, one at a
time, while ZLUDA keeps translating everything else.

Layout: the kernels and dlssnr_hip.h are here, the Python tools in tools/, and
the research proxy they were developed with in harness/. The gate that runs
them in a game is runtime/gate, and its native_hook.h is shared with the
research proxy.

Nothing derived from NVIDIA's network is in this repository: no PTX, no
weights, no captured launches. The tools that need them read their locations
from the environment (tools/workspace.py), and captures are passed to them as
directories. The data directories named below (capture_*) are such captures,
made with tools/capture_launch.py.

## Why this shape

The translated path spends most of its time on work a native kernel does not do
at all: moving fragments between lanes to turn NVIDIA's layout into AMD's (54%
of a kernel, measured) and widening e4m3 to f16 (another 28%). The multiply
itself does not measure. No amount of tuning inside the translation removes
those two, because they exist only because of the translation.

The whole-frame picture that decides where to aim, measured at 2560x1440: 380 ms
in kernels, 43 distinct kernels, the top twelve are 72% of the frame, and launch
overhead is 2%.

Weight alone does not decide the order, though, because most of these kernels
are not free-standing. Of the thirteen hottest, eleven either copy through
`cp.async.bulk` with an `mbarrier`, or spin on a global flag another kernel
writes (`ld.relaxed` plus `nanosleep` until the value clears), or both. A native
replacement for one of those has to honour a synchronisation protocol shared
with the kernels around it, and getting that wrong does not produce a poor
image -- it hangs.

Only two of the thirteen are self-contained -- and they are self-contained for a
reason that costs more than it saves:

    cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8  30.6 ms  13 tex.2d reads
    cc_tinlayout_fused_post_block_swin_1h_32_fp8      28.7 ms  14 tex.2d reads,
                                                              2 sust.p.2d writes

They sit at the edges of the network and talk to the D3D12 resources directly,
which is exactly why they need no flags from anyone. Reproducing a textured read
means reproducing the sampler: addressing mode, filtering, and format
conversion, all of which must match bit for bit and none of which is stated in
the PTX. This project has already lost time once to a texture fetch that
sampled through the wrong pointer.

Set against that, the entanglement of the interior kernels is modest when read
rather than counted. The protocol is: spin on a global word until it rises above
-1 (`ld.relaxed` plus `nanosleep`), and store 0 with release semantics at the
end. That is five lines of HIP. The `cp.async.bulk` alongside it copies 512
bytes from global into LDS under an mbarrier, which on AMD is a cooperative copy
and a barrier.

So the first target is the shortest interior kernel, not the most isolated one:

    cc_split_swin_16h_ffwd_512_chained_fp8   14.0 ms   2301 lines   96 mma
                                             56-byte parameter, no textures,
                                             one flag in, one flag out

Three chained stages of 32 multiplies, a SiLU between them, four 128-bit stores
out. The heavier kernels follow once the mechanism has been proved on the one
that can be got right.

## Surviving a DLSS update

A native kernel encodes the shape of a layer -- tiles, channels, heads, fusion --
but not the weights. So a new DLSS build breaks it only if the *topology*
changes; a retrain does not. The design makes that failure harmless:

  - **identity gating**: each native kernel declares the name, the hash of the
    PTX body it replaces, and the launch geometry it expects. Anything that does
    not match falls back to the translated kernel automatically, per kernel;
  - **weights found by shape, not by address**: `extract_weights.py` anchors on the
    record names (`blockN.layerM.layer`), so it does not care where the archive
    sits. The published mods pin an offset and a SHA-256 and refuse any other
    build -- that is exactly the failure to avoid;
  - **ZLUDA stays the floor**: with every native kernel disabled the network
    still runs, only slower.

So an update costs a census and a diff (`kernel_census.py`,
`diff_versions.py`): if the kernels we replaced still hash the same, nothing to
do; if a family changed shape, that family loses its native path until it is
rewritten.

## The tools

    kernel_census.py   every kernel in a DLL or a PTX corpus: name, module,
                           hash of its body, parameter count -> a JSON manifest
                           (the PTX is zstd-compressed and is decompressed here)
    extract_kernel.py       one kernel's PTX out of that corpus, with the shared
                           arrays it names and nothing else: the shape ptxsim
                           and mappa_mma want, and the way the swin_*.ptx files
                           in this directory were made
    diff_versions.py       two manifests compared: identical / changed / added /
                           removed, and which native kernels that invalidates
    extract_weights.py         the weight archive, located by record shape; lists or
                           dumps the 153 records with sizes and hashes
    capture_arguments.py   the packed argument buffer of one kernel, captured
                           from a real run through the proxy -- the only way to
                           learn the ABI a native kernel has to honour
    native_hook.h          the gate itself (runtime/gate), included by the proxy:
                           binds the HIP runtime by name, loads a code object on
                           first use, and launches it on the caller's stream --
                           or returns false and lets the translation run
    kernels.txt            which native kernel stands in for which translated
                           one, with the block shape and argument size it is
                           gated on (NVCUDA_PROXY_NATIVE points at it)
    build_kernel.py        <name>.hip -> <name>.hsaco carrying every RDNA3,
                           RDNA3.5 and RDNA4 target in one code object, plus the
                           disassembly and, per kernel, registers, spills and
                           LDS. The runtime picks the architecture;
                           hipModuleLoadData takes the bundle as it is,
                           llvm-objdump does not, so one target is unbundled
                           before disassembling. --arch replaces the target
                           list, --flags the kernel's <name>.flags
    build_kernels.py       every kernel kernels.txt names, in parallel
    dump_buffers.h         one launch's buffers read back before and after it
                           runs (NVCUDA_PROXY_DUMP), so a native kernel can be
                           checked against real data element by element rather
                           than against a whole frame
    compare_images.py  two frames compared by how far apart they are rather
                           than by whether they are the same bytes
    capture_launch.py      one launch of any kernel captured: its argument
                           buffer and every buffer it points at, before and
                           after -- the ground truth a native kernel is checked
                           against
    ptxsim.py              a PTX interpreter: runs a kernel's own instructions
                           over a capture. What it reproduces is the meaning of
                           the PTX; what it cannot reproduce is the matrix
                           unit's rounding (see the note on WMMA below), so it
                           lands around 99.9% of bytes and that is the ceiling
    run_ptx.py             ptxsim on a capture, buffer by buffer. --z-serial
                           runs the grid's z slices in turn, for kernels whose
                           slices hand a partial sum to the next through a flag
    test_instructions.py    shfl, movmatrix and prmt checked against the hardware
                           through ZLUDA: a wrong model of one of them makes an
                           emulation that runs and produces nonsense, which is
                           exactly what happened
    ref_layers.py          a layer's arithmetic in channel order, checked
                           against ptxsim and a capture. This is what a native
                           kernel is written from; ffwd_ref.py is the same for
                           the ffwd layer, with the layout helpers both use
    dlssnr_hip.h           the native kernels' vocabulary: e4m3 widening and
                           narrowing, the fragment permutation, the 32-deep
                           slice (two WMMAs around an f16 accumulator, as ZLUDA
                           does it), the partner exchange, the tile flags, and
                           the RDNA4 path behind the same names (below)
    hip_run.py             a code object run from Python through HIP, with
                           buffers of our choosing and event timing
    ab_capture.py          a native kernel on a captured launch, compared with
                           the translated one byte for byte, and timed. --pool
                           rebuilds the capture's buffers as one allocation at
                           their captured distances, which a kernel whose
                           surface is larger than the dumped window needs
    mma_map.py           a layer's shape read off the multiplies it runs:
                           ptxsim keeps every mma's operands, and each weight
                           operand is located by its bytes in the blob, so the
                           listing is the layer's structure in execution order.
                           These kernels loop and reuse registers, so reading
                           the PTX as text, or the registers afterwards, says
                           something that was true at another moment
    probe_ptx.py           a probe written into a translated kernel: one PTX
                           register stored per lane per block, through a
                           parameter slot the kernel never reads
    probe_zluda.py         that kernel run on a capture through ZLUDA, on this
                           card, saving the probe or the whole output. The
                           output is the reference a native kernel must match
                           exactly -- the capture's own post dump is not, since
                           it only covers a window of each buffer
    test_e4m3.hip          the narrowing helpers over every f16 bit pattern
    bench_wmma.hip         what one WMMA costs on this card, in isolation

    test_gate.py          proves the fallback: the network run three times, with
                           no manifest and with two kinds of deliberately wrong
                           declaration, and all three images compared
    ab_native.py           native against translated on the real network, with
                           the image digest and the native-launch count as the
                           two things that can invalidate the measurement
    check_gfx12.py         the RDNA4 path of every kernel run on this card: all
                           built with DLSSNR_EMULATE_GFX12, the network run on
                           them, the reference image and every native launch
                           required. --only emulates some sources and keeps the
                           normal builds for the rest, to find which one differs
    gfx12_unit.py/.hip     the RDNA4 building blocks against the RDNA3 ones,
                           on thousands of random tiles, byte for byte

## Where it stands

Thirty-five kernels run natively, each producing the translated kernel's output bytes
and ready flags exactly, on a captured launch and on the whole network (the
image stays a48928166420fc45):

    cc_split_swin_16h_ffwd_512_chained_fp8        0.92 -> 0.11 ms   ffwd_512.hip
    cc_split_swin_16h_proj_512_chained_fp8        0.54 -> 0.09 ms   conv1x1_512.hip
    cc_split_swin_16h_ffwd_proj_512_chained_fp8   0.63 -> 0.07 ms
    cc_vit_1d_ffn_expand_chained_fp8              2.57 -> 0.26 ms   vit_ffn_expand.hip
    cc_vit_1d_ffn_contract_chained_fp8            2.25 -> 0.35 ms   vit_gemm.hip
    cc_vit_1d_projection_chained_fp8              0.68 -> 0.10 ms
    cc_tinlayout_fused_swin_1h_32_1_chained_fp8   7.28 -> 2.33 ms   swin_1h.hip
    cc_tinlayout_fused_swin_2h_64_2_chained_fp8   4.51 -> 2.74 ms   swin_2h.hip
    cc_split_swin_16h_qkv_512_chained_fp8         1.41 -> 0.34 ms   qkv_512.hip
    cc_tinlayout_fused_swin_4h_128_4_chained_fp8  4.06 -> 1.03 ms   swin_4h.hip
    cc_tinlayout_fused_swin_8h_256_8_chained_fp8  4.28 -> 0.89 ms   swin_8h.hip
    cc_tinlayout_fused_swin_1h_32_1_ds_wait_fp8   7.41 -> 2.86 ms   swin_1h.hip
    cc_tinlayout_fused_swin_2h_64_2_ds_wait_fp8   4.88 -> 3.49 ms   swin_2h.hip
    cc_tinlayout_fused_swin_4h_128_4_ds_wait_fp8  4.36 -> 1.17 ms   swin_4h.hip
    cc_tinlayout_fused_swin_8h_256_8_ds_wait_fp8  4.46 -> 0.99 ms   swin_8h.hip
    cc_tinlayout_fused_swin_1h_32_1_upsample_tilesync_fp8
                                                  7.87 -> 2.23 ms   swin_1h.hip
    cc_tinlayout_fused_swin_2h_64_2_upsample_tilesync_fp8
                                                 13.96 -> 9.45 ms   swin_2h.hip
    cc_tinlayout_fused_swin_4h_128_4_upsample_tilesync_fp8
                                                 11.72 -> 3.34 ms   swin_4h.hip
    cc_tinlayout_fused_swin_8h_256_8_upsample_tilesync_fp8
                                                 10.03 -> 3.17 ms   swin_8h.hip
    cc_tinlayout_fused_swin_1h_32_1_inpview_tilesync_fp8
                                                 19.74 -> 8.30 ms   swin_1h.hip
    cc_tinlayout_fused_swin_1h_32_1_outview_wait_fp8
                                                 21.89 -> 8.63 ms   swin_1h.hip
    cc_tinlayout_fused_swin_2h_64_2_inpview_tilesync_fp8
                                                 11.77 -> 10.26 ms  swin_2h.hip
    cc_tinlayout_fused_swin_2h_64_2_outview_wait_fp8
                                                 14.01 -> 10.60 ms  swin_2h.hip
    cc_tinlayout_fused_swin_4h_128_4_inpview_tilesync_fp8
                                                 10.88 -> 3.74 ms   swin_4h.hip
    cc_tinlayout_fused_swin_4h_128_4_outview_wait_fp8
                                                 11.91 -> 3.69 ms   swin_4h.hip
    cc_tinlayout_fused_swin_8h_256_8_inpview_tilesync_fp8
                                                 10.24 -> 3.29 ms   swin_8h.hip
    cc_tinlayout_fused_swin_8h_256_8_outview_wait_fp8
                                                 10.50 -> 3.21 ms   swin_8h.hip
    cc_vit_1d_qkv_chained_fp8                    40.35 -> 10.66 ms  vit_qkv.hip
    cc_vit_1d_attention_chained_fp8              39.08 -> 14.86 ms  vit_attention.hip
    cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8
                                                 95.43 -> 44.74 ms  swin_1h.hip
    cc_tinlayout_fused_post_block_swin_1h_32_fp8 85.50 -> 27.61 ms  swin_1h.hip
    cc_vit_1d_ffn_expand_publish_fp8              7.37 ->  1.06 ms  vit_ffn_expand.hip
    cc_split_swin_16h_ffwd_inpview_512_tilesync_fp8
                                                  3.62 ->  0.90 ms  ffwd_512.hip
    cc_split_swin_16h_ffwd_proj_inpview_512_chained_fp8   conv1x1_512.hip
    cc_split_swin_16h_proj_512_outview_wait_fp8           conv1x1_512.hip

(The figures from the ViT attention on are ab_native.py's frame sums over
three repeats, per-launch synchronised, so three times a frame's worth.)

The network went from 382 to 216 ms per frame at 2560x1440 on a 7900 XT, which
is -43%, measured with the first sixteen in one run (the translated baseline
drifts a few per cent between sessions, so both sides are always measured
together). At that point 156 of the 216 ms were still translated: the pre and
post blocks (62), the ViT stage's own attention in two kernels (28), and twelve
siblings of the fused blocks (61). The eleven siblings done since were each
measured one at a time against the translation, and each of those runs has
all the native kernels of the moment on its native side: in ab_native.py's
frame sum, which synchronises after every launch and so reads higher than a
plain frame, the network went 1145 -> 648 ms (-43%) with the first sixteen,
1134 -> ~600 (-47%) with the Swin siblings, and 1145 -> 543 (-52.6%) with the
ViT attention's two kernels as well. What is left translated is the pre and
post blocks -- see below for why they wait.

The four downsampling siblings are done, one per block size, each a second
entry in its own code object. A sibling is not the same kernel under another
name: `swin_1h_32_1_ds_wait` runs the same block on the same weights and takes
the same parameter, and then does one thing more. The window's 8x8 f16 result is averaged 2x2 -- `((a + b) +
(c + d)) * 0.25`, in that order -- narrowed, and put through one matrix of 2C
outputs that sits immediately after the output scales, at `FusedSwin.DS`. The
result is 16 tokens of 64 channels, written into a second surface at half the
resolution through a pointer the plain kernel leaves at zero, as four planes of
sixteen channels: plane p at `p * out_x * out_y * 16`, a token sixteen bytes
wide at `(y * out_x + x) * 16`, its bytes in fragment-column order. The
halved dimensions are the parameter's last pair, and the kernel raises no flag:
it waits, and signals nobody. The 88-byte siblings put the surface at +0x48 and
those dimensions at +0x50, where the chained kernel has its flag pointer and
nothing; the 96-byte 1H puts them at +0x40 and +0x48.

Scaling it across the family took only the split: the 2C outputs are two groups
of C, and a warp takes 32 of each, so a warp's pieces are `2w` and `2w+1`
within each group and the surface has 2C/16 planes. What the 8H needed on top
was room -- the average happens on the f16 result, before the narrowing, so its
window cannot stay as codes; its two 16 KB regions, both spent by then, become
one 32 KB buffer of 64 rows by 256 halves.

It also needed the padding. Its halved surface is 48 rows where the halved
image is 46, because the stage that reads it wants a multiple of its own
window, and no block's window reaches those two rows: they are zero, and the
last row of blocks has to fill them. This is the one thing a capture could not
have caught by itself -- the two rows are 4% of that surface, the kernel was
byte-identical everywhere else, and it was `ab_native.py` refusing the image
hash that said so. Which is what it is for: a kernel that matches a capture is
not a kernel that is right.

Twelve more siblings came after those, three kinds. `upsample_tilesync` was scouted far
enough to say what it is, and it is the `ds_wait` one read backwards: it takes
the halved surface a `ds_wait` kernel wrote, exactly as that kernel laid it out
-- same planes, same sixteen bytes a token -- for the 4x4 region under its 8x8
window, and puts those 2C channels through one matrix of C outputs. Its weight
blob is the plain one with that matrix inserted: `0x2000` for the 1H, so
everything after it moves by `0x800` (scale1 `0x2810`, qkv `0x2860`, bias
`0x3460`, the query scale `0x5460`, the projection `0x5470`, scale2 `0x5870`),
and C more f16 follow at `0x58B0`. Its parameter keeps the full-resolution
input at +0x50 with its dimensions at +0x58, puts the halved surface at +0x00,
and where `ds_wait` waits and signals nobody this one signals at +0x38 and
waits for nobody -- which is what the two suffixes mean.

It is written, as a third entry in swin_1h.hip, and it is exact. The block's
input is

    sum = up + f16(in * T)          T: 32 f16 right after the residual scale
    x   = e4m3(sum)                 what the branches see
    S   = f16(sum * s1) + branches  the residual takes the sum *unrounded*

where `up` is the halved surface's 4x4 under the window through one matrix at
`0x2000` (C outputs by 2C inputs, its four planes the four k-blocks in order)
and each low-resolution token serves the four output tokens of its 2x2. The
weight map is the plain one with that matrix and T inserted: the residual scale
is 0x800 along at `0x2810`, T sits at `0x2860`, and q,k,v, the bias, the query
scale, the projection and the output scale are 0x840 along. It waits for nobody
and raises its flag at +0x38.

Four things stood between the first version and the exact one, and each is a
kind of mistake worth recognising next time:

  * three offsets deduced from the plain map instead of measured. Each one that
    was then located from the operands the multiplies actually carry came out
    different. On a variant, measure the map;
  * a table taken for padding: T, which the PTX loads and multiplies by the
    input just before the sum. It is small (~0.1), so a missing T hides inside
    x's e4m3 rounding and only shows further on;
  * the residual reading the narrowed x where the translation reads the sum
    before it is narrowed;
  * a fused multiply-add. The PTX's `mul.f16x2` and `add.f16x2` carry no `.rn`,
    and the translation keeps them apart; HIP contracts them into one fma by
    default, which leaves one value in forty an ulp too far out -- invisible
    in x, 3.4% of the output. The probe (probe_ptx.py on the sum's register,
    against the same value written raw by a debug build) found it in one run,
    after several of guessing: 1963 of 2016 lanes identical, the rest exactly
    one ulp apart, all on the same side. `#pragma clang fp contract(off)` on
    that pair made the output byte-identical.

The other three sizes are done the same way, each a third entry in its own
code object, and exact. The stage scales as the downsampling one did: a warp
takes 32 of the C outputs and chunk c is planes 2c and 2c+1. Their maps were
located again from the operands, not scaled from the 1H one: the matrix sits
where the plain residual scale was (`0x7000`, `0x18000`, `0x58000`), then the
residual scale, then T, and everything else moves along by `0x2060`, `0x80E0`,
`0x201E0`. Two things did not carry over:

  * the residual. From 2H up it takes x, the sum narrowed, where the 1H takes
    the sum as it came -- the PTX shows it, the residual multiply's operand is
    a conversion from e4m3. A formula that is right for one size is a
    hypothesis for the next;
  * the halved surface's shape, which the reading side has to know. It is
    rounded up to a multiple of 4 per side, the patch, not of 8: 46 rows become
    48, 92 stay 92. And the rows added by that rounding are read -- the last row
    of blocks' 4x4 reaches them -- but count as outside: the bound is the
    halved image, `dim / 2`, not the surface. Reading the padding as data left
    the 8H's last patch row 38% wrong and everything else identical.

`inpview_tilesync` and `outview_wait` are done too, all eight, and they are the
simplest siblings: the chained block unchanged -- same multiplies, same weight
map, same template configuration but for the view flags -- with its input, or
its output, as a plain surface instead of patches. The surface is the one the
halved outputs already use, planes of sixteen channels at full resolution, and
its dimensions are the parameter's last pair (+0x48 in the 96-byte 1H, +0x50 in
the others), zero meaning the tokens' own. inpview signals and waits for
nobody; outview waits and signals nobody.

The map between the two layouts is one function, `surface_word` in
dlssnr_hip.h: a patch is 512-byte chunks of 32 channels, chunk c is planes 2c
and 2c+1, and word w of lane L in a chunk is word `L & 3` of token
`(L >> 2) + 8 * (w & 1)` in plane `2c + (w >> 1)`. It was read off the PTX,
by matching which mma results the chained kernel's b128 stores and the outview
kernel's 4-byte stores carry, and it was right first time in all eight.

One of them could not be checked on its capture: the 4H inpview's weight
pointer sits 128 KB before the end of its allocation, the proxy clamps the dump
there, and the kernel reads 0x30230 bytes, so run on the capture it sees zeros where the
weights go on. The network's image, which does not depend on dumps, is
identical with it native.

The ViT attention is two kernels, both done. `vit_1d_qkv` (vit_qkv.hip)
projects 1024 channels onto q, k and v of 32 heads and normalises q and k per
head, q then scaled by f16(sqrt 32) and a learned f32 per head; each is written
in the layout the attention reads -- q as activations, k in the B-fragment
layout with tokens as rows, v transposed in 32-token tiles. Those layouts were
found by voting: for every output byte, which (token, channel) of the
reference sum lands there, over all tiles and heads, which pins a permutation
down even where the reference is an ulp off. The PTX splits K over two z
slices and hands a partial across; the native kernel lets the z slice pick
the head instead and runs both halves of K itself, in the same two chains,
so the scratch buffer is never written. `vit_1d_attention` (vit_attention.hip)
is global attention in blocks of 64 keys with the Swin exponential trick on
its own constants and the normalisation after P . V. Its row sums go through a
transpose-and-reduce across lanes that is hard to read by eye; a small
symbolic interpreter of the PTX (32 lanes in lockstep, expression trees for
values) printed the exact tree, and the kernel was byte-identical first time.

The pre and post blocks are native too, and they are the network's edges:
they read the frame's textures and the post block writes the output image.
A ZLUDA texture handle is a pointer to the image descriptor with the sampler
12 dwords in, so a native kernel samples through the same intrinsic on the
same descriptors (sample2d, store2d in dlssnr_hip.h) and the filtering,
addressing and format conversion are the hardware's either way. The network's
harness binds only the colour texture, so half of their branches -- history
reprojection, the five-depth search, the five-tap Catmull-Rom history, a fifth
texture -- never run there. pre_block_check.py and post_block_check.py run
them instead: synthetic textures and buffers made through ZLUDA's own driver
API, the translated PTX run through ZLUDA and the native kernel through HIP in
the same process on the same objects, seven to ten branch combinations on
several sizes, every byte identical. approx_check.py and texture_check.py
hold the pieces they rest on: ZLUDA's approximate f32 operations and its
texture fetch, reproduced bit for bit.

The pre block builds 16 f16 inputs per token (noise from a hash through a
Box-Muller, colour, history, constants), puts them through one f16 matrix onto
32 channels and runs the 1H block on that -- the matrix's result, unrounded,
is the residual, as in the upsampling sibling -- then writes patches and a
halved surface. The post block runs the 1H block on the halved surface
(nearest) and the full-resolution features, each times a table, then one f16
matrix onto a colour and a blend logit per pixel, composed with the input
colour and the reprojected history. One trap cost a byte in 262 thousand: HIP
folds a multiply and the following f32 -> f16 conversion into one
mixed-precision instruction, which rounds once where the PTX rounds twice; on
a product sitting exactly on an f16 tie the two disagree. to_f16() rounds in
f32 first.

With them the network spends 126 ms of ab_native.py's synchronised frame sum
where the translation spent 382, -67%; 3.2 ms of it is still translated, in
eight small kernels (proj_pool, the ViT projection's waiting sibling, the
final head, the decoder's input upsampling, a clear, a copy, two repacks).

Tuning after correctness, measured with same_output.py -- two builds on one
capture, outputs compared byte for byte and timed alternately, because the
clock moves a few per cent between runs:

  * the 2H block spilled 639 registers; reading the state operands from LDS
    where they are used, and keeping v transposed and q's operands in LDS,
    took it to 160 and its time down a fifth;
  * the attention bias gathered from global memory in scattered 2-byte loads
    costs a few per cent; staging each 512-byte piece through LDS
    (bias_tile) pays in the 1H and 2H blocks and not in the wider ones, where
    the extra LDS or the synchronisation cost more -- kept where it pays;
  * the scheduler strategy max-memory-clause makes the 1H code object 3.5%
    faster; kept in swin_1h.flags, which build_kernel.py reads. The other
    blocks did not gain beyond the noise;
  * ablations on the 1H block put a fifth of its time in the four branches,
    and no single stage dominates: the rest is spread thin.

The proxy used to read at most 32 manifest lines and drop the rest without a
word; the table is now 256 long and says so when a manifest outgrows it.

The 2H block is the pattern for those: 64 channels, two heads, two warps, and
the work split three times -- a patch row each through the branches, a head each
through the attention, half the output channels each through the projection,
which is exactly one 512-byte chunk of every patch. It is also where the shape
of the layer stopped being the 1H block with bigger numbers: its four branches
produce a 32-wide result that one more matrix expands to 64, and the whole thing
runs twice, chained, before the attention. It gains less than the 1H block
(-39% against -68%) because 32 KB of LDS leaves two workgroups to a compute
unit; sharing the branch weights between the warps would halve that, and is the
first thing to try when the family is done.

The 16-head stage's attention is the one that came whole rather than in pieces.
`cc_split_swin_16h_qkv_512_chained_fp8` takes one 8x8 window of 64 tokens per
block and the grid's z slice with the warp index pick the head, so a warp
carries head `4z + warp` from the projection to the output on its own. Its 512
input channels cross LDS one 32-channel chunk at a time -- all four warps are on
the same chunk -- which holds the block to 28 KB while every warp still sees
every token. Once the chunks are done the arithmetic is the 2H block's, head for
head, and the attention reuses 2 KB of the weight rows it has just finished
with. It was bit-identical on the first run, which is what the method is for:
everything surprising about this kernel had already been settled in numpy.

Two things it taught, both about reading a kernel rather than writing one. It
stages its activations with `cp.async.bulk`, and one copy carries four lanes
with four different addresses: a trace that reads only lane zero says the block
touches 8 KB when it touches 32. And its attention operands did not match our
q and k until the channels were put through PERM32 -- the fragment order, which
cancels in q.k and so need not appear in the reference at all, but decides the
operand order in the kernel.

The 2H, 4H and 8H blocks turned out to be one layer in three sizes -- what the
2H's decoding called a "pass" is a head -- and their weight blobs follow a
single formula in the channel count and the head count, which the three
independently decoded maps confirm to the byte. `ref_layers.py`'s `FusedSwin`
states it once and reproduces all three references exactly; a fourth size would
be decoded by filling in two numbers.

The 4H block is the 2H one with four heads and 128 channels, and its four warps
split the work four times rather than three: a head each through the branches,
then 32 output channels each for the expansion -- which is where the four heads
chain, in head order, so the result each warp produces has to cross shared
memory first -- a head each again through the attention, and 32 output channels
each for the projection. Its weights arrive a 32-channel chunk at a time, as
`qkv_512`'s do: four warps holding whole matrices would not fit, and the chunk
is the unit the f16 chain rounds on anyway. Two 16 KB regions of LDS hold two
things each in turn, which is what keeps the block at 40 KB.

Its reference agrees with the translation on only 77-95% of the output codes,
against 98% for the 2H block, and that is not a fault in it: every stage was
checked against the emulator's own multiplies -- the residual exactly, the
scores to one f16 ulp -- and what accumulates is the gap between a reference
that sums exactly and hardware that truncates, over four chained heads instead
of two passes. The kernel written from it was bit-identical on the first run,
as the 2H one was from a 98% reference.

The 8H block has the same split again, one head and one 32-channel output group
to a warp, and what is new in it is the memory. Eight warps and 256 channels do
not fit with the 4H block's f16 buffers -- the window alone would be 32 KB -- so
what crosses shared memory stays e4m3 and is widened where it is used. That is
not a compromise: sixteen bytes read in one go are exactly one operand's sixteen
fragment columns, so it is a single LDS read where an f16 row costs two, and
widening a code is exact, so the values are the same to the bit. Two 16 KB
regions hold two things each in turn, as in the 4H, and the block lands at 48 KB
with eight waves on a compute unit.

The route each of these took, and the one the rest will take:

  0. lift the kernel's PTX out of the corpus (kernel_census.py, then
     extract_kernel.py) if it is not here already;
  1. capture a launch (capture_launch.py);
  2. run the PTX over it (run_ptx.py) -- if that does not reproduce the capture,
     the emulator is missing an instruction or modelling one wrongly, and
     nothing downstream is trustworthy;
  3. write the layer as matrices (ref_layers.py) and prove it identical to the
     emulation, not merely close: that is where a misread constant or a
     misunderstood chain of roundings shows up;
  4. write the HIP kernel from that statement, with dlssnr_hip.h's pieces, and
     check it against the capture (ab_capture.py);
  5. register it in kernels.txt and measure the network (ab_native.py).

Step 4 has a step of its own when the bytes nearly match but not quite. The
emulation cannot arbitrate there -- it is itself ~99.9% against the hardware --
so the translated kernel is made to hand over the intermediate value: a probe
goes into its PTX (probe_ptx.py), it runs on the same card through ZLUDA
(probe_zluda.py), and the native kernel writes its own view of the same value
into the other half of the probe buffer. Walking the two forward stage by stage
turns "a thousandth of the bytes differ" into one named quantity.

Numerics: a 32-deep slice is done the way ZLUDA does it on RDNA3 -- the f16
accumulator widened to f32, two 16-deep WMMAs, narrowed back -- which makes the
native kernels bit-identical to the translation rather than merely close. The
same WMMA hardware is not IEEE (it truncates a fixed-point window and negates
like one's complement), which is why ptxsim, which is exact, stops at ~99.9%.

A second trap, from the downsampling sibling: a guard that depends on the lane
must not enclose a cross-lane exchange. Write `if (lane < 16)` around anything
that reaches `permlanex16` -- `swap_rows`, and so `tile_to_operand` -- and the
compiler pulls the guard up over the exchange; the partner lane is inactive
when it runs; and `permlanex16` with `fetch_inactive` clear hands back the
lane's own value. Nothing warns. The result comes out with every odd row a copy
of the even one before it. A `wave_barrier` does not help, being a scheduling
barrier and not a constraint on control flow. Either let the whole wave store,
or keep the guard dependent on `lane & 15` alone, so that a lane and its
partner are always together.

A device-side trap worth naming, because it costs an afternoon: a `constexpr`
array indexed by a runtime variable inside a kernel reads element zero, every
time, with no diagnostic. The 2H block ran its second pass with the first
pass's weights until the offsets became plain constants passed to a lambda. What
found it was running the first pass twice on purpose and seeing the same bytes
the ordinary run produced.

Two consequences of that truncation are easy to walk past. The order of the
sixteen terms inside one WMMA is visible in the last bit, so an operand built
straight out of shared memory, in index order, is not the same product as the
one the translation issues from its fragments: swin_1h.hip reorders the
probabilities and v into fragment order for that reason alone. And an f16 sum
of many terms depends on the shape of its tree, so the sums in a native kernel
follow the translation's tree rather than a convenient one.

## RDNA4

Every native kernel also carries a gfx1200 and gfx1201 image, and there the
e4m3 products run on the card's own FP8 matrix instruction,
`v_wmma_f32_16x16x16_fp8_fp8`, at twice the f16 rate and with nothing widened.
No RDNA4 card has run them yet. What has been checked, and how, is below.

**The layouts.** They come from AMD's Matrix Instruction Calculator
(github.com/ROCm/amd_matrix_instruction_calculator), which reproduces the RDNA3
layout these kernels were verified against on hardware. RDNA4 differs in three
ways:

- each lane holds eight of an operand's sixteen k, one half of the wave the
  low eight and the other the high eight;
- the f16 operand interleaves those halves in blocks of four, and fp8 does
  not;
- the accumulator's register v holds row 8u + v instead of 2v + u.

**One vocabulary for both.** The kernels still speak RDNA3: results as rows
2v + u, operands in fragment-column order. RDNA4 is reached by feeding the A
operand with its rows permuted, sigma(m) = 2(m % 8) + m / 8, which makes the
RDNA4 accumulator come out in exactly the RDNA3 arrangement. Nothing past the
multiply changes.

For an operand read from LDS the permutation is only an address: `lds_a8`
reads row sigma(i). For one held in registers it costs one `v_permlane16` per
dword.

**Operands as codes.** e4m3 operands are an `op8` built by a few helpers in
dlssnr_hip.h:

- on RDNA3 an `op8` is the same widened half16 as before, and the RDNA3 build
  is the same code under new names: every capture and the image are unchanged,
  and so are the timings;
- on RDNA4 an `op8` is the lane's eight codes, and LDS rows hold codes in the
  first half of their usual stride;
- rows that carry unrounded f16 (the upsampled input, the pre and post blocks'
  sums, the halved output's mean, the probabilities) stay f16 on both;
- products whose operands are not e4m3 (the pre block's input matrix, the post
  block's head) use the RDNA4 f16 instruction through the same `wmma()`.

**Checked here.** A build with `DLSSNR_EMULATE_GFX12` runs the RDNA4 code path
on this RDNA3 card. Every operand and accumulator goes through the RDNA4
layouts, and only the instruction itself is emulated, written from the
calculator's formulas alone. The network on those builds gives
a48928166420fc45 with all 444 native launches (check_gfx12.py). Every capture,
and both pre_block_check.py and post_block_check.py with `--hsaco`, agree byte
for byte too. So all the code an RDNA4 card runs differently has been executed
and found exact. On gfx1201 the ffwd kernel goes from 216 to 140 VGPRs, with
no `v_permlane16` at all.

**Not checked.** Three things are still open:

- that the card follows the calculator's layout;
- how it rounds inside one accumulation, since RDNA3's is not IEEE and
  RDNA4's is unmeasured. Products and f32 accumulation are otherwise those of
  the RDNA3 path, so an RDNA4 image may differ from RDNA3's in last bits
  only;
- whether it flushes e4m3 subnormals. None of the native kernels runs in the
  flushing mode, and dlssnr_hip.h refuses to build that combination.

On a first RDNA4 run, ab_native.py's image will not match the RDNA3 digest.
compare_images.py is the measure there. A wrong layout would not show up as a
last-bit difference: it would scramble the image.

**The trap the emulation found.** It is the guard trap above, in a new form.
The 1H block stores its halved output under a guard on the lane's token, and
the compiler sank the operand loads, the permutation and the WMMA under it. On
RDNA3 that is harmless, because each lane feeds the WMMA only its own operand.
The permutation reads a neighbour's operand, and that neighbour was inactive
and never loaded it. Only the tokens at the image's edge came out wrong.

Every exchange of the RDNA4 path (`rows_sigma`, `exchange`) is therefore pinned
with an empty `asm volatile` that consumes its result where the kernel asks for
it. An RDNA4 card would have shown the same failure.
