# Design

How DLSS 5 Neural Rendering reaches a game on an AMD GPU in this project, and
why each piece has the shape it has.

## The pieces

```
game ── OptiScaler (fork, optiscaler/) ──┐
                                         │  C API (runtime/include/dlss5nr.h)
                                         ▼
                              dlss5nr_runtime.dll (runtime/core)
                               │            │
             D3D12: capture,   │            │  worker thread
             compose, exposure │            ▼
                               │     nvngx.dll (runtime/ngx) ── nvngx_dlssnr.dll (NVIDIA)
                               │            │
                               │            ▼
                               │     nvcuda.dll (runtime/gate) ── native kernels (kernels/)
                               │            │
                               │            ▼
                               │     ZLUDA (third_party/ZLUDA) ── HIP
                               ▼
                         shared textures (D3D12 <-> HIP)
```

- **The network** is NVIDIA's own `nvngx_dlssnr.dll`, version 310.8.0.0,
  shipped as it is in the release package only; whoever packages keeps a copy
  in `third_party/`, which is not committed. Nothing derived
  from it is part of this repository: no PTX, no weights, no captures, no
  translated cache (that one is built for a release, and only shipped).
- **Its host code** runs unchanged. `nvngx.dll` stands in for NVIDIA's NGX
  core and drives the snippet through its CUDA entry points.
- **ZLUDA** is RedDukeDev's fork, and translates the network's PTX for AMD.
- **The gate** is a thin `nvcuda.dll` in front of ZLUDA. It replaces the hot
  translated kernels with the native HIP kernels in `kernels/`. Each native
  kernel is byte-identical to the translation it replaces, and falls back to
  that translation whenever something does not match.
- **The runtime** owns everything between a D3D12 frame and the network, and
  exposes a small C API. A ReShade add-on could call it just as OptiScaler
  does.
- **OptiScaler** finds the game's colour, depth and motion vectors, which is
  the hard part of integrating with any game. The fork adds only the calls
  into the runtime, its settings and its menu.

## Why asynchronous, and why a residual

OptiScaler records the upscaler into the game's own command list, inside
`IFeature_Dx12::Evaluate`. The network runs on a HIP stream, and HIP work
cannot be inserted in the middle of a D3D12 command list the game will submit
later. ROCm cannot import a D3D12 fence either, so the only point where the two
APIs can meet is the CPU. Waiting on the CPU inside the game's frame would
stall the game for the whole cost of the network.

So the network runs beside the game rather than inside its frame:

1. **Capture.** Inside the game's command list, a compute pass writes colour,
   depth and motion vectors into one of the runtime's shared slots. While
   doing so it shows the network a display encoded proxy of the game's colour
   (see *Colour* below), and resamples to the network's resolution. The motion
   vectors lead back to the previous capture (see *Tracking* below).
2. **Submit.** At `Present`, the runtime signals a fence on the game's queue.
3. **Evaluate.** A worker thread waits for that fence and runs the network on
   the slot. It synchronises the HIP context and then marks the slot's output
   ready. None of this blocks the game.
4. **Compose.** On every frame the runtime composes the latest ready result
   into the image, again in the game's command list, through a *residual*:
   network output minus network input, both in the proxy's linear space. The
   residual is found through the displacement map of its capture, so it lands
   where its pixels are now, laid on the current frame's own proxy, and
   brought back as a luminance ratio against the current frame.

A residual, rather than the network's image, is what makes this work. The game
image stays the game's own current frame, and the network only contributes the
change it makes, which follows the scene however late it arrives. When the
network is fast enough for every frame, the composition is simply one frame
late. When it is not, it updates at its own rate and the game's frame rate is
left alone.

### Tracking

A result arrives several frames after the frame it was captured in: the
network's evaluation time, in frames. Composed where the network put it, it
would trail behind everything that moved in the meantime -- ghosting, by the
distance the scene travelled during one evaluation.

So every capture keeps a *displacement map*: for each pixel of the current
frame, where the same point of the scene was in the frame of the capture. It
starts at zero in that frame and is moved on by every frame after it, along
the game's motion vectors: a pixel's displacement is its motion vector plus the
displacement, one frame earlier, of the point it came from. A point that was
outside the image, or behind something else then -- its depth one frame earlier
disagrees with its depth now -- is marked as gone. The composition samples
the residual through the map of its capture.

How late a result is decides how much of this there is to do, so the capture
that waits for the network is replaced on every frame: an evaluation starts
from the newest frame there is, and a result is one evaluation late, not two.

#### What the network never saw

A pixel marked as gone has no residual of its own: whatever moved off it was
in front of it in the capture. Left untouched, such pixels are a patch the
shape of that object, trailing behind it until the next result. In order:

1. **A borrowed displacement.** A map cannot say where a point was once it has
   been hidden in any frame on the way. But the surface around it was seen and
   moved the same way, so the nearest pixel with a displacement, at this
   pixel's depth, lends its own. The residual found there is taken only if
   that capture showed this pixel's surface at that spot: the capture's depth
   is kept in the residual's alpha, and compared with the depth now.
2. **The result before.** Its residual and its map are kept. What the newer
   capture did not see, the older one may have: anything that moved further
   than its own width in between leaves behind what the older capture showed.
   It is asked through its own displacement, then through a borrowed one.
3. **The network's gain.** Failing both, the pixel gets what the network did
   to the light around it: its output over its input, each averaged over
   16 x 16 pixels, from the nearest pixels on the same surface. A gain rather
   than a difference, because the network relights, and not its detail, which
   carried over shows as a faint copy of where it came from.

An answer of the network for that very point, even an older one, beats a
guess. What is left in the end is a patch with the right light and none of
the network's detail, for as long as the next result takes.

`wait_for_network` avoids most of it at the cost of the frame rate:
`dlss5nr_present` waits for the frame's own result, every frame is composed
with a result one frame old, and the game runs at the network's rate.

`wait_inside` (experimental) avoids all of it: the game's command list itself
waits for the network, so that its answer is composed into the frame it was made
from, and the game runs at the network's rate. It is slower than the other, and
here is why: the host learns that the capture is in place from a word the GPU
writes, and that write stays in the GPU's cache for between ten and a couple of
hundred milliseconds (every way of writing it that was tried: the command
processor's marker, a shader's store or atomic, a copy), until the reads of the
wait itself push it out; the network only starts when the host has seen it.
Waiting on the GPU for the capture, from the network's side, would avoid it
(not done). Every
frame is captured, so the network's own history is kept (the capture before is
the previous frame, and the game's motion vectors lead back to it): without it
every frame's detail is decided anew and flickers. The game may record the next
frame while the network works on this one, which hides the time the game itself
takes; frames are answered in order, and none is passed over.

How the wait works. After the capture the list has the command processor write
the capture's number into host memory (`WriteBufferImmediate`, which reaches
memory at once, where a shader's write waits in the GPU's cache), and then a
compute pass that reads lines of host memory, one after the other, until it
finds that number. The worker thread, which was woken when the capture was
recorded and looks for the number, runs the network and then writes the number
into every line the pass has yet to read. The reading has to go from line to
line: what a GPU has read stays in its cache, a word it keeps looking at is
never seen to change, and only a line it has not read before comes from memory
as it is then (measured here; every attempt with one word failed). A line is
128 bytes, and a few frames' worth of them are kept apart so that none is read
twice while still cached. A few microseconds pass between two looks, so that the
lines, and the host memory they take, are not too many for the longest wait.
The pass has a limit of lines, about three times what the network last took,
so that a wait that ran out ends: the frame then goes on without the effect. The number of lines read is written back to the host;
three waits in a row that ran out, and the frame goes back to waiting at
present for ten seconds. The time one look takes (about a microsecond) is
measured when it is first needed (two lengths, so that what a submission takes
is not counted), together with a test that the answer is seen, and corrected
as the waits go on.

When the images are copied through host memory (Wine) the frame's own list
copies the capture to the staging buffers and, after the wait, the answer from
them: the worker only moves them between those and the network's arrays. If
the answer is not seen (the test above), or the list does not allow it, the
frame waits at present, and is composed with a result one frame old.

The capture's motion vectors lead back to the capture the network's own
history holds, through that capture's map (see *What the network is told*).

### Slots

Every slot holds one frame of network input (colour, depth, motion) and one
network output, as shared textures. At any time a slot is in exactly one of
these states:

| state | owner | meaning |
|---|---|---|
| free | nobody | may be captured into |
| captured | game GPU | its capture was recorded in frame N |
| evaluating | worker | the network is reading and writing it |
| ready | composer | holds a finished result |

A capture only takes a free slot. If none is free, the frame is simply not
captured: the network is busy, and it picks up a later frame. The composer
always uses the newest ready slot. An older ready slot returns to free only
once the game's queue has finished every frame that composed from it, as the
`Present` fence tells.

### Two places to compose

- **After the upscaler** (default). The network runs on the upscaled image,
  at output resolution or at a configurable fraction of it. Depth and motion
  vectors are resampled to that resolution during capture. The residual is
  added to the upscaler's output.
- **Before the upscaler** (optional, *pre-upscaling*). The network runs at the
  game's render resolution, on the upscaler's own inputs, and the residual is
  added to the colour the upscaler receives. This is far cheaper, since the
  network costs in proportion to its pixels. The price is quality: the
  upscaler then accumulates an image the network has already altered.

## Colour

The network was trained on finished SDR pictures, sRGB encoded in [0, 1]. A
game hands over scene-referred linear light instead, on a scale of its own, and
the network's answer has to go back onto that scale. How this is done follows
RenoDX's DLSS 5 add-on by clshortfuse, as documented by the OptiScaler-DLSSNR
fork that adopted it, reimplemented here (see `THIRD_PARTY.md`).

- **Encode.** The frame is divided by its white point, the game's pre-exposure
  over its exposure (the game's exposure texture when there is one, else one
  measured from the image), so paper white lands at 1. Luminance above 0.75
  is rolled off by a soft knee rather than clipped, keeping the hue, and the
  result is sRGB encoded: an sRGB curve, not a 2.2 power, because that is what
  an SDR game buffer carries. There is no tone curve beyond the knee. The game
  tone maps its picture later, and tone mapping it here as well shows the
  network a doubly compressed, flat image that it answers weakly. A frame the
  game declares display encoded already is shown as it is.
- **Compose as a ratio, not a sum.** Adding the network's difference back
  loses what it does in highlights, where the knee compressed the frame, and
  nothing bounds the sum, so colour can run away. Instead the residual is laid
  on the current frame's proxy (scaled as a whole so it stays inside the unit
  cube, which keeps its direction), and the answer is rescaled to the frame's
  luminance: below the proxy's luminance by the frame's own ratio, above it
  with the headroom the knee took away added back on top. The hue is taken
  from the network's answer in OkLab.
- **Bounded.** The change is measured as a luminance ratio against the frame,
  floored so a nearly black pixel cannot double from a tiny change and boil,
  and clamped both ways by `max_ratio`. Colour strength blends from the game's
  own hue to the network's; above 1 the network's OkLab chroma is increased,
  and anything pushed out of gamut is moved towards grey of the same
  luminance, never clipped per channel.

A synchronous implementation measured that filtering the network's answer over
time is a dead end: the network re-decides its detail with every framing. The
runtime does not do that either. Tracking only carries one answer to where its
pixels moved while the next is being computed, and every fresh result replaces
the previous one outright.

`DLSS5NR_COMPOSITION_REPLACE` skips the ratio and returns the network's answer
laid on the frame's proxy. With a display encoded frame and a fresh result, that
is the network's own image, which is how the test harness checks it against the
reference.

## What the network is told

Neural rendering keeps a history of its own, blended with each new frame along
its motion vectors; its parameters confirm it (a history blend model, and a
temporal history it resets itself when its controls change). The runtime
evaluates only some frames, so the one-frame motion vectors of the game would
not describe the step between two evaluations. Each capture gets instead the
displacement to the previous capture, however many frames ago that was, and
the history carries on. The reset flag is set only when that step is unknown:
after a cut, on the first capture, and when the network failed on the capture
the motion vectors lead back to.

This is off by default (`network_history`). With it the network's answer for
a frame is no longer the one it gives that frame alone, the one every check
here is made against; it is the same on the native and the translated
kernels, byte for byte, but how it looks across evaluations a tenth of a
second apart has not been judged in a game yet.

User interface correction stays off. It needs the game's UI in separate
textures, which OptiScaler does not have, and without them the network
composites against nothing and returns black.

## Licences

`kernels/`, `runtime/` and `docs/` are MIT. `optiscaler/` is a fork of
OptiScaler and is GPL-3.0, like its upstream. `third_party/ZLUDA` keeps its
own MIT/Apache-2.0 licence.
