# Third-party work

## RenoDX DLSS 5 add-on

The runtime's colour handling (`runtime/core/shaders.h`) follows the design of
the DLSS 5 add-on in RenoDX by clshortfuse
(https://github.com/clshortfuse/renodx): the network is shown a display encoded
proxy of the frame, and its answer is composed back as a two-branch luminance
ratio against the frame, with the network's hue restored in OkLab, a blend
between a luminance-only result and the network's colour, and out-of-gamut
colours moved towards neutral. We learned of it through the OptiScaler-DLSSNR
fork that adopted it. It is their design, reimplemented here; no code was
copied.

## OkLab

The OkLab conversion uses the matrices published by Bjorn Ottosson
(https://bottosson.github.io/posts/oklab/), public domain or MIT.

## ZLUDA

`third_party/ZLUDA` is a submodule and keeps its own MIT / Apache-2.0 licence.
