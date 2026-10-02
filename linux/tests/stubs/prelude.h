/* Stand-ins that let the compiler here read the Linux branch of bridge/core.c.
 * Nothing in them runs: the point is that the types and calls agree. */
#pragma once
#include <stddef.h>
#include <stdint.h>
typedef intptr_t ssize_t;
#define __attribute__(x)
