#pragma once
#include "prelude.h"
#define RTLD_NOW 2
#define RTLD_GLOBAL 0x100
#define RTLD_NOLOAD 4
#define RTLD_DEFAULT ((void *)0)
void *dlopen(const char *file, int mode);
void *dlsym(void *handle, const char *name);
char *dlerror(void);
