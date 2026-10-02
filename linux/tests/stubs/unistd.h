#pragma once
#include "prelude.h"
int getpid(void);
int unlink(const char *path);
ssize_t readlink(const char *path, char *buffer, size_t size);
