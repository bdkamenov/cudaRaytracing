#ifndef __UTIL_H__
#define __UTIL_H__

#include "constants.h"

#include <stdlib.h>
#include <math.h>
#include <string>

std::string upCaseString(std::string s); //!< returns the string in UPPERCASE
std::string extensionUpper(const char* fileName); //!< Given a filename, return its extension in UPPERCASE

/// a simple RAII class for FILE* pointers.
class FileRAII {
    FILE* held;
public:
    FileRAII(FILE* init): held(init) {}
    ~FileRAII() { if (held) fclose(held); held = NULL; }
    FileRAII(const FileRAII&) = delete;
    FileRAII& operator = (const FileRAII&) = delete;
};

#endif // __UTIL_H__
