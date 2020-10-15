#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include <string>
using namespace std;

string upCaseString(string s)
{
    for (int i = 0; i < (int) s.length(); i++)
        s[i] = toupper(s[i]);
    return s;
}

string extensionUpper(const char* fileName)
{
    int l = (int) strlen(fileName);
    if (l < 2) return "";

    for (int i = l - 1; i >= 0; i--) {
        if (fileName[i] == '.') {
            string result = "";
            for  (int j = i + 1; j < l; j++) result += toupper(fileName[j]);
            return result;
        }
    }
    return "";
}