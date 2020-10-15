#ifndef __SDL_H__
#define __SDL_H__

#include <SDL2/SDL.h>
#undef main

namespace CXX {


/// a simple RAII class for SDL library
class SdlObject
{
public:
    SdlObject(const SdlObject&) = delete;
    SdlObject(SdlObject&&) = delete;
    ~SdlObject();
    SdlObject& operator=(const SdlObject&) = delete;
    SdlObject& operator=(SdlObject&&) = delete;

    static SdlObject& instance();

    void closeGraphics(void);
    template<typename T>
    void displayVFB(T* vfb); //!< displays the VFB (Virtual framebuffer) to the real one.
    void waitForUserExit(void); //!< Pause. Wait until the user closes the application
    int frameWidth(void); //!< returns the frame width (pixels)
    int frameHeight(void); //!< returns the frame height (pixels)

private:
    SdlObject(int frameWidth, int frameHeight);

    SDL_Window* window_ = nullptr;
    SDL_Surface* screen_ = nullptr;
    const char* error_ = nullptr;
    SDL_GLContext gl_context_ = nullptr;
};

/// displays a VFB (virtual frame buffer) to the real framebuffer, with the necessary color clipping
template<typename T>
void SdlObject::displayVFB(T* vfb)
{
    int rs = screen_->format->Rshift;
    int gs = screen_->format->Gshift;
    int bs = screen_->format->Bshift;

    for (int y = 0; y < screen_->h; y++)
    {
        Uint32* row = (Uint32*)((Uint8*)screen_->pixels + y * screen_->pitch);
        for (int x = 0; x < screen_->w; x++)
        {
            int index = screen_->w * y + x;
            row[x] = vfb[index].toRGB32(rs, gs, bs);
        }
    }
    SDL_FreeSurface(screen_);
    SDL_UpdateWindowSurface(window_);
}
}

#endif // __SDL_H__
