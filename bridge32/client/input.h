// DirectInput hooks, so that the menu can have the mouse of a game that reads it through DirectInput.
#pragma once

namespace input {

// Hooks DirectInput8Create. Call it before the game can have called it, as early as possible.
bool install();

} // namespace input
