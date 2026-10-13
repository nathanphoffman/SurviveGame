# SurviveGame

A small RTS/MOBA-style prototype in Rust using [Bevy](https://bevy.org) 0.16.
One always-selected space marine in a small grass and mud arena, with RTS camera controls.

## Run

```
cargo run
```

The first build takes a few minutes (Bevy is large); rebuilds are quick.

Requires a Vulkan-capable GPU and X11 (works under XWayland). On Linux you need
the usual ALSA/X11 dev packages installed.

## Controls

| Input | Action |
| --- | --- |
| Right-click / left-click | Move the character (hold right-click to keep following the cursor) |
| W A S D / arrow keys | Pan camera (hold Shift to go faster) |
| Mouse at screen edge | Edge scrolling |
| Middle-mouse drag | Grab and pan the ground |
| Mouse wheel | Zoom in / out |
| Q / E | Rotate camera |
| Space (hold) | Recenter camera on the character |
| Click a droid | Attack it (the marine also auto-fires at the nearest droid in range when standing still) |
| 1 or click the ability icon | Aim **Suppressive Barrage**; left-click the ground to fire, right-click/Esc to cancel |
| 2 or click the jetpack icon | Aim **Jetpack**; left-click to fly there (short range), right-click/Esc to cancel |
| Esc | Cancel aiming, otherwise quit |

## Layout

- `src/main.rs` – everything: terrain generation, character model and animation, camera, input.

## Waves

Enemies come in waves with a 10 second break between them, and each wave is bigger.
From wave 3 on, big armored droids join in (more of them each wave). You regenerate
1 health per second and 6 energy per second. If you die, the game restarts at wave 1.

## Ally

A green-armored clone of your marine fights with you, controlled by a simple AI: it
follows you, shoots the nearest enemy in range, and regroups if it falls behind. Enemies
target whichever of you is closest. If it dies, a new one arrives after 15 seconds.
