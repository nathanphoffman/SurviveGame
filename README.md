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
| Esc | Cancel aiming, otherwise quit |

## Layout

- `src/main.rs` – everything: terrain generation, character model and animation, camera, input.
