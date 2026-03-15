"""
Debug: live monitor candidate fireball RAM addresses.

Shows RAM bytes at suspected fireball regions in real time.
Non-zero values are highlighted in green.
Shoot fireballs and watch which addresses light up.

No SkipFrame — every NES frame.

Run:
    conda run -n mario python tests/debug_fireball_ram.py
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pygame
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


# Candidate RAM regions for fireballs (from SMBDIS.ASM by doppelganger)
MONITOR_REGIONS = [
    ("Fireball_State  $24-$28", 0x0024, 5),
    ("Fireball_X_Pos  $30-$32", 0x0030, 3),
    ("Fireball_PgLoc  $33-$35", 0x0033, 3),
    ("Fireball_Y_Pos  $34-$36", 0x0034, 3),
    ("Misc_X_Speed    $58-$5C", 0x0058, 5),
    ("Misc_Y_Speed    $70-$74", 0x0070, 5),
    ("Misc_PageLoc    $A4-$A8", 0x00A4, 5),
    ("Misc_X_Pos      $A8-$AC", 0x00A8, 5),
    ("Misc_Y_HiPos    $B8-$BC", 0x00B8, 5),
    ("Misc_Y_Pos      $C4-$C8", 0x00C4, 5),
]


def keys_to_action(keys):
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    jump = keys[pygame.K_UP] or keys[pygame.K_w] or keys[pygame.K_SPACE]
    run = keys[pygame.K_x] or keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

    if right and jump and run:
        return 4
    if right and run:
        return 3
    if right and jump:
        return 2
    if right:
        return 1
    if jump:
        return 5
    if left:
        return 6
    return 0


def main():
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env.reset()

    pygame.init()
    screen = pygame.display.set_mode((900, 520))
    pygame.display.set_caption("Fireball RAM Live Monitor — shoot and watch!")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 13, bold=True)
    font_sm = pygame.font.SysFont("monospace", 11)

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                if event.key == pygame.K_r:
                    env.reset()

        if not running:
            break

        keys = pygame.key.get_pressed()
        action = keys_to_action(keys)
        obs, reward, done, info = env.step(action)
        if done:
            env.reset()

        ram = env.unwrapped.ram

        # -- Draw --
        screen.fill((15, 15, 25))

        # Pixel render (left half)
        pixel_frame = env.unwrapped.screen
        pixel_surf = pygame.surfarray.make_surface(np.transpose(pixel_frame, (1, 0, 2)))
        pixel_surf = pygame.transform.scale(pixel_surf, (400, 375))
        screen.blit(pixel_surf, (4, 24))

        # Status
        status = (
            f"x:{info.get('x_pos',0):4d}  status:{info.get('status','?'):8s}  "
            f"powerup:0x{ram[0x0756]:02X}  action:{SIMPLE_MOVEMENT[action]}"
        )
        screen.blit(font.render(status, True, (200, 200, 200)), (4, 4))

        # RAM monitor (right side)
        rx = 420
        ry = 10
        screen.blit(font.render("LIVE RAM MONITOR", True, (255, 255, 100)), (rx, ry))
        screen.blit(font_sm.render("(green = non-zero)", True, (100, 200, 100)), (rx + 180, ry + 2))
        ry += 24

        for label, start_addr, count in MONITOR_REGIONS:
            # Region label
            screen.blit(font_sm.render(label, True, (180, 180, 180)), (rx, ry))
            ry += 16

            # Byte values
            for i in range(count):
                addr = start_addr + i
                val = int(ram[addr])
                if val != 0:
                    color = (0, 255, 0)       # GREEN = non-zero
                    bg_color = (0, 60, 0)
                else:
                    color = (80, 80, 80)       # gray = zero
                    bg_color = None

                text = f"${addr:04X}={val:3d}"
                txt_surf = font_sm.render(text, True, color)

                bx = rx + i * 90
                if bg_color:
                    pygame.draw.rect(screen, bg_color, (bx - 2, ry - 1, 86, 15))
                screen.blit(txt_surf, (bx, ry))

            ry += 20

        # Also show a summary line of ALL non-zero misc addresses
        ry += 10
        screen.blit(font_sm.render("ALL non-zero in $20-$40:", True, (255, 200, 100)), (rx, ry))
        ry += 16
        nz_parts = []
        for a in range(0x20, 0x40):
            v = int(ram[a])
            if v != 0:
                nz_parts.append(f"${a:02X}={v}")
        nz_text = "  ".join(nz_parts) if nz_parts else "(all zero)"
        screen.blit(font_sm.render(nz_text[:80], True, (255, 200, 100)), (rx, ry))
        if len(nz_text) > 80:
            ry += 14
            screen.blit(font_sm.render(nz_text[80:160], True, (255, 200, 100)), (rx, ry))

        pygame.display.flip()
        clock.tick(30)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
