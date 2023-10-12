# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:23:00 2023

@author: UBG5KOR
"""

from picarx import Picarx
import time


if __name__ == "__main__":
    try:
        px = Picarx()
        px.backward(05)
        time.sleep(1.5)

        px.forward(0)
        time.sleep(1)

    finally:
        # px.forward(0)
        px.stop()
        time.sleep(.2)