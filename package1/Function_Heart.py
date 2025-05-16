import numpy as np
import numpy as np
from .Function2D import Function2D
import math as math

class Function_Heart(Function2D):
    def __init__(self, name, xdomainstart,xdomainend,ydomainstart,ydomainend):
        self.name = name
        self.xdomainstart = xdomainstart
        self.xdomainend = xdomainend
        self.ydomainstart = ydomainstart
        self.ydomainend = ydomainend

    def evaluate(self,x, y):
        """
        Gibt die Höhe im Bereich des Herzens zurück.
        Höhe = 0 außerhalb, 1 im Zentrum.
        Der Anstieg ist möglichst gleichmäßig zur Mitte hin.
        """
        # Herzfunktion (implizit)
        f = (x**2 + y**2 - 1)**3 - x**2 * y**3

        # Innerhalb des Herzens: f < 0
        inside = f < 0

        # Wir definieren die Höhe als -f (also wie tief der Punkt "im Herz" steckt)
        # Dabei sorgen wir mit einer nichtlinearen Skalierung (sqrt) für gleichmäßige Steigung
        height = np.zeros_like(f)
        depth = -f[inside]  # Nur die inneren Punkte
        norm_depth = depth / np.max(depth)  # auf [0, 1] skalieren
        height[inside] = np.sqrt(norm_depth)  # gleichmäßiger Verlauf

        return height
