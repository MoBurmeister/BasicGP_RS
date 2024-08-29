import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import math

# Materialabhängige Konstanten
lambda_th = 24.3 # Wärmeleitfähigkeit des Materials; 
                    # H13 Vollmaterial: 24.3 W/(m*K), Quelle: https://www.matweb.com/search/datasheet_print.aspx?matguid=e30d1d1038164808a85cf7ba6aa87ef7
                    # H13 PBF: 22.9 W/(m*K)
                    # AlSi10Mg: 140 W/(m*K)
c_p = 389.36 # spez. Wärmekapazität in J/(kg*K)
                    # H13 Vollmaterial: 460 J/(kg * K), Quelle: https://www.matweb.com/search/datasheet_print.aspx?matguid=e30d1d1038164808a85cf7ba6aa87ef7
                    # H13 PBF: 389.36 J/(kg * K)
alpha = 0.22  # Absorptionskoeffizient des Lasers; 
                    # H13 Vollmaterial geschliffen
                    # H13 PBF: 0.22
                    # AlSi10Mg: 0,45
rho = 7800 # Dichte in kg/m³
k = lambda_th/(rho*c_p) # Thermal Diffusivity


# Parameter für Temperatur
T_start = 298 # Raumtemperatur in K
T_Haerten = 1010+273+20 # Zieltemperatur für Härten in K + ca. 20-40K Obergrenze
# also: 1303 max

# T_Lösungsglühen = 800+273+20 für AlSi10Mg

# Parameter für Simulation (Schritte + Geometrie) --> hiermit spielen für höhere Auflösung der Simulation

# Resolution can be adjusted here:
resolution = 10000
x_start, x_end, dx = -0.02, 0.01, 0.001
y_start, y_end, dy = -0.005, 0.005, 0.001
z = 0 # hier kann eingestellt werden, ob Temperatur an Oberfläche oder in bestimmten Abstand berechnet werden soll
t = np.logspace(-6, 6, resolution) # Zeitliche Parameter

# Laserparameter --> das hier sind die Eingangsgrößen, die wir varrieren können
laser_pwr = [349.4612]   # Laserleistung (W); min 20; max 400
laser_speed = 0.2 #0.015 #200e-3 / 60  # Verfahrgeschwindigkeit (m/s); min 200 mm/s; max 3000 mm/s
laser_width = 0.000686#3.1225e-04#500e-6  # Strahldurchmesser (m); min 83µm, max 1000µm (ggf. geht auch noch mehr)

# # Berechnung von Fokusabstand --> nur für Einstellung der Anlage notwendig, nicht relevant für die Simulation
# def berechne_fokusabstand(w, w0, zR):
#     return zR * math.sqrt((w / w0) ** 2 - 1)

# # Gegebene Werte
# w0 = 83e-6  # Strahldurchmesser im Fokus (m)
# zR = 3.62e-3  # Rayleighlänge (m)
# w_gewuenscht = laser_width  # Gewünschter Strahldurchmesser (m)
# z_notwendig = berechne_fokusabstand(w_gewuenscht, w0, zR)
# print("Der notwendige Fokusabstand für einen Strahldurchmesser von", w_gewuenscht * 1e6, "µm beträgt:", z_notwendig * 1000, "mm")


################################################
# Simulation 
################################################

# Initialisierung des Temperaturfeldes
T = np.zeros((int(abs((x_start - x_end) / dx)), int(abs((y_start - y_end) / dy))))

# Temperatur Funktion nach Jarwitz et al. 2017
def Temp_fkt(x, y, z, laser_power):
    T_Integral = np.exp(-(((x + laser_speed * t) ** 2 + y ** 2) / ((laser_width ** 2 / 8) + 4 * k * t) + (z ** 2 / (4 * k * t)))) / (np.sqrt(t) * ((laser_width ** 2 / 8) + 4 * k * t))
    T_Int_Loesung = integrate.simpson(T_Integral, t)
    T_Jarwitz = ((alpha * laser_power) / (np.pi ** 1.5 * np.sqrt(lambda_th * rho * c_p))) * T_Int_Loesung + 293
    return T_Jarwitz

for laser_power in laser_pwr:
    x1, y1 = x_start, y_start
    time_vector = []
    i = j = 0

    while x1 < x_end:
        y1 = y_start
        while y1 < y_end:
            T[i][j] = Temp_fkt(x1, y1, z, laser_power)
            y1 += dy
            j += 1
        i += 1
        time_vector.append(x1 / laser_speed)
        x1 += dx
        j = 0

    T_verlauf = T[:, int(abs((y_start - y_end) / dy) / 2)]
    print("T_max = ", T.max(), "°C")
    print("T_diff = ", T.max() - T_Haerten, "°C")



    #time_haerten = sum(1 for k in range(int(abs((x_start - x_end) / dx))) if T_verlauf[k] > T_Haerten) * (dx / laser_speed)

    # Calculate the number of steps where the temperature exceeds T_Haerten
    steps_above_threshold = sum(1 for k in range(int(abs((x_start - x_end) / dx))) if T_verlauf[k] > T_Haerten)

    # Print the number of steps
    print(f"Number of steps where temperature exceeds T_Haerten: {steps_above_threshold}")

    # Calculate the hardening time
    time_haerten = steps_above_threshold * (dx / laser_speed)
    print(int(abs((x_start - x_end) / dx)))
    print("t (Härten) =", time_haerten)

    

    # --> Ziel: Zieltemperatur mit Intervall +20K für möglichst lange Zeit erreichen

    # # Temperaturverlauf
    # plt.plot(time_vector, T_verlauf)
    # plt.xlabel('Zeit in s')
    # plt.ylabel('Temperatur in K')
    # plt.title('Temperaturverlauf auf der Oberfläche')
    # plt.show()

    # # Temperaturfeld
    # plt.figure(1)
    # plt.imshow(T, interpolation='hanning', cmap='rainbow')
    # plt.xlabel('y-Achse')
    # plt.ylabel('x-Achse')
    # plt.colorbar(label='Temperatur in K')
    # plt.show()
