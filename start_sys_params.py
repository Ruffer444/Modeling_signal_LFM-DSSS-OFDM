import os
# import sys

def clear_console():
    # Очистка консоли
    os.system("cls" if os.name == "nt" else "clear")

def start_parametrs():
    print('Параметры создаются...')
    params = {}
    params["fc"] = 1000  # Центральная частота (Гц)
    params["bandwidth"] = 3150  # Полоса сигнала (Гц)
    params["T_sym"] = 0.0075  # Длительность символа (с)
    params["fs"] = 100000  # Частота дискретизации (Гц)
    print("Параметры сигнала загружены (+) ")
    print("Параметры созданы можно начать моделирование :) ")

    return params