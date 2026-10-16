import numpy as np

from scipy.signal import chirp
from start_sys_params import start_parametrs, clear_console
from sign_create import make_signal,sign_status_table, sign_plt
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Очищаем консоль перед запуском
    clear_console()
    
    print('Начало работы алгоритма')
    start_parametrs()

    lfm  = make_signal('lfm',  f0=4e3, f1=16e3, T=20e-3)
    hfm  = make_signal('hfm',  f0=4e3, f1=16e3, T=20e-3)
    ofdm = make_signal('ofdm', N_fft=512, N_cp=64, N_sym=8, mod_order=4)
    dsss = make_signal('dsss', bits=[1,0,1,1,0], sf=63)
    lfm_rf = make_signal('lfm', f0=4e3, f1=16e3, T=20e-3,
                         real=True, f_c=12e3)

    sign_status_table([
        ('ЛЧМ baseband',  lfm),
        ('HFM baseband',  hfm),
        ('OFDM',          ofdm),
        ('DSSS',          dsss),
        ('ЛЧМ passband',  lfm_rf),
    ])
    sign_plt(lfm)

    plt.show()
