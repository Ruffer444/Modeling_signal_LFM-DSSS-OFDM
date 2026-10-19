from start_sys_params import start_parametrs, clear_console
from sign_create import make_signal, sign_status_table, sign_plt
import matplotlib.pyplot as plt
from create_insert_msg_to_signal import (
    show_encoding, show_capacity,
    transmit, run_link_packet,
    plot_ofdm_constellation, plot_dsss_correlation, plot_chirp_bpsk,
    plot_ber_per_vs_snr,
)
from window import (
    plot_windows, plot_spectrum_windows,
    compare_windows_on_chirp,
)
from effect import (
    run_range_demo, run_multitarget_demo,
    run_doppler_demo, run_2d_map, run_multipath_demo,  
)

if __name__ == "__main__":
    clear_console()
    params = start_parametrs()
    fs = params["fs"]

    # ==========================================================
    # СООБЩЕНИЕ
    # ==========================================================
    message = "Место: 20,20,-20"
    # message = read_user_message()

    # Один пакет на всё: и в консоль, и в сигнал, и в сравнение
    packet, bits = show_encoding(message)
    show_capacity(message)

    # ==========================================================
    # ЭТАП 1: сигналы БЕЗ данных (для сравнения)
    # ==========================================================
    lfm    = make_signal('lfm',  fs=fs, f0=4e3, f1=16e3, T=20e-3)
    hfm    = make_signal('hfm',  fs=fs, f0=4e3, f1=16e3, T=20e-3)
    ofdm   = make_signal('ofdm', fs=fs, N_fft=512, N_cp=64, N_sym=8, mod_order=4)
    dsss   = make_signal('dsss', fs=fs, bits=[1, 0, 1, 1, 0], sf=63)
    lfm_rf = make_signal('lfm',  fs=fs, f0=4e3, f1=16e3, T=20e-3,
                         real=True, f_c=12e3)

    sign_status_table([
        ('ЛЧМ baseband',  lfm),
        ('HFM baseband',  hfm),
        ('OFDM',          ofdm),
        ('DSSS',          dsss),
        ('ЛЧМ passband',  lfm_rf),
    ])

    # ==========================================================
    # ЭТАП 2: сигналы С ДАННЫМИ
    # ==========================================================
    tx_ofdm_data, _ = transmit('ofdm', message, fs=fs,
                               N_fft=512, N_cp=64, N_sym=8, mod_order=4)
    tx_dsss_data, _ = transmit('dsss', message, fs=fs, sf=63, seed=0)
    tx_lfm_data,  _ = transmit('lfm',  message, fs=fs,
                               T=100e-3, f0=4e3, f1=16e3, n_bits_chirp=1000)
    tx_hfm_data,  _ = transmit('hfm',  message, fs=fs,
                               T=100e-3, f0=4e3, f1=16e3, n_bits_chirp=1000)

    # ==========================================================
    # ЭТАП 3: визуализация — общий вид (без данных vs с данными)
    # ==========================================================
    # sign_plt(ofdm,         fs=fs, name='OFDM без данных',       block=False)
    # sign_plt(tx_ofdm_data, fs=fs, name='OFDM с сообщением',     block=False)
    # sign_plt(dsss,         fs=fs, name='DSSS без данных',       block=False)
    # sign_plt(tx_dsss_data, fs=fs, name='DSSS с сообщением',     block=False)
    # sign_plt(lfm,          fs=fs, name='LFM без данных (20 мс)', block=False)
    # sign_plt(tx_lfm_data,  fs=fs, name='LFM с сообщением (100 мс)', block=False)
    # sign_plt(hfm,          fs=fs, name='HFM без данных (20 мс)', block=False)
    # sign_plt(tx_hfm_data,  fs=fs, name='HFM с сообщением (100 мс)', block=False)

    # ==========================================================
    # ЭТАП 4: специализированные графики (где видно биты «в лицо»)
    # ==========================================================
    # plot_ofdm_constellation(message, fs, snr_db=20,
    #                         N_fft=512, N_cp=64, N_sym=8, mod_order=4)

    # plot_dsss_correlation(message, fs, snr_db=20, sf=63, seed=0)

    # plot_chirp_bpsk(message, fs, T=100e-3, f0=4e3, f1=16e3,
    #                 n_bits_chirp=1000, kind='lfm')
    # plot_chirp_bpsk(message, fs, T=100e-3, f0=4e3, f1=16e3,
    #                 n_bits_chirp=1000, kind='hfm')

    # ==========================================================
    # ЭТАП 5: передача при SNR = 20 дБ через ОДИН И ТОТ ЖЕ пакет
    # ==========================================================
    print(f' ЭТАП 5: ПЕРЕДАЧА {message!r} ПРИ SNR=20 ДБ')

    result_ofdm = run_link_packet('ofdm', packet, fs=fs, snr_db=20,
                                  N_fft=512, N_cp=64, N_sym=8, mod_order=4)
    result_dsss = run_link_packet('dsss', packet, fs=fs, snr_db=20,
                                  sf=63, seed=0)
    result_lfm  = run_link_packet('lfm',  packet, fs=fs, snr_db=20,
                                  T=100e-3, f0=4e3, f1=16e3, n_bits_chirp=1000)
    result_hfm  = run_link_packet('hfm',  packet, fs=fs, snr_db=20,
                                  T=100e-3, f0=4e3, f1=16e3, n_bits_chirp=1000)

    # ==========================================================
    # ЭТАП 6: неверный ключ — доказывает, что данные реально в сигнале
    # ==========================================================
    print(' ЭТАП 6: НЕВЕРНЫЙ КЛЮЧ (SNR=100 дБ — идеальный канал)')
    WRONG_KEY = bytes(range(1, 33))
    result_wrong = run_link_packet('ofdm', packet, fs=fs, snr_db=100,
                                   key=WRONG_KEY,
                                   N_fft=512, N_cp=64, N_sym=8, mod_order=4)
    print('  Ожидание: пакет повреждён (тег Poly1305 не сойдётся)')

    # ==========================================================
    # ЭТАП 7: BER / PER vs SNR — по каждому типу
    # ==========================================================
    print()
    print('Строим BER/PER vs SNR... (это займёт пару секунд)')
    # plot_ber_per_vs_snr('ofdm', message, fs,
    #                     N_fft=512, N_cp=64, N_sym=8, mod_order=4)
    # plot_ber_per_vs_snr('dsss', message, fs,
    #                     sf=63, seed=0)
    # plot_ber_per_vs_snr('lfm',  message, fs,
    #                     T=100e-3, f0=4e3, f1=16e3, n_bits_chirp=1000)
    # plot_ber_per_vs_snr('hfm',  message, fs,
    #                     T=100e-3, f0=4e3, f1=16e3, n_bits_chirp=1000)

    # ==========================================================
    # ЭТАП 8: окна
    # ==========================================================
    print(' ЭТАП 8: СРАВНЕНИЕ ОКОН')
    # plot_windows()
    # plot_spectrum_windows(tx_ofdm_data, fs=fs,
    #                       title='OFDM с сообщением: спектр с разными окнами')
    # compare_windows_on_chirp(fs=fs, T=100e-3, f0=4e3, f1=16e3)

    # ==========================================================
    # ЭТАП 9: задержка и дальность
    # ==========================================================
    print(' ЭТАП 9: ЗАДЕРЖКА И ДАЛЬНОСТЬ')
    # --- Часть 1: что будет при fs = 100 кГц (глобальный) ---
    # Демонстрация квантования: 1000 и 5000 м «съедаются» шагом 1500 м.
    print('--- Часть 1: радар на fs=100 кГц (глобальный) — видно ограничение ---')
    # for R_target in (1000, 3000, 5000):
    #     run_range_demo(R_target, fs=fs, window_name='hamming', snr_db=20,
    #                 T=100e-3, f0=4e3, f1=16e3)

    # --- Часть 2: отдельные параметры для радара ---
    # fs=1 МГц, чирп 50…450 кГц, T=10 мс.
    # Шаг = 150 м, разрешение = 375 м — всё три цели видны.
    print()
    print('--- Часть 2: радар с fs=1 МГц, B=400 кГц, T=10 мс ---')
    fs_radar = 1e6
    # for R_target in (1000, 3000, 5000):
    #     run_range_demo(R_target, fs=fs_radar, window_name='hamming', snr_db=20,
    #                 T=10e-3, f0=50e3, f1=450e3)

    # --- Часть 3: две цели рядом ---
    # 1000 и 3000 м — 2000 м друг от друга > ΔR=375 м → видны раздельно.
    # Сильная (1000 м, амп=1.0) и слабая (3000 м, амп=0.05 = −26 дБ).
    # Прямоугольное окно маскирует слабую (боковые −13 дБ), hamming — вытаскивает.
    print()
    print('--- Часть 3: две цели (1000 м — сильная, 5000 м — слабая −26 дБ) ---')
    # run_multitarget_demo(
    #     fs=fs_radar,
    #     targets_m=[1000, 5000],
    #     amps=[1.0, 0.05],
    #     snr_db=20,
    #     T=10e-3, f0=50e3, f1=450e3,
    #     windows=('rect', 'hamming'),
    # )
 


    # ==========================================================
    # ИТОГ
    # ==========================================================
    print(' ИТОГ')
    print(f'  Сообщение : {message!r}')
    print(f'  OFDM      : {"OK" if result_ofdm["ok"] else "СБОЙ"}  '
          f'{result_ofdm.get("text", "")!r}')
    print(f'  DSSS      : {"OK" if result_dsss["ok"] else "СБОЙ"}  '
          f'{result_dsss.get("text", "")!r}')
    print(f'  LFM       : {"OK" if result_lfm["ok"]  else "СБОЙ"}  '
          f'{result_lfm.get("text", "")!r}')
    print(f'  HFM       : {"OK" if result_hfm["ok"]  else "СБОЙ"}  '
          f'{result_hfm.get("text", "")!r}')



    plt.show()