from start_sys_params import start_parametrs, clear_console
from sign_create import make_signal, sign_status_table, sign_plt
from create_insert_msg_to_signal import (
    show_encoding, show_capacity,
    transmit, run_link_packet,
    plot_ofdm_constellation, plot_dsss_correlation, plot_chirp_bpsk,
    plot_ber_per_vs_snr,
)
import matplotlib.pyplot as plt

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