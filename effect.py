import numpy as np
import matplotlib.pyplot as plt

from sign_create import make_signal
from window import get_window

C_LIGHT = 3e8


# ============================================================
# БАЗОВЫЕ ЭФФЕКТЫ КАНАЛА
# ============================================================
# Каждый эффект — отдельная функция. Их можно комбинировать.
#   y = apply_doppler(apply_delay(x, n), fs, f_d)
# Это удобно для будущих сценариев (доплер + многолучёвость + шум).


def apply_delay(x, n_samples):
    # Чистая задержка на n_samples (нули в начале).
    # Аналог «сигнал пришёл на n отсчётов позже».
    if n_samples <= 0:
        return x.copy()
    y = np.zeros(len(x) + n_samples, dtype=complex)
    y[n_samples:] = x
    return y


def apply_doppler(x, fs, f_d_hz):
    # Доплеровский сдвиг: умножение на exp(j·2π·f_d·t).
    # Для цели со скоростью v: f_d = 2·v·f_c/c, знак зависит от
    # «приближается / удаляется». Знак f_d_hz задаётся вручную.
    t = np.arange(len(x)) / fs
    return x * np.exp(1j * 2 * np.pi * f_d_hz * t)


def apply_awgn(x, snr_db, seed=0):
    # AWGN по заданному SNR. Локальная копия из crypto-модуля —
    # специально без импорта, чтобы effect.py не зависел от
    # файла про данные. В реальном проекте вынести в общий utils.
    rng = np.random.default_rng(seed)
    p_sig = np.mean(np.abs(x) ** 2)
    p_noise = p_sig / 10 ** (snr_db / 10)
    noise = np.sqrt(p_noise / 2) * (
        rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    return x + noise


# ============================================================
# ДАЛЬНОСТЬ: задержка + согласованный фильтр
# ============================================================

def delay_by_range(x, fs, R_m, c=C_LIGHT):
    # Задержка tau = 2R/c (сигнал туда-обратно), дискретизованная по fs.
    # Возвращает (y, tau, n_delay) — сам сигнал, точную задержку и число отсчётов.
    tau = 2 * R_m / c
    n = int(round(tau * fs))
    y = np.zeros(len(x) + n, dtype=complex)
    y[n:] = x
    return y, tau, n


def matched_filter(rx, ref, window_name='rect'):
    # Согласованный фильтр: корреляция rx с опорой ref, взвешенной окном.
    # mode='full' — выход длиннее сигнала, пик при задержке 0 на индексе len(ref)-1.
    w = get_window(window_name, len(ref))
    return np.correlate(rx, ref * w, mode='full')


def mf_to_range_axis(mf_len, ref_len, fs, c=C_LIGHT):
    # Ось дальности для выхода MF. Пик (ref_len-1) ↔ R=0.
    return (np.arange(mf_len) - (ref_len - 1)) * c / (2 * fs)


def estimate_range(mf, ref_len, fs, c=C_LIGHT):
    # Пик MF → оценка дальности. Возвращает (R_est_м, idx_пика).
    idx_peak = int(np.argmax(np.abs(mf)))
    R_est = (idx_peak - (ref_len - 1)) * c / (2 * fs)
    return R_est, idx_peak


# ============================================================
# ДЕМОНСТРАЦИИ ПО ДАЛЬНОСТИ
# ============================================================

def run_range_demo(R_target_m, fs, window_name='hamming', snr_db=20,
                   T=100e-3, f0=4e3, f1=16e3):
    # Один цикл: чирп → отражение от цели → шум → MF → оценка R → график.
    tx = make_signal('lfm', fs=fs, f0=f0, f1=f1, T=T)

    # Отражение с задержкой
    rx, tau, n_delay = delay_by_range(tx, fs, R_target_m)

    # Шум
    rx = apply_awgn(rx, snr_db, seed=0)

    # Сжатие + оценка
    mf = matched_filter(rx, tx, window_name=window_name)
    R_est, idx_peak = estimate_range(mf, len(tx), fs)

    print()
    print(f'--- Дальность (окно={window_name}, SNR={snr_db} дБ) ---')
    print(f'  Задано R              = {R_target_m/1e3:.3f} км  '
          f'(tau={tau*1e6:.1f} мкс, {n_delay} отсч.)')
    print(f'  Оценено R             = {R_est/1e3:.3f} км')
    print(f'  Ошибка                = {R_est - R_target_m:.1f} м')
    print(f'  Квантование (1 отсч.) = {C_LIGHT/(2*fs):.1f} м')

    # График
    R_axis = mf_to_range_axis(len(mf), len(tx), fs)
    mf_abs = np.abs(mf)
    mf_db = 20 * np.log10(mf_abs / mf_abs.max() + 1e-12)

    lo = max(0, idx_peak - 100)
    hi = min(len(mf), idx_peak + 100)

    fig, ax = plt.subplots(2, 1, figsize=(11, 6))
    ax[0].plot(R_axis[lo:hi] / 1e3, mf_db[lo:hi])
    ax[0].axvline(R_target_m / 1e3, color='r', linestyle='--',
                  label=f'R_true = {R_target_m/1e3:.1f} км')
    ax[0].set_xlabel('Дальность, км'); ax[0].set_ylabel('|MF|, дБ')
    ax[0].set_title(f'Сжатие чирпа (окно={window_name}, SNR={snr_db} дБ)')
    ax[0].set_ylim(-100, 5); ax[0].grid(alpha=0.3); ax[0].legend()

    ax[1].plot(R_axis[lo:hi] / 1e3, mf_abs[lo:hi] / mf_abs.max())
    ax[1].axvline(R_target_m / 1e3, color='r', linestyle='--',
                  label=f'R_true = {R_target_m/1e3:.1f} км')
    ax[1].set_xlabel('Дальность, км'); ax[1].set_ylabel('|MF| норм.')
    ax[1].set_title('Главный лепесток (линейная шкала)')
    ax[1].grid(alpha=0.3); ax[1].legend()
    plt.tight_layout()
    # plt.show(block=False)

    return R_est


def run_multitarget_demo(fs, targets_m, amps, snr_db=20,
                         T=100e-3, f0=4e3, f1=16e3,
                         windows=('rect', 'hamming')):
    # Несколько целей с разными амплитудами.
    # Слабая цель маскируется боковыми лепестками сильной — окно вытаскивает.
    tx = make_signal('lfm', fs=fs, f0=f0, f1=f1, T=T)
    N = len(tx)

    # Сигнал длиной «до самой дальней цели + длительность чирпа»
    max_n = max(int(round(2 * R / C_LIGHT * fs)) for R in targets_m)
    rx = np.zeros(N + max_n + 100, dtype=complex)

    rng = np.random.default_rng(0)
    for R, a in zip(targets_m, amps):
        n_delay = int(round(2 * R / C_LIGHT * fs))
        phase = rng.uniform(0, 2 * np.pi)
        rx[n_delay:n_delay + N] += a * np.exp(1j * phase) * tx

    # Шум
    rx = apply_awgn(rx, snr_db, seed=0)

    # По одному MF на каждое окно
    fig, axes = plt.subplots(1, len(windows), figsize=(6 * len(windows), 5))
    if len(windows) == 1:
        axes = [axes]

    for ax, wname in zip(axes, windows):
        mf = matched_filter(rx, tx, window_name=wname)
        R_axis = mf_to_range_axis(len(mf), N, fs)
        mf_db = 20 * np.log10(np.abs(mf) / np.abs(mf).max() + 1e-12)

        ax.plot(R_axis / 1e3, mf_db)
        for R, a in zip(targets_m, amps):
            ax.axvline(R / 1e3, color='r', linestyle='--', alpha=0.5,
                       label=f'{R/1e3:.0f} км, амп={a}')
        ax.set_xlabel('Дальность, км'); ax.set_ylabel('|MF|, дБ')
        ax.set_title(f'Окно: {wname}')
        ax.set_ylim(-100, 5); ax.grid(alpha=0.3)
        ax.set_xlim(min(targets_m) / 1e3 - 15, max(targets_m) / 1e3 + 15)
        ax.legend(fontsize=7)

    plt.tight_layout()
    # plt.show(block=False)


# ============================================================
# ЗАДЕЛ НА БУДУЩЕЕ
# ============================================================
# 
#   run_doppler_demo(fs, R, v, ...)  — оценка скорости по доплеру
#   run_2d_map(fs, R, v, ...)        — доплеровская карта (delay × Doppler)
#   run_multipath_demo(...)          — многолучёвость
# Всё уже есть: apply_doppler, apply_delay, apply_awgn.